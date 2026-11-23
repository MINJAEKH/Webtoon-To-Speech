"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : entity parquet과 이미지를 Milvus/MinIO에 동기화하는 파이프라인 마지막 단계
 *               Milvus에는 text chunk, sparse text, dense embedding, page metadata, labels를 적재하고
 *               MinIO에는 문서 이미지 파일을 업로드/삭제한다.
 *               full_build, 증분 sync, Milvus만 재구축하는 rebuild 모드를 지원한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *    2       2026.11.02   soyeone.park        NERP&SEGAM 제거 및 GPPM용 수정
 *********** ************ ************ ***************************************************** ************* *************
"""

import os
import time
import json
import shutil
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
from pymilvus import MilvusClient

from src.components.database.milvus_schema import MilvusSchemaBuilder
from src.utils.utils import measure_time
from src.utils.label_utils import normalize_labels_for_milvus

logger = logging.getLogger(__name__)

class MilvusMinioSync:
    """
    Milvus 벡터 DB와 MinIO(S3 호환 객체 스토리지)를 동기화하는 클래스.

    이 클래스는 embedder가 생성한 entity parquet을 읽어 Milvus에 insert하고,
    다운로드된 이미지 파일을 MinIO에 업로드한다.

    처리 대상:
        - Milvus:
            text chunk, sparse text, dense embedding, page metadata, labels
        - MinIO:
            Confluence 문서에 포함된 이미지 파일

    실행 모드:
        1. full_build:
            Milvus collection이 없을 때 새 collection을 만들고 전체 entity/image를 적재한다.
        2. full_rebuild:
            run_sync_db_all=True일 때 기존 collection과 MinIO prefix를 초기화한 뒤 전체 재적재한다.
        3. incremental:
            변경된 source만 Milvus에서 삭제 후 재삽입하고,
            created/updated page의 이미지만 MinIO에 재동기화한다.

    MinIO 저장 구조:
        bucket_name/
        └── collection_name/
            └── hash_dir/
                └── images/
                    └── file

    주의:
        - Milvus는 source URL 단위로 기존 entity를 삭제한 뒤 새 entity를 삽입한다.
        - MinIO는 page_id를 기준으로 기존 이미지를 삭제/재업로드한다.
        - moved/ancestor/label 변경은 Milvus metadata에는 반영되지만 이미지 재업로드 대상은 아니다.
    """

    def __init__(
        self,
        milvus_batch_size,
        collection_name,
        db_source_dir,
        milvus_uri,
        text_chunk_max_length,
        embedding_dimension,
        n_local_dense_embeddings,
        minio_uri,
        bucket_name,
        local_image_dir,
        download_dir,
        minio_access_key,
        minio_secret_key,
        # secure_connection: bool = False,
    ):
        """
        Milvus client, MinIO client, 동기화 경로와 schema 설정을 초기화한다.

        Args:
            milvus_batch_size: Milvus insert batch 크기.
            collection_name: Milvus collection 이름이자 MinIO prefix 이름.
            db_source_dir: entity_batch_*.parquet 파일이 저장된 디렉터리.
            milvus_uri: Milvus 접속 URI.
            text_chunk_max_length: Milvus varchar 필드 길이 제한에 맞춘 text 최대 byte 기준.
            embedding_dimension: dense embedding 차원 수.
            n_local_dense_embeddings: local dense embedding 필드 개수.
            minio_uri: MinIO endpoint URL.
            bucket_name: MinIO bucket 이름.
            local_image_dir: MinIO 업로드 전 이미지 staging 디렉터리.
            download_dir: Confluence downloader가 저장한 원본 다운로드 디렉터리.
            minio_access_key: MinIO access key.
            minio_secret_key: MinIO secret key.
        """
        self.milvus_batch_size = milvus_batch_size or 512
        self.collection_name = collection_name
        self.db_source_dir = db_source_dir
        self.milvus_uri = milvus_uri
        self.text_chunk_max_length = text_chunk_max_length
        self.embedding_dimension = embedding_dimension
        self.n_local_dense_embeddings = n_local_dense_embeddings
        self.minio_uri = minio_uri
        self.bucket_name = bucket_name
        self.local_image_dir = local_image_dir
        self.download_dir = download_dir
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        # self.secure_connection = secure_connection

        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.local_image_directory = os.path.join(self.local_image_dir, self.collection_name)

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.minio_uri,
            aws_access_key_id=self.minio_access_key,
            aws_secret_access_key=self.minio_secret_key,
            config=Config(signature_version="s3v4"),
            verify=False,
        )

    # ==========================================================================
    # Main
    # ==========================================================================
    @measure_time
    def run_sync(self, changed_urls, run_sync_db_all, source_baseurl):
        """
        changed_urls와 실행 모드에 따라 Milvus와 MinIO를 동기화한다.

        전체 흐름:
            1. 실행 모드 결정
                - collection 없음: full_build
                - run_sync_db_all=True: full_rebuild
                - 그 외: incremental
            2. MinIO bucket과 collection prefix 상태 확인
            3. Milvus 삭제 대상 source 선삭제
            4. entity parquet 순회 후 Milvus insert
            5. download_dir 이미지를 local_image_dir로 복사
            6. local_image_dir 기준으로 MinIO 업로드
            7. incremental인 경우 local-MinIO 정합성 보정 수행

        Args:
            changed_urls:
                변경 유형별 page_id 목록.
            run_sync_db_all:
                True이면 기존 Milvus collection과 MinIO prefix를 초기화하고 전체 재구성한다.
            source_baseurl:
                page_id를 source URL로 변환할 때 사용하는 base URL.
        """
        chunk_count = 0

        # ----------------------------------------------------------------------
        # Dashboard parsing용 로그: 유지
        # ----------------------------------------------------------------------
        logger.info(f"[Milvus] updated sources: {changed_urls.get('updated', [])}")
        logger.info(f"[Milvus] created sources: {changed_urls.get('created', [])}")
        logger.info(f"[Milvus] moved sources: {changed_urls.get('moved', [])}")
        logger.info(
            f"[Milvus] ancestor_path_changed sources: {changed_urls.get('ancestor_path_changed', [])}"
        )
        logger.info(
            f"[Milvus] ancestor_title_changed sources: {changed_urls.get('ancestor_title_changed', [])}"
        )
        logger.info(
            f"[Milvus] structure_changed sources: {changed_urls.get('structure_changed', [])}"
        )
        logger.info(f"[Milvus] deleted sources: {changed_urls.get('deleted', [])}")
        
        structure_patch_ids = list(
            set(
                changed_urls.get("moved", [])
                + changed_urls.get("ancestor_path_changed", [])
                + changed_urls.get("ancestor_title_changed", [])
                + changed_urls.get("structure_changed", [])
            )
        )
        logger.info(f"[Milvus] structure_patch_ids sources: {structure_patch_ids}")
        logger.info(f"[Milvus] labels_changed sources: {changed_urls.get('labels_changed', [])}")

        mode = self._decide_run_mode(run_sync_db_all)
        logger.info(f"[run_insert_for_batches] 실행 모드: {mode}")

        # MinIO bucket 확인/생성
        self._ensure_bucket()

        # collection prefix 존재 여부 확인
        minio_prefix_exists = self._collection_prefix_exists_in_minio()
        logger.info(
            f"[run_insert_for_batches] MinIO collection prefix 상태: "
            f"bucket='{self.bucket_name}', prefix='{self.collection_name}/', exists={minio_prefix_exists}"
        )

        if mode == "full_build":
            # Milvus collection이 없는 최초 DB 구축 모드.
            # collection을 새로 만들고, MinIO prefix가 이미 있으면 기존 이미지를 정리한다.
            logger.info(
                f"[run_insert_for_batches] Milvus 컬렉션 '{self.collection_name}'이 존재하지 않아 full build를 수행합니다."
            )
            self.create_collection()
            if minio_prefix_exists:
                self._purge_collection_prefix_in_minio()
            full_insert = True
            full_image_sync = True

        elif mode == "full_rebuild":
            # 강제 전체 재구축 모드.
            # 기존 Milvus collection을 drop/recreate하고 MinIO collection prefix도 초기화한다.
            logger.info(
                f"[run_insert_for_batches] run_sync_db_all=True 이므로 "
                f"Milvus 컬렉션 '{self.collection_name}'과 MinIO collection prefix를 재구성합니다."
            )
            self.create_collection()
            if minio_prefix_exists:
                self._purge_collection_prefix_in_minio()
            full_insert = True
            full_image_sync = True

        else:
            # 일반 증분 업데이트 모드.
            # Milvus는 변경 source만 교체하고, MinIO는 이미지 변경 page만 재동기화한다.
            logger.info(
                f"[run_insert_for_batches] Milvus 컬렉션 '{self.collection_name}'에 대해 incremental update를 수행합니다."
            )
            full_insert = False
            # collection은 있는데 MinIO 폴더가 없으면 이미지 쪽  전체 다운로드 더 안전
            full_image_sync = not minio_prefix_exists
            if full_image_sync:
                logger.warning(
                    f"[run_insert_for_batches] MinIO collection prefix '{self.collection_name}/'가 없어 "
                    f"이미지 sync는 전체 기준으로 수행합니다."
                )

        milvus_changed_page_ids = set(
            map(
                str,
                changed_urls.get("updated", [])
                + changed_urls.get("created", [])
                + changed_urls.get("moved", [])
                + changed_urls.get("ancestor_path_changed", [])
                + changed_urls.get("ancestor_title_changed", [])
                + changed_urls.get("labels_changed", []),
            )
        )
        
        image_changed_page_ids = set(
            map(
                str,
                changed_urls.get("updated", [])
                + changed_urls.get("created", []),
            )
        )
        
        changed_sources = [
            source_baseurl + page_id
            for page_id in sorted(milvus_changed_page_ids)
        ]
        
        logger.info("[Milvus] milvus_changed_page_ids: %s", sorted(milvus_changed_page_ids))
        logger.info("[MinIO] image_changed_page_ids: %s", sorted(image_changed_page_ids))
        # deleted는 먼저 Milvus에서 제거
        deleted_sources = [
            source_baseurl + str(page_id) for page_id in changed_urls.get("deleted", [])
        ]
        if deleted_sources and not full_insert:
            logger.info(
                f"[run_insert_for_batches] 삭제 대상 source {len(deleted_sources)}개를 먼저 Milvus에서 제거합니다."
            )
            for deleted_source in deleted_sources:
                try:
                    res = self.milvus_client.delete(
                        collection_name=self.collection_name,
                        filter=f"source like '%{deleted_source}%'",
                    )
                    logger.info(
                        f"Delete {res['delete_count']} entities from source {deleted_source}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[run_insert_for_batches] 삭제 source 처리 실패: {deleted_source} | {e}"
                    )
        print(os.listdir(self.db_source_dir))

        batch_sources = sorted(
            os.listdir(self.db_source_dir),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        logger.info("[run_insert_for_batches] parquet 파일을 순회하며 Milvus insert를 수행합니다.")

        for source_filename in batch_sources:
            source_filepath = os.path.join(self.db_source_dir, source_filename)
            if not source_filepath.endswith(".parquet"):
                continue

            db_source_df = pd.read_parquet(source_filepath)

            # 전체 재적재가 아니면 변경된 source만 넣기
            if not full_insert:
                before_count = len(db_source_df)
                db_source_df = db_source_df[db_source_df["source"].isin(changed_sources)]
                after_count = len(db_source_df)
                logger.info(
                    f"[run_insert_for_batches] source 필터링 결과: {source_filename} | "
                    f"{before_count} -> {after_count}"
                )

            database_entity_list = db_source_df.to_dict("records")
            chunk_count += len(database_entity_list)

            logger.info("=" * 80)
            logger.info(
                f"[run_insert_for_batches] Milvus insert 대상 parquet 파일 처리 시작: {source_filepath}"
            )
            self.insert_to_milvus(database_entity_list, full_insert)

        logger.info(
            f"[run_insert_for_batches] {self.collection_name} 컬렉션에 대해 "
            f"이번 실행에서 insert 대상으로 선택된 총 entity 수: {chunk_count}개"
        )
        # logger.info(f"{self.collection_name}에 INSERT된 총 문서 : {chunk_count}개")

        logger.info("------------- save_local_images() 시작: 로컬 이미지 준비 단계 -------------")
        if full_image_sync:
            self.save_local_images(target_page_ids=None)
        else:
            self.save_local_images(target_page_ids=image_changed_page_ids)

        logger.info("------------- insert_to_minio() 시작: MinIO 이미지 동기화 단계 -------------")
        deleted_ids = set(map(str, changed_urls.get("deleted", [])))
        
        self.insert_to_minio(
            changed_page_ids=image_changed_page_ids,
            deleted_ids=deleted_ids,
            full_image_sync=full_image_sync,
        )

    def _decide_run_mode(self, run_sync_db_all: bool) -> str:
        """
        Milvus collection 존재 여부와 config 값을 기준으로 DB sync 모드를 결정한다.

        Returns:
            str:
                - full_build: collection이 없는 최초 구축
                - full_rebuild: run_sync_db_all=True 강제 재구축
                - incremental: 기존 collection에 대한 증분 갱신
        """
        collection_exists = self.milvus_client.has_collection(collection_name=self.collection_name)

        if not collection_exists:
            return "full_build"
        if run_sync_db_all:
            return "full_rebuild"
        return "incremental"

    # ==========================================================================
    # Collection
    # ==========================================================================
    def create_collection(self):
        """
        Milvus collection을 생성한다.

        기존 collection이 있으면 drop 후 새로 만든다.
        schema와 index는 MilvusSchemaBuilder에서 생성한다.
        """

        
        if self.milvus_client.has_collection(collection_name=self.collection_name):
            self.milvus_client.drop_collection(collection_name=self.collection_name)
            logger.info(f"기존 Milvus 컬렉션 '{self.collection_name}' 삭제 완료")

        logger.info(f"Milvus 컬렉션 '{self.collection_name}' 생성 시작")
        schema_builder = MilvusSchemaBuilder(self.milvus_client)

        schema = schema_builder.build_schema(
            text_chunk_max_length=self.text_chunk_max_length,
            embedding_dimension=self.embedding_dimension,
            n_local_dense_embeddings=self.n_local_dense_embeddings,
        )

        index_params = schema_builder.build_index(
            n_local_dense_embeddings=self.n_local_dense_embeddings,

        )

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Milvus 컬렉션 '{self.collection_name}' 생성 완료")

    # ==========================================================================
    # Milvus Insert
    # ==========================================================================
    def insert_to_milvus(
        self,
        entity_list: List[Dict],
        full_insert: bool,
    ):
        """
        entity list를 Milvus collection에 batch 단위로 삽입한다.

        incremental 모드에서는 insert 전에 동일 source의 기존 entity를 먼저 삭제한다.
        이렇게 해야 updated/metadata-only source의 이전 chunk가 중복으로 남지 않는다.

        Args:
            entity_list: Milvus schema에 맞춰 저장할 entity dict 목록.
            full_insert: True이면 기존 source 삭제 없이 전체 insert만 수행한다.
        """
        
        logger.info(f"Milvus 컬렉션 '{self.collection_name}'에 대한 insert 작업을 시작합니다.")

        total_count = len(entity_list)
        if total_count > 0:
            logger.info(
                f"총 {total_count}개의 entity를 '{self.collection_name}'에 삽입합니다. "
                f"(배치 크기: {self.milvus_batch_size})"
            )
        else:
            logger.info("Milvus에 삽입할 대상 entity가 없습니다. total_count = 0")
            return

        if not full_insert:
            updated_sources = list(set(entity["source"] for entity in entity_list))
            logger.info(
                f"[insert_to_milvus] 변경된 source {len(updated_sources)}개에 대해 기존 entity를 먼저 제거합니다."
            )

            for source in updated_sources:
                try:
                    res = self.milvus_client.delete(
                        collection_name=self.collection_name,
                        filter=f"source like '%{source}%'",
                    )
                    logger.info(
                        f"Delete {res['delete_count']} entities from source {source}"
                    )
                except Exception:
                    logger.warning(f"기존 source 삭제 실패: {source}")

        processed_count = 0
        success_count = 0
        failed_batches = []
        start_time = time.time()

        for i in range(0, total_count, self.milvus_batch_size):
            batch = entity_list[i : i + self.milvus_batch_size]
            batch_num = i // self.milvus_batch_size + 1
            total_batches = (total_count + self.milvus_batch_size - 1) // self.milvus_batch_size

            try:
                cleaned_batch = self._prepare_batch_data(batch)

                self.milvus_client.insert(
                    collection_name=self.collection_name,
                    data=cleaned_batch,
                )

                cleaned_batch_sources = [chunk["source"] for chunk in cleaned_batch]
                for unique_source in list(set(cleaned_batch_sources)):
                    logger.info(
                        f"Insert {cleaned_batch_sources.count(unique_source)} entity from source {unique_source}"
                    )
                logger.info(f"Insert {len(cleaned_batch)} entities")

                batch_success = len(batch)
                processed_count += batch_success
                success_count += batch_success

                elapsed_time = time.time() - start_time
                speed = processed_count / elapsed_time if elapsed_time > 0 else 0

                logger.info(
                    f"배치 [{batch_num}/{total_batches}]: "
                    f"{processed_count}/{total_count} "
                    f"({processed_count / total_count * 100:.1f}%) | "
                    f"속도: {speed:.0f} entities/sec | "
                    f"경과: {elapsed_time:.1f}초"
                )

                if batch_num % 10 == 0:
                    self.milvus_client.flush(collection_name=self.collection_name)
                    logger.debug("Milvus 중간 flush 수행 완료")        
            except Exception as e:
                    failed_batches.append(batch_num)
                    logger.error(f"Milvus 배치 #{batch_num} 삽입 실패: {e}")
    
                    cleaned_batch = self._prepare_batch_data(batch)
                    individual_result = self._insert_each_entity_to_milvus(cleaned_batch)
                    processed_count += individual_result["total"]
                    success_count += individual_result["success"]
    
                    logger.info(
                        f"개별 삽입 재시도 완료: {individual_result['success']}/{len(batch)}건 성공"
                    )

        logger.info("Milvus 최종 flush 수행 중...")
        self.milvus_client.flush(collection_name=self.collection_name)

        self._log_insert_summary(
            success_count,
            total_count,
            start_time,
            failed_batches,
        )

    def _truncate_str_by_bytes(self, value, max_bytes: int) -> str:
        """
        문자열을 UTF-8 byte 기준 max_bytes 이하로 자른다.

        Milvus varchar 계열 필드는 byte 길이 제한이 있으므로,
        멀티바이트 한글/특수문자가 중간에서 잘리지 않도록 decode 가능한 지점까지 줄인다.
        """
        if value is None:
            return ""
    
        value = str(value)
        encoded = value.encode("utf-8")
    
        if len(encoded) <= max_bytes:
            return value
    
        truncated = encoded[:max_bytes]
    
        # 멀티바이트 문자 중간이 잘릴 수 있으니 안전하게 복구
        while True:
            try:
                return truncated.decode("utf-8")
            except UnicodeDecodeError:
                truncated = truncated[:-1]

   
    
    def _prepare_batch_data(self, batch: List[Dict]) -> List[Dict]:
        """
        Milvus insert 전에 entity batch를 schema에 맞게 정리한다.

        처리 내용:
            - id 제거: Milvus auto id 또는 schema 충돌 방지
            - 누락 필드 기본값 채우기
            - varchar 필드 byte 길이 제한 적용
            - labels를 Milvus JSON field용 dict로 정규화
            - business_area 기본값/길이 제한 처리

        Args:
            batch: parquet에서 읽은 raw entity dict 목록.

        Returns:
            List[Dict]:
                Milvus insert 가능한 형태로 정리된 entity 목록.
        """
        cleaned_batch = []
    
        for entity in batch:
            cleaned_entity = entity.copy()
    
            cleaned_entity.pop("id", None)
    
            cleaned_entity.setdefault("parent_id", -1)
            cleaned_entity.setdefault("child_id", -1)
            cleaned_entity.setdefault("include_image", False)
            cleaned_entity.setdefault("has_parent", False)
            cleaned_entity.setdefault("parent_page", [])
            cleaned_entity.setdefault("page", [])
            cleaned_entity.setdefault("multi_source", [])
            cleaned_entity.setdefault("image_info", [])
            cleaned_entity["parent_text"] = self._truncate_str_by_bytes(
                cleaned_entity.get("parent_text", ""), self.text_chunk_max_length
            )
            cleaned_entity["text_for_sparse"] = self._truncate_str_by_bytes(
                cleaned_entity.get("text_for_sparse", ""), self.text_chunk_max_length * 2
            )

            cleaned_entity["labels"] = normalize_labels_for_milvus(cleaned_entity.get("labels"))
            # 예전 파이프라인이 남긴 NERP 전용 컬럼은 schema에 없으므로 dynamic field로 새지 않게 제거한다
            for legacy_key in ("module", "nerp_tcode", "program_tcode", "is_program"):
                cleaned_entity.pop(legacy_key, None)
            cleaned_entity["business_area"] = self._truncate_str_by_bytes(
                cleaned_entity.get("business_area", "") or "", 50
            )

            cleaned_batch.append(cleaned_entity)
    
        return cleaned_batch

    def _insert_each_entity_to_milvus(self, cleaned_batch: List[Dict]) -> Dict[str, int]:
        """
        batch insert 실패 시 entity를 1개씩 Milvus에 삽입해 재시도한다.

        목적:
            batch 전체 실패 상황에서 문제 없는 entity까지 모두 버리지 않기 위함이다.
            개별 insert 실패 로그를 통해 schema/type 오류가 있는 entity를 추적할 수 있다.

        Returns:
            Dict[str, int]:
                success, failed, total 개수.
        """
        logger.info(f"Milvus 개별 삽입 재시도를 시작합니다. (총 {len(cleaned_batch)}개)")
        individual_success = 0
        individual_failed = 0

        for idx, entity in enumerate(cleaned_batch):
            try:
                self.milvus_client.insert(
                    collection_name=self.collection_name,
                    data=[entity],
                )
                individual_success += 1

                if (idx + 1) % 50 == 0:
                    logger.debug(
                        f"Milvus 개별 삽입 진행 중... ({individual_success}/{len(cleaned_batch)})"
                    )

            except Exception as e:
                individual_failed += 1
                if individual_failed <= 5:
                    logger.warning(
                        f"Milvus 개별 삽입 실패 ({individual_failed}건째): {str(e)[:200]}"
                    )

        return {
            "success": individual_success,
            "failed": individual_failed,
            "total": individual_success + individual_failed,
        }

    def _log_insert_summary(
        self,
        success_count: int,
        total_count: int,
        start_time: float,
        failed_batches: List[int],
    ):

        """
        Milvus insert 결과 요약 로그를 출력한다.

        Args:
            success_count: 성공 entity 수.
            total_count: insert 대상 전체 entity 수.
            start_time: insert 시작 시각.
            failed_batches: batch insert 실패가 발생한 batch 번호 목록.
        """
        
        total_time = time.time() - start_time
        failed_count = total_count - success_count

        logger.info("=" * 80)
        logger.info("삽입 완료")
        logger.info("=" * 80)
        logger.info(
            f"성공: {success_count}/{total_count} ({success_count / total_count * 100:.1f}%)"
        )

        if failed_count > 0:
            logger.warning(
                f"실패: {failed_count}/{total_count} ({failed_count / total_count * 100:.1f}%)"
            )
            failed_batch_str = ", ".join(map(str, failed_batches[:10]))
            if len(failed_batches) > 10:
                failed_batch_str += "..."
            logger.warning(f"실패한 배치: [{failed_batch_str}]")

        logger.info(f"소요 시간: {total_time:.2f}초")

        if total_time > 0:
            logger.info(f"평균 속도: {success_count / total_time:.2f} entities/sec")

    # ==========================================================================
    # Local Image
    # ==========================================================================
    @measure_time
    def save_local_images(self, target_page_ids: Optional[Set[str]] = None):
        """
        download_dir에 있는 이미지 파일을 local_image_directory로 복사한다.

        local_image_directory는 MinIO 업로드 전 staging 영역이다.
        download_dir은 parser 산출물 구조이고, MinIO 업로드는 collection_name 기준 구조를 사용하므로
        이 단계에서 이미지를 collection별 디렉터리로 정리한다.

        Args:
            target_page_ids:
                None이면 전체 이미지 복사.
                set이면 해당 page_id에 해당하는 이미지만 복사.
        """
        copied_count = 0
        error_count = 0
        skip_count = 0

        os.makedirs(self.local_image_directory, exist_ok=True)

        for root, _, files in os.walk(self.download_dir):
            if os.path.basename(root) == "images":
                relative_dir = os.path.relpath(root, self.download_dir)
                path_parts = relative_dir.split(os.sep)

                if len(path_parts) >= 2:
                    first_hash = path_parts[0]
                    page_id = self._extract_page_id_from_hash(first_hash)

                    if target_page_ids is not None and page_id not in target_page_ids:
                        skip_count += len(files)
                        continue

                    target_dir = os.path.join(
                        self.local_image_directory, first_hash, "images"
                    )
                    os.makedirs(target_dir, exist_ok=True)

                    for file in files:
                        local_file_path = os.path.join(root, file)

                        try:
                            target_file_path = os.path.join(target_dir, file)
                            shutil.copy(local_file_path, target_file_path)
                            copied_count += 1
                        except Exception:
                            error_count += 1
                else:
                    logger.warning(
                        f"복사 건너뜀: {root} - 경로 구조가 올바르지 않음 (최소 hash1/images 필요)"
                    )
                    skip_count += len(files)

        logger.info(
            f"[save_local_images] 로컬 이미지 복사 결과: "
            f"복사 성공={copied_count}, 복사 실패={error_count}, 건너뜀={skip_count}"
        )

    # ==========================================================================
    # MinIO
    # ==========================================================================
    @measure_time
    def insert_to_minio(self, changed_page_ids: Set[str], deleted_ids: Set[str], full_image_sync: bool = False):
        """
        local_image_directory의 이미지를 MinIO에 업로드한다.

        이미지 반영 기준:
            - 업로드/재업로드: created + updated
            - 삭제: 업로드/재업로드 대상 + deleted
            - structure_changed / moved / label 변경: 이미지 반영 대상 아님

        Args:
            changed_page_ids:
                이미지가 새로 다운로드되었거나 바뀐 page id 목록.
            deleted_ids:
                삭제된 page id 목록.
            full_image_sync:
                True이면 전체 이미지 업로드.
                False이면 changed_page_ids만 업로드하고 삭제 대상만 정리.
        """
        image_sync_ids = changed_page_ids
        delete_target_ids = image_sync_ids | deleted_ids

        try:
            self._ensure_bucket()

            if full_image_sync:
                logger.info(
                    f"[insert_to_minio] 전체 이미지 sync를 수행합니다. "
                    f"(bucket='{self.bucket_name}', prefix='{self.collection_name}/')"
                )
            else:
                logger.info(
                    f"[insert_to_minio] incremental image sync 시작: "
                    f"업로드/재업로드 대상 page={len(image_sync_ids)}, 삭제 대상 page={len(deleted_ids)}"
                )
                self._delete_existing_collection_objects(delete_target_ids)

        except ClientError as e:
            logger.error(f"MinIO 버킷 확인 또는 기존 객체 삭제 중 오류 발생: {e}")
            raise

        if not os.path.isdir(self.local_image_directory):
            error_msg = (
                f"MinIO 업로드 대상 로컬 디렉토리 '{self.local_image_directory}'를 찾을 수 없습니다."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(
            f"로컬 이미지 디렉토리 '{self.local_image_directory}' 기준으로 MinIO 업로드를 시작합니다..."
        )

        upload_count = 0
        error_count = 0
        skip_count = 0

        for root, _, files in os.walk(self.local_image_directory):
            if os.path.basename(root) == "images":
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_dir = os.path.relpath(root, self.local_image_directory)
                    path_parts = relative_dir.split(os.sep)

                    if len(path_parts) >= 2:
                        first_hash = path_parts[0]
                        object_name = f"{self.collection_name}/{first_hash}/images/{file}"

                        try:
                            page_id = self._extract_page_id_from_hash(first_hash)

                            if full_image_sync or page_id in image_sync_ids:
                                self.s3_client.upload_file(
                                    local_file_path,
                                    self.bucket_name,
                                    object_name,
                                )
                                upload_count += 1
                                logger.debug(
                                    f"업로드 성공: {local_file_path} -> {object_name}"
                                )
                            else:
                                skip_count += 1

                        except ClientError as e:
                            error_count += 1
                            logger.error(f"MinIO 업로드 실패: {local_file_path} - {e}")
                    else:
                        skip_count += 1
                        logger.warning(
                            f"업로드 건너뜀: '{local_file_path}' - 경로 구조가 올바르지 않음 (최소 hash1/images 필요)"
                        )

        logger.info("=" * 80)
        logger.info(
            f"MinIO 이미지 업로드 완료: 업로드 성공={upload_count}개, "
            f"업로드 실패={error_count}개, 업로드 제외={skip_count}개"
        )

        if not full_image_sync:
            logger.info("[insert_to_minio] local-MinIO 정합성 보정(reconciliation)을 수행합니다.")
            self._reconcile_minio_with_local(deleted_ids=deleted_ids)

    
    def _ensure_bucket(self):
        """
        MinIO bucket이 존재하는지 확인하고, 없으면 생성한다.
        """

        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"MinIO 버킷 '{self.bucket_name}' 확인 완료")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"MinIO 버킷 '{self.bucket_name}' 생성 완료")
            else:
                raise

    def _collection_prefix_exists_in_minio(self) -> bool:
        """
        MinIO에 collection_name prefix 아래 객체가 하나라도 있는지 확인한다.

        Returns:
            bool:
                collection_name/ prefix에 객체가 있으면 True.
        """
        prefix = f"{self.collection_name}/"
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix,
            MaxKeys=1,
        )
        return "Contents" in response and len(response["Contents"]) > 0

        
    def _purge_collection_prefix_in_minio(self):
        """
        MinIO에서 collection_name prefix 아래 모든 객체를 삭제한다.

        full_rebuild 또는 full_build 전 기존 이미지 객체를 초기화할 때 사용한다.
        """
        prefix_to_delete = f"{self.collection_name}/"
        logger.info(
            f"컬렉션 '{self.collection_name}'의 전체 MinIO prefix 초기화를 위해 기존 객체를 확인합니다."
        )

        try:
            delete_object_list = []
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix_to_delete):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        delete_object_list.append({"Key": obj["Key"]})

            if delete_object_list:
                logger.info(
                    f"컬렉션 '{self.collection_name}'의 기존 MinIO 객체 {len(delete_object_list)}개를 전체 삭제합니다..."
                )
                self._delete_object_list(delete_object_list)
                logger.info(
                    f"컬렉션 '{self.collection_name}'의 전체 MinIO prefix 초기화 완료"
                )
            else:
                logger.info(
                    f"컬렉션 '{self.collection_name}'에 초기화할 기존 MinIO 객체가 없습니다."
                )

        except ClientError as e:
            logger.error(f"MinIO prefix 초기화 중 오류 발생: {e}")
            raise

            
    def _delete_existing_collection_objects(self, changed_ids):
        """
        collection prefix 아래에서 특정 page_id에 해당하는 MinIO 객체를 삭제한다.

        Args:
            changed_ids:
                기존 이미지를 제거해야 하는 page_id 집합.
                created/updated 재업로드 대상과 deleted 대상이 들어온다.
        """

        changed_ids = set(map(str, changed_ids or []))
        prefix_to_delete = f"{self.collection_name}/"

        logger.info(
            f"컬렉션 '{self.collection_name}'의 삭제 대상 MinIO 객체를 확인합니다."
        )

        try:
            delete_object_list = []
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix_to_delete):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        page_id = self._extract_page_id_from_object_key(obj["Key"])
                        if page_id is not None and page_id in changed_ids:
                            delete_object_list.append({"Key": obj["Key"]})

            if delete_object_list:
                logger.info(
                    f"컬렉션 '{self.collection_name}'에서 삭제 대상 MinIO 객체 {len(delete_object_list)}개를 삭제합니다..."
                )
                self._delete_object_list(delete_object_list)
                logger.info(
                    f"컬렉션 '{self.collection_name}'의 삭제 대상 MinIO 객체 정리 완료"
                )
            else:
                logger.info(
                    f"컬렉션 '{self.collection_name}'에서 삭제할 MinIO 객체가 없습니다."
                )

        except ClientError as e:
            logger.error(f"객체 삭제 중 오류 발생: {e}")
            raise

    
    def _delete_object_list(self, delete_object_list: List[Dict]):
        """
        MinIO 객체 key 목록을 순회하며 삭제한다.

        boto3 delete_objects batch API 대신 개별 delete를 사용해
        실패 객체를 로그로 명확히 남긴다.
        """
        total_obj = len(delete_object_list)
        error_cnt = 0

        for image_object in delete_object_list:
            try:
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=image_object["Key"],
                )
            except ClientError as e:
                logger.error(
                    f"MinIO 객체 삭제 오류: {image_object['Key']} - {e}"
                )
                error_cnt += 1

        logger.info(
            f"MinIO 객체 삭제 완료: 성공={total_obj - error_cnt}개 / 대상={total_obj}개"
        )

    # ==========================================================================
    # Reconciliation
    # ==========================================================================
    def _reconcile_minio_with_local(self, deleted_ids: Optional[Set[str]] = None):
        """
        incremental image sync 이후 local_image_directory와 MinIO 상태를 비교해 정합성을 보정한다.

        보정 기준:
            - local_only page:
                local에는 이미지가 있는데 MinIO에 없으면 업로드한다.
            - rebuild page:
                local과 MinIO 양쪽에 있지만 hash_dir 또는 파일 목록이 다르면 삭제 후 재업로드한다.
            - minio_only page:
                local에는 없고 MinIO에만 있으면 stale 객체일 수 있다.
                deleted_ids로 확인된 page만 자동 삭제하고, 그 외는 경고만 남긴다.

        안전장치:
            local 상태가 비어 있는데 MinIO 객체가 존재하는 경우,
            mount/copy 실패 가능성이 있으므로 자동 삭제하지 않는다.
        """
        deleted_ids = set(map(str, deleted_ids or set()))

        local_state = self._build_local_page_state()
        minio_state = self._build_minio_page_state()

        local_pages = set(local_state.keys())
        minio_pages = set(minio_state.keys())

        if not local_pages and minio_pages:
            logger.warning(
                "[reconcile] local 이미지 상태는 비어 있는데 MinIO 객체는 존재합니다. "
                "경로/mount/copy 이슈 가능성이 있어 자동 삭제를 수행하지 않습니다."
            )
            return

        upload_only_pages = local_pages - minio_pages
        minio_only_pages = minio_pages - local_pages
        common_pages = local_pages & minio_pages

        rebuild_pages = set()

        for page_id in common_pages:
            local_hash = local_state[page_id]["hash_dir"]
            minio_hash = minio_state[page_id]["hash_dir"]
            local_files = local_state[page_id]["files"]
            minio_files = minio_state[page_id]["files"]

            if local_hash != minio_hash or local_files != minio_files:
                rebuild_pages.add(page_id)

        logger.info(f"[debug][reconcile] sample local_only={list(sorted(upload_only_pages))[:20]}")
        logger.info(f"[debug][reconcile] sample minio_only={list(sorted(minio_only_pages))[:20]}")
        logger.info(f"[debug][reconcile] sample rebuild={list(sorted(rebuild_pages))[:20]}")

        logger.info(
            f"[reconcile] local_only={len(upload_only_pages)}, "
            f"minio_only={len(minio_only_pages)}, rebuild={len(rebuild_pages)}, "
            f"in_sync={len(common_pages) - len(rebuild_pages)}"
        )

        if upload_only_pages:
            logger.info(
                f"[reconcile] MinIO에 없는 local page {len(upload_only_pages)}개를 업로드합니다."
            )
            self._upload_pages_to_minio(upload_only_pages)

        if rebuild_pages:
            logger.info(
                f"[reconcile] local-MinIO 불일치 page {len(rebuild_pages)}개를 재동기화합니다."
            )
            self._delete_existing_collection_objects(rebuild_pages)
            self._upload_pages_to_minio(rebuild_pages)

        stale_deletable_pages = minio_only_pages & deleted_ids
        stale_warning_pages = minio_only_pages - deleted_ids

        if stale_deletable_pages:
            logger.info(
                f"[reconcile] deleted 이벤트로 확인된 MinIO-only page {len(stale_deletable_pages)}개를 삭제합니다."
            )
            self._delete_existing_collection_objects(stale_deletable_pages)

        if stale_warning_pages:
            logger.warning(
                f"[reconcile] local에는 없고 MinIO에만 있는 page {len(stale_warning_pages)}개를 발견했습니다. "
                "deleted 이벤트로 확인되지 않아 자동 삭제는 수행하지 않습니다."
            )

    def _upload_pages_to_minio(self, target_page_ids: Set[str]):
        """
        local_image_directory에서 target_page_ids에 해당하는 이미지만 MinIO에 업로드한다.

        reconciliation 단계에서 local_only 또는 rebuild page를 보정할 때 사용한다.
        """
        upload_count = 0
        error_count = 0
        skip_count = 0

        for root, _, files in os.walk(self.local_image_directory):
            if os.path.basename(root) == "images":
                if not files:
                    continue
                relative_dir = os.path.relpath(root, self.local_image_directory)
                path_parts = relative_dir.split(os.sep)

                if len(path_parts) < 2:
                    skip_count += len(files)
                    continue

                first_hash = path_parts[0]
                page_id = self._extract_page_id_from_hash(first_hash)

                if page_id not in target_page_ids:
                    skip_count += len(files)
                    continue

                for file in files:
                    local_file_path = os.path.join(root, file)
                    object_name = f"{self.collection_name}/{first_hash}/images/{file}"

                    try:
                        self.s3_client.upload_file(
                            local_file_path,
                            self.bucket_name,
                            object_name,
                        )
                        upload_count += 1
                    except ClientError as e:
                        error_count += 1
                        logger.error(f"MinIO 업로드 실패: {local_file_path} - {e}")

        logger.info(
            f"[reconcile] 보정 업로드 결과: 성공={upload_count}, 실패={error_count}, 제외={skip_count}"
        )

    
    def _build_local_page_state(self) -> Dict[str, Dict]:
        """
        local_image_directory를 page_id 기준 상태 dict로 변환한다.

        Returns:
            Dict[str, Dict]:
                {
                    page_id: {
                        "hash_dir": str,
                        "files": set[str],
                    }
                }

        빈 images 폴더는 실제 업로드 대상이 아니므로 제외한다.
        동일 page_id에 여러 hash_dir가 있으면 경고를 남긴다.
        """
        state = {}
        multiple_hash_map = defaultdict(set)
    
        if not os.path.isdir(self.local_image_directory):
            return state
    
        for root, _, files in os.walk(self.local_image_directory):
            if os.path.basename(root) != "images":
                continue
    
            # 빈 images 폴더는 local state에서 제외
            if not files:
                continue
    
            relative_dir = os.path.relpath(root, self.local_image_directory)
            path_parts = relative_dir.split(os.sep)
            if len(path_parts) < 2:
                continue
    
            first_hash = path_parts[0]
            page_id = self._extract_page_id_from_hash(first_hash)
            if page_id is None:
                continue
    
            multiple_hash_map[page_id].add(first_hash)
    
            if page_id not in state:
                state[page_id] = {
                    "hash_dir": first_hash,
                    "files": set(files),
                }
            else:
                state[page_id]["files"].update(files)
    
        duplicated = {k: v for k, v in multiple_hash_map.items() if len(v) > 1}
        if duplicated:
            logger.warning(
                f"[local_state] 동일 page_id에 여러 hash_dir가 존재합니다. count={len(duplicated)}"
            )
    
        logger.info(f"[local_state] 로컬 이미지 page 수집 완료: {len(state)}개")
        return state

    def _build_minio_page_state(self) -> Dict[str, Dict]:
        """
        MinIO collection prefix 아래 객체를 page_id 기준 상태 dict로 변환한다.

        Returns:
            Dict[str, Dict]:
                {
                    page_id: {
                        "hash_dir": str,
                        "files": set[str],
                    }
                }

        object key는 {collection_name}/{hash_dir}/images/{filename} 구조를 기대한다.
        """
        state = {}
        multiple_hash_map = defaultdict(set)
        prefix = f"{self.collection_name}/"

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                parsed = self._parse_object_key(obj["Key"])
                if parsed is None:
                    continue

                page_id, hash_dir, filename = parsed
                multiple_hash_map[page_id].add(hash_dir)

                if page_id not in state:
                    state[page_id] = {
                        "hash_dir": hash_dir,
                        "files": {filename},
                    }
                else:
                    state[page_id]["files"].add(filename)

        duplicated = {k: v for k, v in multiple_hash_map.items() if len(v) > 1}
        if duplicated:
            logger.warning(
                f"[minio_state] 동일 page_id에 여러 hash_dir가 존재합니다. count={len(duplicated)}"
            )

        logger.info(f"[minio_state] MinIO 이미지 page 수집 완료: {len(state)}개")
        return state

    # ==========================================================================
    # Helper
    # ==========================================================================

    def _extract_page_id_from_hash(self, hash_value: str) -> Optional[str]:
        """
        hash_dir 문자열에서 page_id를 추출한다.

        build_hashed_dir_name이 {hash}_{page_id} 형태를 만든다는 전제에 따라
        마지막 '_' 뒤 값을 page_id로 사용한다.
        """
        try:
            page_id = str(hash_value).split("_")[-1]
            return page_id if page_id else None
        except Exception:
            return None

    
    def _extract_page_id_from_object_key(self, object_key: str) -> Optional[str]:
        """
        MinIO object key에서 page_id만 추출한다.
        """        
        parsed = self._parse_object_key(object_key)
        if parsed is None:
            return None
        page_id, _, _ = parsed
        return page_id

    
    def _parse_object_key(self, object_key: str) -> Optional[Tuple[str, str, str]]:
        """
        MinIO object key를 page_id, hash_dir, filename으로 파싱한다.

        expected:
            {collection_name}/{hash_dir}/images/{filename}

        Returns:
            Optional[Tuple[str, str, str]]:
                (page_id, hash_dir, filename)
                형식이 맞지 않으면 None.
        """
        try:
            parts = object_key.split("/")
            if len(parts) < 4:
                return None

            collection = parts[0]
            hash_dir = parts[1]
            folder = parts[2]
            filename = parts[-1]

            if collection != self.collection_name or folder != "images":
                return None

            page_id = self._extract_page_id_from_hash(hash_dir)
            if page_id is None:
                return None

            return page_id, hash_dir, filename
        except Exception:
            return None

###############################
def rebuild_milvus_only(sync, collection_name, batch_folder):
    if sync.milvus_client.has_collection(collection_name):
        sync.milvus_client.drop_collection(collection_name)
        logger.info("[Milvus Only] dropped collection: %s", collection_name)

    sync.create_collection()

    batch_sources = sorted(
        os.listdir(batch_folder),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    total = 0

    for source_filename in batch_sources:
        if not source_filename.endswith(".parquet"):
            continue

        source_filepath = os.path.join(batch_folder, source_filename)
        df = pd.read_parquet(source_filepath)

        entity_list = df.to_dict("records")
        total += len(entity_list)

        logger.info(
            "[Milvus Only] insert file=%s rows=%s",
            source_filename,
            len(entity_list),
        )

        sync.insert_to_milvus(
            entity_list=entity_list,
            full_insert=True,
        )

    stats = sync.milvus_client.get_collection_stats(collection_name)
    logger.info("[Milvus Only] done. total=%s stats=%s", total, stats)

# def rebuild_milvus_only(sync, collection_name, batch_folder, target_page_ids=None):
#     target_sources = None

#     if target_page_ids is not None:
#         target_page_ids = set(map(str, target_page_ids))
#         target_sources = {
#             f"https://confluence.sec.samsung.net/pages/viewpage.action?pageId={page_id}"
#             for page_id in target_page_ids
#         }

#     if sync.milvus_client.has_collection(collection_name):
#         sync.milvus_client.drop_collection(collection_name)
#         logger.info("[Milvus Only] dropped collection: %s", collection_name)

#     sync.create_collection()

#     batch_sources = sorted(
#         [
#             x for x in os.listdir(batch_folder)
#             if x.endswith(".parquet")
#         ],
#         key=lambda x: int(x.split("_")[-1].split(".")[0]),
#     )

#     total = 0

#     for source_filename in batch_sources:
#         source_filepath = os.path.join(batch_folder, source_filename)
#         df = pd.read_parquet(source_filepath)

#         if target_sources is not None:
#             before = len(df)
#             df = df[df["source"].isin(target_sources)]
#             after = len(df)

#             if after == 0:
#                 continue

#             logger.info(
#                 "[Milvus Only Test] filter %s | %s -> %s",
#                 source_filename,
#                 before,
#                 after,
#             )

#         entity_list = df.to_dict("records")
#         total += len(entity_list)

#         logger.info(
#             "[Milvus Only] insert file=%s rows=%s",
#             source_filename,
#             len(entity_list),
#         )

#         sync.insert_to_milvus(
#             entity_list=entity_list,
#             full_insert=True,
#         )

#     stats = sync.milvus_client.get_collection_stats(collection_name)
#     logger.info("[Milvus Only] done. total=%s stats=%s", total, stats)

def main():
    import os
    import glob
    import logging
    import pandas as pd

    from src.utils.config import init_config

    logging.basicConfig(level=logging.INFO)

    # ==========================================================
    # 2. 새 collection 이름 지정
    # ==========================================================
    NEW_COLLECTION_NAME = "nerp_knowledge_space_v1_1_1"

    # ==========================================================
    # 3. entity parquet 경로 확인
    # ==========================================================
    batch_folder = "./pipeline_storage/embed_and_update_gppm/batch_folder_label_added"

    parquet_files = sorted(
        glob.glob(os.path.join(batch_folder, "entity_batch_*.parquet")),
        key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"entity parquet not found: {batch_folder}"
        )

    logger.info(
        "[BASE BUILD] entity parquet count = %s",
        len(parquet_files),
    )

    # ==========================================================
    # 4. 전체 source/page_id 수집
    # ==========================================================
    all_sources = set()

    for parquet_path in parquet_files:
        df = pd.read_parquet(
            parquet_path,
            columns=["source"],
        )

        all_sources.update(
            df["source"].dropna().astype(str).tolist()
        )

    all_page_ids = sorted(
        {
            source.split("=")[-1]
            for source in all_sources
        }
    )

    logger.info(
        "[BASE BUILD] total page ids = %s",
        len(all_page_ids),
    )

    # ==========================================================
    # 5. changed_urls 구성
    # ==========================================================
    changed_urls = {
        "created": all_page_ids,
        "updated": [],
        "deleted": [],
        "moved": [],
        "ancestor_path_changed": [],
        "ancestor_title_changed": [],
        "structure_changed": [],
        "labels_changed": [],
    }

    # ==========================================================
    # 6. Sync 객체 생성
    # ==========================================================
     
    sync = MilvusMinioSync(
        milvus_batch_size=512,
        collection_name=NEW_COLLECTION_NAME,
        db_source_dir=batch_folder,
        milvus_uri='http://10.40.110.82:19530',
        text_chunk_max_length=31744,
        embedding_dimension=1024,
        n_local_dense_embeddings=8,
        minio_uri='http://10.40.110.82:9000',
        bucket_name='dev-doxa-nerp-objectstorage',
        local_image_dir='./pipeline_storage/milvus_minio_sync/images_gppms',
        download_dir="./pipeline_storage/downloader/downloaded_gppm",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
    )

    # ==========================================================
    # 7. 전체 collection build 실행
    # ==========================================================
    logger.info("=" * 100)
    logger.info("[START] NEW COLLECTION BASE BUILD")
    logger.info("collection_name = %s", NEW_COLLECTION_NAME)
    logger.info("=" * 100)
    source_baseurl = 'https://confluence.sec.samsung.net/pages/viewpage.action?pageId='

    # sync.run_sync(
    #     changed_urls=changed_urls,
    #     run_sync_db_all=True,
    #     source_baseurl=source_baseurl,
    # )

    rebuild_milvus_only(
        sync=sync,
        collection_name=NEW_COLLECTION_NAME,
        batch_folder=batch_folder,
    )
    
    # ==========================================================
    # 8. 결과 확인
    # ==========================================================
    stats = sync.milvus_client.get_collection_stats(
        NEW_COLLECTION_NAME
    )

    logger.info("=" * 100)
    logger.info("[DONE] BASE BUILD COMPLETED")
    logger.info("collection = %s", NEW_COLLECTION_NAME)
    logger.info("stats = %s", stats)
    logger.info("=" * 100)

# def main():
#     import os
#     import glob
#     import logging
#     import pandas as pd

#     logging.basicConfig(level=logging.INFO)

#     TEST_COLLECTION_NAME = "nerp_knowledge_space_label_test"

#     batch_folder = "./pipeline_storage/embed_and_update_new_embedding/batch_folder"

#     TARGET_PAGE_IDS = [
#         "1299912270",  # biz=factory
#         "1299912324",  # region=os
#         "1299912412",  # corp 여러 개
#         "1299912398",
#         "1268764780"
#     ]

#     parquet_files = sorted(
#         glob.glob(os.path.join(batch_folder, "entity_batch_*.parquet")),
#         key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
#     )

#     if not parquet_files:
#         raise FileNotFoundError(f"entity parquet not found: {batch_folder}")

#     sync = MilvusMinioSync(
#         milvus_batch_size=512,
#         collection_name=TEST_COLLECTION_NAME,
#         db_source_dir=batch_folder,
#         milvus_uri="http://10.40.110.82:19530",
#         text_chunk_max_length=31744,
#         embedding_dimension=1024,
#         n_local_dense_embeddings=8,
#         is_nerp=False,
#         minio_uri="http://10.40.110.82:9000",
#         bucket_name="dev-doxa-nerp-objectstorage",
#         local_image_dir="./pipeline_storage/milvus_minio_sync/images",
#         download_dir="./pipeline_storage/downloader/downloaded_voc",
#         minio_access_key="minioadmin",
#         minio_secret_key="minioadmin",
#     )

#     rebuild_milvus_only(
#         sync=sync,
#         collection_name=TEST_COLLECTION_NAME,
#         batch_folder=batch_folder,
#         target_page_ids=TARGET_PAGE_IDS,
#     )

#     res = sync.milvus_client.query(
#         collection_name=TEST_COLLECTION_NAME,
#         filter="source != ''",
#         output_fields=["source", "labels"],
#         limit=100,
#     )

#     print("\n[QUERY RESULT]")
#     for r in res:
#         print("=" * 80)
#         print("source:", r["source"])
#         print("labels:", r["labels"])
#         print("labels type:", type(r["labels"]))

#     stats = sync.milvus_client.get_collection_stats(TEST_COLLECTION_NAME)
#     logger.info("[DONE] TEST COLLECTION STATS = %s", stats)

if __name__ == "__main__":
    main()