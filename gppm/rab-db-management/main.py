"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : Confluence 문서 변경분을 기준으로 RAG DB를 갱신하는 파이프라인 전체 진입점
 *               tree discovery -> changed_urls 산출 -> 변경 page 다운로드 -> dump/captioning ->
 *               chunking/embedding -> Milvus/MinIO 동기화 순으로 단계를 실행하며,
 *               설정 로드, 로깅 초기화, 산출물 파일 관리, 단계별 재시도를 담당한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *    2       2026.11.02   soyeone.park              NERP&SEGAM 제거
 *********** ************ ************ ***************************************************** ************* *************
"""

import sys
sys.path.append("/app/site-packages2")

import logging
import time
from easydict import EasyDict
from datetime import datetime

from src.pipeline.make_confluence_tree import ConfluenceTreeDiscovery
from src.pipeline.change_urls import (
    load_latest_changed_urls_versioned,
    build_changed_urls_from_compare_results,
    log_changed_urls,
    save_changed_urls_versioned,
)
from src.pipeline.subsidiary_registry import SubsidiaryRegistry
from src.pipeline.confluence_downloader import ConfluenceDocumentExtractor
from src.pipeline.update_confluence_dump_and_captioning import ConfluenceDocumentProcessor
from src.pipeline.chunking_and_embedding import DocumentEmbedder
from src.pipeline.milvus_minio_sync import MilvusMinioSync

from src.utils.config_utils import update_bootstrap_roots
from src.test import test_connection_all
from src.utils.log_config import setup_logging, log_file_managing
from src.utils.config import print_update_steps, init_config, parse_args, validate_directories, get_config_path
from src.utils.confluence_patch import patch_confluence_request

from src.utils.utils import (
    all_file_managing,
    batch_folder_managing,
    RetryOnException,
)

def pipeline(config: EasyDict, subsidiary_list: list):
    """
    Confluence 문서 변경사항을 기준으로 RAG DB를 업데이트하는 메인 파이프라인.

    전체 흐름:
        1. Confluence tree discovery
        2. changed_urls 생성 또는 로드
        3. 변경/누락 문서 다운로드
        4. dump 생성 및 이미지 captioning
        5. 문서 chunking 및 embedding
        6. Milvus / MinIO 동기화

    Args:
        config: 실행 환경, 경로, API endpoint, DB 설정 등을 포함한 설정 객체.
    """
    logger = logging.getLogger(__name__)

    # 네트워크/API 요청 실패에 대비한 전역 retry 정책 설정
    RetryOnException.configure(
        max_retries=config.max_retries,
        initial_delay=config.initial_delay,
        backoff_factor=config.backoff_factor,
    )

    # Confluence 요청 동작을 파이프라인 환경에 맞게 patch
    patch_confluence_request()

    # =========================================================
    # 0. init
    # =========================================================
    # 단계별 실행 여부가 config로 제어되므로,
    # 이후 단계에서 참조할 수 있는 중간 결과를 기본값으로 초기화한다.
    changed_root_ids = []
    compare_results = {}
    new_tree = None
    tree_path = None
    discovery_result = {}
    downloader_result = {}
    processor_result = {}
    
    dump_filepath = getattr(config, "repair_dump_filepath", None)
    
    if dump_filepath:
        logger.info(
            "[REPAIR MODE] using repair_dump_filepath=%s",
            dump_filepath,
        )

    # 변경 유형별 page_id 목록.
    # 이후 downloader, processor, embedder, DB sync 단계에서 공통 입력으로 사용된다.
    changed_urls = {
        "created": [],
        "updated": [],
        "deleted": [],
        "moved": [],
        "ancestor_path_changed": [],
        "ancestor_title_changed": [],
        "structure_changed": [],
        "labels_changed": [],
    }

    # =========================================================
    # 0. branch discovery
    # =========================================================

    if config.run_subsidiary_registry: # DEV에서만
        if subsidiary_list : 
            logger.info("[Local] GPPM pipeline will be active, length =%s, type=%s", len(subsidiary_list), type(subsidiary_list))
            
            registry = SubsidiaryRegistry(
                page_root=config.page_root,
                confluence_token=config.confluence_token,
                subsidiary_root_id=config.subsidiary_root_id,
                subsidiary_path=config.subsidiary_path,
            )
            
            local_bootstrap_roots = registry.sync(subsidiary_list)
            update_bootstrap_roots(get_config_path(), local_bootstrap_roots)
            config.bootstrap_roots = {**config.bootstrap_roots, **local_bootstrap_roots}
            logger.info("[Local] bootstrap_roots updated | added=%s", local_bootstrap_roots)
        else :
            logger.info("[Global] GPPM pipeline will be active")
    # =========================================================
    # 1. branch discovery
    # =========================================================
    if config.run_tree_discovery:
        logger.info("Running branch discovery...")

        # Confluence 전체/부분 tree를 수집하고,
        # 이전 tree snapshot과 비교해 변경된 root 및 상세 변경사항을 계산한다.
        discovery = ConfluenceTreeDiscovery(
            page_root=config.page_root,
            confluence_token=config.confluence_token,
            tree_path=config.tree_path,
            bootstrap_roots=config.bootstrap_roots,
        )

        discovery_result = discovery.prepare()

        new_tree = discovery_result["new_tree"]
        tree_path = discovery_result["tree_path"]
        changed_root_ids = discovery_result.get("changed_root_ids", [])
        compare_results = discovery_result.get("compare_results", {})

        logger.info(
            "[DISCOVERY DONE] changed_root_ids=%s | compare_results=%s | new_tree_nodes=%s",
            len(changed_root_ids),
            len(compare_results),
            len(new_tree) if new_tree is not None else 0,
        )
    else:
        logger.warning(
            "Branch discovery skipped. In tree-based flow, "
            "run_branch_discovery should normally be True."
        )

    # =========================================================
    # 2. changed_urls
    # =========================================================
    if config.load_changed_urls:
        logger.info("Loading changed_urls from saved json...")

        # 기존에 저장된 changed_urls json을 재사용한다.
        # Prod 단계에서 DEV에서 페이지 변경사항을 추적하고 아래 과정을 수행하기 위해 사용된다.
        changed_urls = load_latest_changed_urls_versioned(
            output_dir=config.changed_urls_dir,
            filename_prefix=config.changed_urls_file_prefix,
        )
    else:
        # changed_urls를 새로 만들려면 compare_results가 필요하므로
        # tree discovery 단계가 선행되어야 한다.
        if not config.run_tree_discovery:
            raise ValueError(
                "load_changed_urls=False 인 경우 run_tree_discovery=True 여야 합니다."
            )

        changed_urls = build_changed_urls_from_compare_results(
            compare_results=compare_results,
            delete_limit=int(config.delete_limit),
            excluded_ids=getattr(config, "excluded_ids", []),
        )
    # =========================================================
    # 3. confluence downloader
    # =========================================================
    # created / updated / missing page는 다운로드 대상.
    # deleted page는 기존 다운로드 산출물 제거 대상.
    # moved / ancestor_* / label_changed는 downloader에서 직접 다운로드하지 않는다.
    excluded_ids = set(map(str, getattr(config, "excluded_ids", []) or []))
    delete_page_ids = sorted(set(changed_urls["deleted"]))

    if config.run_downloader:
        logger.info("Running confluence downloader...")

        if new_tree is None:
            raise ValueError(
                "run_downloader=True 인데 new_tree가 없습니다. "
                "tree 기반 흐름에서는 run_branch_discovery=True 이어야 합니다."
            )

        downloader = ConfluenceDocumentExtractor(
            parser_url=config.parser_url,
            parser_api_token=config.parser_api_token,
            confluence_token=config.confluence_token,
            collection_name=config.collection_name,
            download_dir=config.download_dir,
            abandoned_dir=config.abandoned_dir,
            tree_store=new_tree,
        )

        # tree에는 존재하지만 download_dir에 없는 page를 찾아 복구 대상으로 포함한다.
        missing_in_download_dir_ids = [
            page_id
            for page_id in downloader.get_all_missing_page_ids_from_tree()
            if page_id not in excluded_ids
        ]

        # 신규/수정/누락 page를 다운로드 후보로 구성한다.
        candidate_download_page_ids = sorted(
            (
                set(changed_urls["created"])
                | set(changed_urls["updated"])
                | set(missing_in_download_dir_ids)
            )
            - excluded_ids
        )

        excluded_no_body_ids = []
        download_page_ids = []

        # 본문이 없는 page는 parser 처리 대상에서 제외한다.
        for page_id in candidate_download_page_ids:
            node = new_tree.get(str(page_id))
            if node is None:
                continue

            if node.has_body is False:
                excluded_no_body_ids.append(str(page_id))
                continue

            download_page_ids.append(str(page_id))

        download_page_ids = sorted(set(download_page_ids))
        excluded_no_body_ids = sorted(set(excluded_no_body_ids))

        # created/updated가 아니었지만 누락 복구 때문에 다운로드되는 page 목록
        recovered_missing_ids = sorted(
            set(download_page_ids)
            - set(changed_urls["created"])
            - set(changed_urls["updated"])
        )

        # 누락 복구 page도 이후 processor/embedder/DB update 대상이어야 하므로 updated에 합친다.
        changed_urls["updated"] = sorted(
            set(changed_urls["updated"]) | set(recovered_missing_ids)
        )

        logger.info("config_excluded_ids: %s", sorted(excluded_ids)[:50])
        logger.info("missing_in_download_dir_ids: %s", missing_in_download_dir_ids[:50])
        logger.info("excluded_no_body_ids: %s", excluded_no_body_ids[:50])
        logger.info("download_page_ids: %s", download_page_ids[:50])
        logger.info("recovered_missing_ids: %s", recovered_missing_ids[:50])
        logger.info("updated_ids_with_recovered_missing: %s", changed_urls["updated"][:50])
        logger.info("delete_page_ids: %s", delete_page_ids[:50])

        # 실제 다운로드에 사용된 changed_urls를 versioned json으로 저장한다.
        changed_urls_save_path = save_changed_urls_versioned(
            changed_urls=changed_urls,
            output_dir=config.changed_urls_dir,
            filename_prefix=config.changed_urls_file_prefix,
            meta={
                "env": config.type,
                "tree_path": tree_path,
                "compare_count": len(compare_results),
                "includes_missing_as_updated": True,
                "missing_count": len(recovered_missing_ids),
            },
        )
        logger.info("[CHANGED_URLS SAVED] %s", changed_urls_save_path)

        downloader_result = downloader.download_confluence_incremental(
            download_page_ids=download_page_ids,
            delete_page_ids=delete_page_ids,
        )

        logger.info("[DOWNLOADER DONE] result=%s", downloader_result)
    else:
        logger.info("Confluence downloader skipped.")

    # 변경 유형별 page_id 수와 목록을 로그로 남긴다.
    log_changed_urls(changed_urls)

    # =========================================================
    # 4. dump / markdown / caption update
    # =========================================================
    if config.run_document_processor:
        logger.info("Running update_confluence_dump_and_captioning...")

        if new_tree is None:
            raise ValueError(
                "run_document_processor=True 인데 new_tree가 없습니다. "
                "tree 기반 흐름에서는 run_branch_discovery=True 이어야 합니다."
            )

        # 다운로드된 Confluence 문서를 dump/markdown 형태로 정리하고,
        # 이미지 captioning 결과를 문서 데이터에 반영한다.
        document_processor = ConfluenceDocumentProcessor(
            download_dir=config.download_dir,
            abandoned_dir=config.abandoned_dir,
            document_processor_dir=config.document_processor_dir,
            source_baseurl=config.source_baseurl,
            captioning_url=config.captioning_url,
            gauss_text_model_id=config.gauss_text_model_id,
            gauss_i2t_model_id=config.gauss_i2t_model_id,
            gauss_api_token=config.gcc_token,
            gauss_client_id=config.gcc_client,
            gauss_user_email=config.gcc_user_email,
            tokenizer_url=config.tokenizer_url,
            filename_prefix=config.confluence_dump_file_prefix,
            replace_existing_captions=config.replace_existing_captions,
            tree_store=new_tree,
            excluded_ids=getattr(config, "excluded_ids", []),
            tree_path=tree_path,
        )

        dump_filepath = document_processor.process_documents(changed_urls)
        processor_result = {"dump_filepath": dump_filepath}

        logger.info("[DOCUMENT PROCESSOR DONE] result=%s", processor_result)
    else:
        logger.info("Document processor skipped.")

    # =========================================================
    # 5. 문서 청킹 / 임베딩 / 부분 업데이트
    # =========================================================
    if config.run_embed_and_update:
        logger.info("Running embed and update...")

        if not dump_filepath:
            raise ValueError(
                "run_embed_and_update=True 인데 dump_filepath가 없습니다. "
                "run_document_processor=True 이어야 합니다."
            )

        # dump 문서를 chunk 단위로 분리하고,
        # 각 chunk에 대해 dense embedding을 생성한 뒤 DB 입력 source 파일로 저장한다.
        embedder = DocumentEmbedder(
            embedder_url=config.embedder_url,
            embedder_url_prod=config.embedder_url_prod,
            stopword_path=config.stopword_path,
            user_tokenizer_path=config.user_tokenizer_path,
            norm_dict_path=config.norm_dict_path,
            normalization_mode=config.normalization_mode,
            n_local_overlap_words=config.n_local_overlap_words,
            n_local_dense_embeddings=config.n_local_dense_embeddings,
            source_baseurl=config.source_baseurl,
            tokenizer_url=config.tokenizer_url,
            use_token_based_splitter=config.use_token_based_splitter,
            text_chunk_size=config.text_chunk_size,
            text_chunk_overlap_size=config.text_chunk_overlap_size,
            embed_and_update_dir=config.embed_and_update_dir,
            download_dir=config.download_dir,
            abandoned_dir=config.abandoned_dir,
            db_source_dir=config.db_source_dir,
            source_batch_num_info_filename=config.source_batch_num_info_filename,
            source_batch_len_info_filename=config.source_batch_len_info_filename,
        )

        embedder.embed_document_all(dump_filepath, changed_urls)

        # embedding 산출물 batch folder를 보관 정책에 맞게 정리한다.
        batch_folder_managing(config)

    # =========================================================
    # 6. Milvus / MinIO 동기화
    # =========================================================
    # DANGER ZONE:
    # collection_name 설정에 따라 신규 collection 생성 또는 기존 collection 변경이 발생할 수 있다.
    # 운영/개발 환경과 collection_name을 반드시 확인한 뒤 실행해야 한다.
    if config.run_sync_db:
        logger.info("Running milvus insert...")

        # Milvus에는 chunk metadata와 embedding vector를 저장하고,
        # MinIO에는 문서 이미지 파일을 저장한다.
        milvus_minio_client = MilvusMinioSync(
            milvus_batch_size=config.milvus_batch_size,
            collection_name=config.collection_name,
            db_source_dir=config.db_source_dir,
            milvus_uri=config.milvus_uri,
            text_chunk_max_length=config.text_chunk_max_length,
            embedding_dimension=config.embedding_dimension,
            n_local_dense_embeddings=config.n_local_dense_embeddings,
            minio_uri=config.minio_uri,
            bucket_name=config.bucket_name,
            local_image_dir=config.local_image_dir,
            download_dir=config.download_dir,
            minio_access_key=config.minio_access_key,
            minio_secret_key=config.minio_secret_key,
        )

        milvus_minio_client.run_sync(
            changed_urls=changed_urls,
            run_sync_db_all=config.run_sync_db_all,
            source_baseurl=config.source_baseurl,
        )

def run(config, subsidiary_list):
    start_time = time.time()
    logger = logging.getLogger(__name__)
    stage = config.type

    try:
        logger.info(f"[Airflow {stage}][START] process started")

        validate_directories(config)
        print_update_steps(config)

        # if corp_list :
        #     logger.info("[Local] GPPM pipeline will be active, length =%s, type=%s", len(corp_list), type(corp_list))
        # else :
        #     logger.info("[Global] GPPM pipeline will be active")
        if stage in ("TEST", "TEST_GPPM"):
            if not test_connection_all(config):
                raise RuntimeError("TEST connection check failed")
        else:
            pipeline(config, subsidiary_list)
            logger.info("The New Confluence pages are Successfully Saved!")

        all_file_managing(config)
        log_file_managing(stage)

        elapsed_min, elapsed_sec = divmod(time.time() - start_time, 60)
        logger.info(
            f"[{stage}] {config.collection_name} executed in "
            f"{int(elapsed_min)} min {elapsed_sec:.2f} sec."
        )
        logger.info("=" * 80)

        logger.info(f"[Airflow {stage}][SUCCESS] process completed")

    except Exception as e:
        logger.exception(f"[Airflow {stage}][FAIL] process failed: {e}")
        raise

if __name__ == "__main__":
    args = parse_args()
    config_file_path = args.config
    subsidiary_list = args.subsidiary_list
    
    config = init_config(config_file_path)

    setup_logging(
        level=logging.INFO,
        log_prefix=config.type,
        log_timestamp=args.airflow_start_date,
    )

    run(config, subsidiary_list)