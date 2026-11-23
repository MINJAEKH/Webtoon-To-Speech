"""
# chunking_and_embedding.py
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : dump 문서를 청킹하고 임베딩하여 DB 입력용 entity parquet으로 변환하는 4단계
 *               created/updated page는 전체 임베딩을, 구조/labels 변경 page는 metadata patch만
 *               수행하며, batch 단위로 dense/sparse 임베딩을 생성해 저장한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

import os
import logging
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import shutil
import numpy as np
from typing import Optional

from src.components.text_splitter.text_splitter_impl import TextSplitterModule
from src.utils.postprocess import get_database_input_entity_list
from src.components.embedder.custom_embedder import CustomEmbedder
from src.components.database.update_db_source import UpdateDBSource
from src.components.embedder.utils import make_documents
from math import ceil
from itertools import chain

from src.utils.utils import batched, batched_by_size, measure_time, build_hashed_dir_name
from src.utils.label_utils import parse_confluence_labels

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Confluence dump 문서를 chunking하고 embedding을 생성하여 DB 입력용 entity parquet으로 변환하는 클래스.

    이 클래스는 downloader / document processor 이후 단계에서 실행된다.
    dump parquet에 저장된 문서를 읽고, 변경 유형에 따라 다음 작업을 수행한다.

    처리 유형:
        1. full embedding:
            created / updated page는 문서 내용이 새로 생겼거나 바뀐 것으로 보고
            chunking, dense embedding, sparse tokenizing, entity 생성 과정을 다시 수행한다.

        2. metadata-only patch:
            moved / ancestor_path_changed / ancestor_title_changed / labels_changed page는
            본문 embedding을 다시 만들지 않고 기존 entity parquet의 metadata만 갱신한다.

        3. delete:
            deleted page는 기존 entity parquet에서 해당 source를 제거한다.

    최종 산출물:
        - db_source_dir/entity_batch_{idx}.parquet
        - embed_and_update_dir/source batch index 정보 parquet
    """
    def __init__(
        self,
        embedder_url,
        embedder_url_prod,
        stopword_path,
        user_tokenizer_path,
        norm_dict_path,
        normalization_mode,
        n_local_overlap_words,
        n_local_dense_embeddings,
        source_baseurl,
        tokenizer_url,
        use_token_based_splitter,
        text_chunk_size,
        text_chunk_overlap_size,
        embed_and_update_dir,
        download_dir,
        abandoned_dir,
        db_source_dir,
        source_batch_num_info_filename,
        source_batch_len_info_filename,
        parquet_batch_size = 512
    ):
        """
        chunking / embedding / DB source update에 필요한 설정과 helper 객체를 초기화한다.

        Args:
            embedder_url: 기본 embedding endpoint URL.
            embedder_url_prod: 기본 endpoint 실패 시 fallback으로 사용할 운영 embedding endpoint URL.
            stopword_path: sparse tokenizing 또는 normalization에 사용할 stopword 파일 경로.
            user_tokenizer_path: 사용자 정의 tokenizer 파일 경로.
            norm_dict_path: 정규화 사전 파일 경로.
            normalization_mode: 텍스트 정규화 방식.
            n_local_overlap_words: local dense embedding 생성 시 chunk 간 overlap 단어 수.
            n_local_dense_embeddings: chunk당 생성할 local dense embedding 개수.
            source_baseurl: page_id를 source URL로 변환할 때 사용하는 Confluence base URL.
            tokenizer_url: token-based splitter에서 사용할 tokenizer endpoint.
            use_token_based_splitter: token 기준 chunking 사용 여부.
            text_chunk_size: chunk 최대 길이.
            text_chunk_overlap_size: chunk overlap 크기.
            embed_and_update_dir: source-batch 매핑 정보를 저장하는 디렉터리.
            download_dir: 정상 다운로드 산출물 디렉터리.
            abandoned_dir: embedding 실패 또는 삭제 산출물을 이동할 디렉터리.
            db_source_dir: entity parquet batch 파일 저장 디렉터리.
            source_batch_num_info_filename: source별 batch index 정보 파일명.
            source_batch_len_info_filename: batch별 row 수 정보 파일명.
            parquet_batch_size: 신규 entity parquet 저장 시 사용할 기본 batch 크기.
        """

        
        self.embedder_url = embedder_url
        self.embedder_url_prod = embedder_url_prod
        self.stopword_path = stopword_path
        self.user_tokenizer_path = user_tokenizer_path
        self.norm_dict_path = norm_dict_path
        self.normalization_mode = normalization_mode
        self.n_local_overlap_words = n_local_overlap_words
        self.n_local_dense_embeddings = n_local_dense_embeddings
        self.source_baseurl = source_baseurl
        self.tokenizer_url = tokenizer_url
        self.use_token_based_splitter = use_token_based_splitter
        self.text_chunk_size = text_chunk_size
        self.text_chunk_overlap_size = text_chunk_overlap_size
        self.embed_and_update_dir=embed_and_update_dir
        self.download_dir=download_dir
        self.abandoned_dir=abandoned_dir
        self.db_source_dir=db_source_dir
        self.source_batch_num_info_filename=source_batch_num_info_filename
        self.source_batch_len_info_filename=source_batch_len_info_filename
        self.parquet_batch_size = parquet_batch_size
       
        # 기본 embedding client.
        # 우선 이 endpoint로 chunk embedding을 생성한다.
        self.embedder_system = CustomEmbedder(
            stopword_path=self.stopword_path,
            user_tokenizer_path=self.user_tokenizer_path,
            norm_dict_path=self.norm_dict_path,
            normalization_mode=self.normalization_mode,
            embedder_url=self.embedder_url,
            n_local_overlap_words=int(self.n_local_overlap_words),
            n_local_dense_embeddings=self.n_local_dense_embeddings
        )
        # fallback embedding client.
        # 기본 endpoint에서 batch embedding까지 실패한 경우 prod endpoint로 한 번 더 시도한다.
        self.embedder_system_prod = CustomEmbedder(
            stopword_path=self.stopword_path,
            user_tokenizer_path=self.user_tokenizer_path,
            norm_dict_path=self.norm_dict_path,
            normalization_mode=self.normalization_mode,
            embedder_url=self.embedder_url_prod,
            n_local_overlap_words=int(self.n_local_overlap_words),
            n_local_dense_embeddings=self.n_local_dense_embeddings
        )
        # entity parquet batch 파일과 source-batch 매핑 정보를 갱신하는 helper.
        self.update_db_source = UpdateDBSource(
            embed_and_update_dir = self.embed_and_update_dir,
            # download_dir = self.download_dir,
            db_source_dir = self.db_source_dir,
            source_batch_num_info_filename = self.source_batch_num_info_filename,
            source_batch_len_info_filename = self.source_batch_len_info_filename,
        )
    
    @measure_time
    def embed_document_all(self, dump_filepath, changed_urls) -> list[list[dict]]:
        """
        changed_urls를 기준으로 delete, metadata-only patch, full embedding을 수행한다.

        처리 순서:
            1. full embedding 대상 산출
                - created / updated page
            2. metadata-only patch 대상 산출
                - moved / ancestor_path_changed / ancestor_title_changed / labels_changed
                - 단, full embedding 대상과 deleted 대상은 제외
            3. deleted source를 entity parquet에서 제거
            4. metadata-only 대상의 page_tree / labels를 기존 entity parquet에 patch
            5. created / updated source에 대해 chunking + embedding + entity update 수행

        Args:
            dump_filepath:
                document processor가 생성한 최신 confluence dump parquet 경로.

            changed_urls:
                변경 유형별 page_id 목록.
                build_changed_urls_from_compare_results() 결과 또는 저장된 changed_urls 로드 결과.

        Returns:
            list[list[dict]]:
                현재 구현에서는 entity를 직접 반환하지 않고 parquet에 반영하므로 빈 list를 반환한다.
        """
        # ==========================================================
        # 1. full embedding 대상
        # - created / updated는 본문 또는 신규 문서이므로 chunking + embedding 재생성
        # ==========================================================
        full_embed_page_ids = sorted(
            set(map(str, changed_urls.get("updated", [])))
            | set(map(str, changed_urls.get("created", [])))
        )
    
        deleted_page_ids = sorted(set(map(str, changed_urls.get("deleted", []))))
        # UpdateDBSource는 source URL 기준으로 entity를 삭제/갱신하므로 page_id를 source URL로 변환한다.    
        source_urls = sorted(
            f"{self.source_baseurl}{page_id}" for page_id in full_embed_page_ids
        )
        deleted_urls = sorted(
            f"{self.source_baseurl}{page_id}" for page_id in deleted_page_ids
        )
    
        # ==========================================================
        # 2. metadata-only patch 대상
        # - embedding 값은 유지
        # - entity parquet의 title/page_tree/labels만 최신 dump 기준으로 patch
        # ==========================================================
    
        metadata_only_page_ids = sorted(
            (
                set(map(str, changed_urls.get("moved", [])))
                | set(map(str, changed_urls.get("ancestor_path_changed", [])))
                | set(map(str, changed_urls.get("ancestor_title_changed", [])))
                | set(map(str, changed_urls.get("labels_changed", [])))
            )
            - set(full_embed_page_ids)
            - set(deleted_page_ids)
        )
    
        logger.info(
            "[Embedder targets] full_embed=%s | metadata_only=%s | deleted=%s",
            len(full_embed_page_ids),
            len(metadata_only_page_ids),
            len(deleted_page_ids),
        )
    
        batch_num_info_filepath = os.path.join(
            self.embed_and_update_dir,
            self.source_batch_num_info_filename,
        )
        batch_len_info_filepath = os.path.join(
            self.embed_and_update_dir,
            self.source_batch_len_info_filename,
        )
        # source -> batch index 매핑 정보.
        # 기존 source를 업데이트할 때 어느 entity_batch 파일을 열어야 하는지 판단하는 데 사용한다.    
        if os.path.exists(batch_num_info_filepath):
            batch_num_df = pd.read_parquet(batch_num_info_filepath)
        else:
            batch_num_df = pd.DataFrame(columns=["source", "idx"])
    
        if os.path.exists(batch_len_info_filepath):
            batch_len_df = pd.read_parquet(batch_len_info_filepath)
        else:
            batch_len_df = pd.DataFrame(columns=["idx", "data_length"])
    
        # ==========================================================
        # 3. deleted 처리
        # ==========================================================
        # 삭제된 page는 entity parquet에서 해당 source 전체를 제거한다.
        # source-batch 매핑 정보도 delete_entity 내부에서 함께 갱신된다.
        if deleted_urls:
            self.update_db_source.delete_entity(deleted_urls, batch_num_df, batch_len_df)
            batch_num_df.to_parquet(batch_num_info_filepath, engine="pyarrow", index=False)
            batch_len_df.to_parquet(batch_len_info_filepath, engine="pyarrow", index=False)
        # ==========================================================
        # 4. metadata-only patch 처리
        # - row 수가 바뀌지 않으므로 batch_num_df / batch_len_df 수정 불필요
        # ==========================================================
        # moved 또는 ancestor/title/label만 바뀐 page는 embedding을 재생성하지 않는다.
        # 기존 entity row의 metadata만 최신 dump 기준으로 patch한다.
        if metadata_only_page_ids:
            self.patch_entity_metadata_only(
                dump_filepath=dump_filepath,
                page_ids=metadata_only_page_ids,
                batch_num_df = batch_num_df
            )
    
        # ==========================================================
        # 5. updated / created full embedding 처리
        # ==========================================================
        logger.info(f"Total {len(source_urls)} source urls need to be processed in Embedder.")
    
        if not source_urls:
            logger.info("[Embedder] no full embedding targets. skip embedding generation.")
            return []
    
        total_existing_batches = batch_num_df["idx"].max()
        batch_urls_generator = batched(source_urls, batch_num_df)
        self.embed_and_update(
            dump_filepath,
            source_urls,
            total_existing_batches,
            batch_urls_generator,
            is_new_source=False,
        )
        logger.info("Embed and Update existing sources in batch_num_df")
    
        new_source_urls = []
        existing_source_set = set(batch_num_df["source"].astype(str).tolist())
    
        for source_url in source_urls:
            if source_url not in existing_source_set:
                new_source_urls.append(source_url)
    
        if new_source_urls:
            logger.info("There are new urls!")
            for new_source_url in new_source_urls:
                logger.info(f"new page id is : {new_source_url.split('=')[-1]}")
    
        embedding_batch_size = 32
        batch_urls_generator = batched_by_size(new_source_urls, embedding_batch_size)
        total_batches = ceil(len(new_source_urls) / embedding_batch_size)
    
        self.embed_and_update(
            dump_filepath,
            new_source_urls,
            total_batches,
            batch_urls_generator,
            is_new_source=True,
        )
    
        logger.info(f"Embed and Update {len(source_urls) - len(new_source_urls)} existing sources in batch_num_df")
        logger.info(f"Embed and Update {len(new_source_urls)} new sources not in batch_num_df")
    
        return []

    def embed_and_update(self, dump_filepath, source_urls, total_batches, batch_urls_generator, is_new_source = False):
        """
        source URL batch 단위로 dump 문서를 읽어 embedding entity를 생성하고 parquet에 반영한다.

        처리 방식:
            1. dump parquet를 로드한다.
            2. batch_urls에 해당하는 source row만 필터링한다.
            3. dump row를 LangChain-style document 객체로 변환한다.
            4. 각 document를 chunking + embedding하여 entity list를 만든다.
            5. 신규 source면 append 방식으로 저장하고, 기존 source면 기존 batch 내부에서 교체한다.

        Args:
            dump_filepath: 최신 confluence dump parquet 경로.
            source_urls: 전체 처리 대상 source URL 목록.
            total_batches: progress/logging에 사용할 전체 batch 수.
            batch_urls_generator: source URL batch generator.
            is_new_source:
                True이면 신규 source append 경로를 사용한다.
                False이면 기존 source update 경로를 사용한다.
        """
        confluence_dump_df = pd.read_parquet(dump_filepath)
        
        if "labels" not in confluence_dump_df.columns:
            confluence_dump_df["labels"] = [
                self._normalize_labels_for_entity(None)
                for _ in range(len(confluence_dump_df))
            ]
        # else:
        #     confluence_dump_df["labels"] = confluence_dump_df["labels"].apply(_normalize_labels_for_entity)
        #     dump_df["labels"] = dump_df["labels"].apply(
        #         self._normalize_labels_for_entity
        #     )

        
        for batch_idx, batch_urls in enumerate(batch_urls_generator, start=0):
            df = confluence_dump_df.loc[confluence_dump_df['source'].isin(batch_urls)]
            if df.empty:    # deleted_ids
                self.update_db_source.update_entity([])   # 빈 배치도 명시적으로 처리
                continue
            docs = df.apply(make_documents, axis=1)
            # for doc in docs.head(3):
            #     logger.info(
            #         "[LABEL AFTER make_documents] source=%s | type=%s | labels=%s",
            #         doc.metadata.get("source"),
            #         type(doc.metadata.get("labels")),
            #         doc.metadata.get("labels"),
            #     )
            logger.info(
                f"파싱된 문서를 임베딩하여 데이터베이스 입력 엔티티 리스트 생성 중... "
                f"batch {batch_idx}/{total_batches}"
            )
            ent_lists = (self.embed_document([doc]) for doc in tqdm(docs, leave=False))
            entities = list(chain.from_iterable(ent_lists))
            logger.info(f"batch {batch_idx}/{total_batches} 완료 - 생성 엔티티 수: {len(entities)}")
            if is_new_source:
                self.update_db_source.update_entity(entities)
            else:
                self.update_db_source.update_entity_in_batches(entities)

    def embed_document(self, parsed_document):
        """
        단일 parsed document를 chunking하고 embedding하여 DB 입력 entity list를 생성한다.

        처리 흐름:
            1. TextSplitterModule로 문서를 chunk로 분할한다.
            2. chunk text와 metadata를 embedding 입력 형식으로 구성한다.
            3. dense embedding을 생성한다.
                - 1차: 기본 endpoint에서 한 번에 처리
                - 2차: 기본 endpoint에서 batch_size=1로 처리
                - 3차: prod endpoint에서 batch_size=1로 처리
            4. sparse 검색용 tokenized_text를 생성한다.
            5. chunk text, metadata, dense embedding, sparse token을 합쳐 entity를 만든다.

        Args:
            parsed_document:
                make_documents 결과로 생성된 document list.
                현재 호출부에서는 [doc] 형태로 하나의 document를 감싸서 전달한다.

        Returns:
            list[dict]:
                Milvus/DB source 입력에 사용할 entity list.
                embedding 실패 시 빈 list를 반환한다.
        """
        try:
            # 1. 문서 청킹.
            # token-based splitter 사용 여부와 chunk size/overlap 설정은 config에서 제어한다.            
            text_splitter_module = TextSplitterModule(
                text_chunk_size=self.text_chunk_size,
                text_chunk_overlap_size=self.text_chunk_overlap_size,
                tokenizer_url=self.tokenizer_url,
                use_token_based_splitter=self.use_token_based_splitter,
            )
            
            text_splitter_result = text_splitter_module.get_result(parsed_document)

            # 2. embedding 입력 데이터 준비.
            # chunk_text_org_list는 faq, voc 등 title, header 등이 제외된 DB 저장용 version (원본 text)
            # chunk_text는 embedding 대상이고, metadata는 source/page_tree/title 등 DB entity 생성에 사용된다.
            chunk_text_org_list = []
            chunk_text_list = []
            metadata_list = []

            for chunk in text_splitter_result:
                chunk_text_org_list.append(chunk.page_content)
                metadata_temp = chunk.metadata
                chunk_text_list.append(metadata_temp["text_for_emb"])
                del metadata_temp["text_for_emb"]
                metadata_list.append(metadata_temp)
            chunk_data = {
                'chunk_text_org': chunk_text_org_list,
                'chunk_text': chunk_text_list,
                'metadata': metadata_list,
            }
            # 3. Dense 임베딩 생성 (한 번에 시도, 실패 시 배치로)

            try:
                # raise RuntimeError("")
                dense_embedding_list, chunk_caption_list, chunk_caption_match_list = self.embedder_system.process_chunks_at_once(
                        chunk_data=chunk_data
                    )
            except Exception as e:
                try:
                    logger.warning(f'[WARNING]: process_chunks_at_once failed, trying batches: {e}')
                    dense_embedding_list, chunk_caption_list, chunk_caption_match_list = self.embedder_system.process_chunks_in_batches(
                            text_splitter_result=text_splitter_result,
                            chunk_data=chunk_data,
                            batch_size=1  # 배치 크기 지정
                        )
                except Exception as e:
                    logger.warning(f'[WARNING]: process_chunks_at_bathces failed, trying batches in PROD: {e}')
                    try:
                        dense_embedding_list, chunk_caption_list, chunk_caption_match_list = self.embedder_system_prod.process_chunks_in_batches(
                                text_splitter_result=text_splitter_result,
                                chunk_data=chunk_data,
                                batch_size=1  # 배치 크기 지정
                            )
                   
                    except Exception as e:
                        # 모든 embedding 시도가 실패하면 해당 source의 다운로드 산출물을 abandoned_dir로 이동한다.
                        # 다음 pipeline 실행 시 missing page로 다시 감지되어 재다운로드/재처리될 수 있게 하기 위함이다.
                        
                        sources = list(set([metadata['source'] for metadata in chunk_data['metadata']]))
                        
                        for source in sources:
                            page_id = source.split("=")[-1] if source else "unknown"

                            folder_name = build_hashed_dir_name(source, page_id)
                            downloaded_folder_path = os.path.join(self.download_dir, folder_name)
                            abandoned_folder_path = os.path.join(self.abandoned_dir, folder_name)
                            logger.info("[EMBED FAIL MOVE CHECK] page_id=%s", page_id)
                            logger.info("[EMBED FAIL MOVE CHECK] source=%s", source)
                            logger.info("[EMBED FAIL MOVE CHECK] folder_name=%s", folder_name)
                            logger.info("[EMBED FAIL MOVE CHECK] downloaded_folder_path=%s", downloaded_folder_path)
                            logger.info("[EMBED FAIL MOVE CHECK] abandoned_folder_path=%s", abandoned_folder_path)
                            logger.info("[EMBED FAIL MOVE CHECK] downloaded_exists=%s", os.path.exists(downloaded_folder_path))
                            logger.info("[EMBED FAIL MOVE CHECK] abandoned_exists=%s", os.path.exists(abandoned_folder_path))
                        
                            if os.path.exists(downloaded_folder_path):
                                if os.path.exists(abandoned_folder_path):
                                    shutil.rmtree(abandoned_folder_path)
                                shutil.move(downloaded_folder_path, abandoned_folder_path)
                                logger.info("[EMBED FAIL MOVE DONE] moved %s -> %s", downloaded_folder_path, abandoned_folder_path)
                            else:
                                logger.warning("[EMBED FAIL MOVE SKIP] downloaded folder not found: %s", downloaded_folder_path)

                        
                        source = parsed_document[0].metadata.get("source", "")
                        title = parsed_document[0].metadata.get("title", "")
                        page_id = source.split("=")[-1] if source else "unknown"
                        logger.error(
                            "[EMBEDDING FAILED] page_id=%s | title=%s | source=%s | chunk_count=%s | max_chunk_len=%s | error=%s",
                            page_id,
                            title,
                            source,
                            len(chunk_data["chunk_text"]),
                            max((len(x) for x in chunk_data["chunk_text"]), default=0),
                            str(e),
                        )
                    
                        return []
            
            # 4. Sparse 벡터용 토큰화.
            # dense embedding과 별도로 sparse retrieval에 사용할 tokenized_text를 만든다.
            tokenized_text_list = self.embedder_system.tokenize_for_sparse(
                chunk_data['chunk_text']
            )
            
            # 5. 임베딩 결과 통합
            embedded_chunks = {
                'chunk_text': chunk_data['chunk_text_org'],
                'metadata': chunk_data['metadata'],
                'tokenized_text': tokenized_text_list, # tokenized text
                'dense_embedding': dense_embedding_list,
                'chunk_caption': chunk_caption_list,
                'chunk_caption_match': chunk_caption_match_list,
            }

            # 6. 데이터베이스 입력 엔티티 생성
            database_entity_list = get_database_input_entity_list(embedded_chunks)

            for row in database_entity_list:
                page_tree = row.get("page_tree", "")
                row["business_area"] = self._extract_business_area(page_tree)
            
            logger.info("")
            return database_entity_list

        # except Exception as e:
        #     ## TODO : parsed_documents 전체에서 특정 필요한 metadata만 print하도록
        #     logger.error(f'[ERROR]: DocumentEmbedder error | {e} | while embedding {parsed_document[0].metadata["source"]}', exc_info = True)
        #     return []
        except Exception as e:
            source = ""
        
            try:
                source = parsed_document[0].metadata.get(
                    "source",
                    "",
                )
        
                if source:
                    self._move_source_download_to_abandoned(
                        source=source,
                        reason="embed_document_exception",
                    )
        
            except Exception as move_error:
                logger.warning(
                    "[MOVE TO ABANDONED ERROR] source=%s | error=%s",
                    source,
                    move_error,
                )
        
            logger.error(
                "[ERROR]: DocumentEmbedder error | %s | while embedding %s",
                e,
                source,
                exc_info=True,
            )
        
            return []

    def _move_source_download_to_abandoned(
        self,
        source: str,
        reason: str,
    ) -> None:
        page_id = source.split("=")[-1] if source else "unknown"
    
        folder_name = build_hashed_dir_name(
            source,
            page_id,
        )
    
        downloaded_folder_path = os.path.join(
            self.download_dir,
            folder_name,
        )
    
        abandoned_folder_path = os.path.join(
            self.abandoned_dir,
            folder_name,
        )
    
        logger.info(
            "[MOVE TO ABANDONED CHECK] reason=%s | page_id=%s",
            reason,
            page_id,
        )
    
        if not os.path.exists(downloaded_folder_path):
            logger.warning(
                "[MOVE TO ABANDONED SKIP] reason=%s | page_id=%s | downloaded folder not found",
                reason,
                page_id,
            )
            return

        os.makedirs(self.abandoned_dir, exist_ok=True)
        if os.path.exists(abandoned_folder_path):
            shutil.rmtree(abandoned_folder_path)
    
        shutil.move(
            downloaded_folder_path,
            abandoned_folder_path,
        )
    
        logger.warning(
            "[MOVE TO ABANDONED DONE][WILL_RETRY_NEXT_UPDATE] reason=%s | page_id=%s",
            reason,
            page_id,
        )

    
    def patch_entity_metadata_only(
        self,
        dump_filepath: str,
        page_ids: list[str],
        batch_num_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        embedding 재생성 없이 기존 entity parquet의 metadata만 최신 dump 기준으로 patch한다.

        사용 대상:
            - moved:
                page parent가 바뀌어 page_tree 또는 ancestor metadata가 바뀐 경우.
            - ancestor_path_changed:
                상위 경로 id가 바뀐 경우.
            - ancestor_title_changed:
                상위 title이 바뀌어 page_tree가 바뀐 경우.
            - labels_changed:
                page label만 바뀐 경우.

        patch 대상:
            - page_tree
            - labels

        제외 대상:
            - created / updated:
                본문 또는 신규 문서 가능성이 있으므로 full embedding 대상이다.
            - deleted:
                entity 삭제 대상이다.
            - structure_changed:
                parent의 child 구성 변화 자체는 해당 parent 본문 embedding 변경과 별개일 수 있어
                여기서는 직접 patch 대상으로 보지 않는다.

        Args:
            dump_filepath:
                최신 metadata가 들어 있는 confluence dump parquet 경로.

            page_ids:
                metadata만 patch할 page id 목록.

            batch_num_df:
                source -> entity_batch index 매핑 DataFrame.
                있으면 대상 batch만 열어 patch하고, 없으면 db_source_dir 전체 parquet을 scan한다.
        """
    
        page_ids = sorted({str(x) for x in (page_ids or []) if x})
        if not page_ids:
            logger.info("[metadata-only patch] no page ids. skipped.")
            return
    
        if not os.path.exists(dump_filepath):
            raise FileNotFoundError(f"dump_filepath not found: {dump_filepath}")
    
        dump_df = pd.read_parquet(dump_filepath)
    
       
        if "source" not in dump_df.columns:
            raise KeyError("dump parquet must contain 'source' column")
        
        if "labels" not in dump_df.columns:
            dump_df["labels"] = [
                self._normalize_labels_for_entity(None)
                for _ in range(len(dump_df))
            ]
        # else:
        #     dump_df["labels"] = dump_df["labels"].apply(
        #         self._normalize_labels_for_entity
        #     )
        # 대상 source 생성
        target_sources = {
            f"{self.source_baseurl}{page_id}"
            for page_id in page_ids
        }
    
        # 최신 metadata만 추출
        dump_meta_df = (
            dump_df[dump_df["source"].isin(target_sources)]
            .drop_duplicates(subset=["source"], keep="last")
            .set_index("source")
        )
    
        if dump_meta_df.empty:
            logger.warning("[metadata-only patch] no matching sources in dump")
            return
    
        patched_row_count = 0
        patched_source_set = set()
    
        # ==========================================================
        # batch_num_df 있으면 빠른 경로
        # ==========================================================
        if batch_num_df is not None and not batch_num_df.empty:
            target_batch_indices = sorted(
                batch_num_df[batch_num_df["source"].isin(target_sources)]["idx"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
    
            logger.info(
                "[metadata-only patch] using batch_num_df | target_batches=%s",
                target_batch_indices,
            )
    
            for idx in target_batch_indices:
                filepath = os.path.join(self.db_source_dir, f"entity_batch_{idx}.parquet")

                if not os.path.exists(filepath):
                    logger.warning("[metadata-only patch] batch file not found: %s", filepath)
                    continue
    
                entity_df = pd.read_parquet(filepath)
    
                if entity_df.empty or "source" not in entity_df.columns:
                    continue
    
                mask = entity_df["source"].isin(target_sources)
                if not mask.any():
                    continue
    
                # labels 컬럼 없으면 생성
                if "labels" not in entity_df.columns:
                    entity_df["labels"] = [
                        self._normalize_labels_for_entity(None)
                        for _ in range(len(entity_df))
                    ]
    
                for source in entity_df.loc[mask, "source"].dropna().unique():
                    if source not in dump_meta_df.index:
                        continue
    
                    source_mask = entity_df["source"] == source
                    meta_row = dump_meta_df.loc[source]
    
                    # page_tree patch
                    # if "page_tree" in entity_df.columns:
                    #     entity_df.loc[source_mask, "page_tree"] = str(
                    #         meta_row.get("page_tree", "") or ""
                    #     )

                    new_page_tree = str(
                        meta_row.get("page_tree", "") or ""
                    )
                    
                    if "page_tree" in entity_df.columns:
                        entity_df.loc[source_mask, "page_tree"] = new_page_tree

                    # 법인 간 moved 이면 business_area도 바뀌므로 재계산
                    if "business_area" not in entity_df.columns:
                        entity_df["business_area"] = ""
                    entity_df.loc[source_mask, "business_area"] = (
                        self._extract_business_area(new_page_tree)
                    )
                    
                    # labels patch
                    # labels = self._normalize_labels_for_entity(meta_row.get("labels", []))
                    labels = meta_row.get("labels")

                    entity_df.loc[source_mask, "labels"] = pd.Series(
                        [labels for _ in range(int(source_mask.sum()))],
                        index=entity_df.index[source_mask],
                    )
    
                    patched_count = int(source_mask.sum())
                    patched_row_count += patched_count
                    patched_source_set.add(source)
    
                entity_df.to_parquet(filepath, engine="pyarrow", index=False)
    
        # ==========================================================
        # fallback (batch_num_df 없을 때 전체 scan)
        # ==========================================================
        # source-batch 매핑 정보가 없으면 db_source_dir의 모든 entity parquet을 순회한다.
        # 느리지만 mapping 파일이 손상/누락된 상황에서도 metadata patch를 수행할 수 있다.
        else:
            logger.warning("[metadata-only patch] batch_num_df not found. fallback full scan")
    
            parquet_files = sorted(
                [
                    filename
                    for filename in os.listdir(self.db_source_dir)
                    if filename.endswith(".parquet")
                ],
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
    
            for filename in parquet_files:
                filepath = os.path.join(self.db_source_dir, filename)
                entity_df = pd.read_parquet(filepath)
    
                if entity_df.empty or "source" not in entity_df.columns:
                    continue
    
                mask = entity_df["source"].isin(target_sources)
                if not mask.any():
                    continue

                if "labels" not in entity_df.columns:
                    entity_df["labels"] = [
                        self._normalize_labels_for_entity(None)
                        for _ in range(len(entity_df))
                    ]

                
                for source in entity_df.loc[mask, "source"].dropna().unique():
                    if source not in dump_meta_df.index:
                        continue
    
                    source_mask = entity_df["source"] == source
                    meta_row = dump_meta_df.loc[source]
    
                    # if "page_tree" in entity_df.columns:
                    #     entity_df.loc[source_mask, "page_tree"] = str(
                    #         meta_row.get("page_tree", "") or ""
                    #     )

                    new_page_tree = str(
                        meta_row.get("page_tree", "") or ""
                    )
                    
                    if "page_tree" in entity_df.columns:
                        entity_df.loc[source_mask, "page_tree"] = new_page_tree
                    
                    # 법인 간 moved 이면 business_area도 바뀌므로 재계산
                    if "business_area" not in entity_df.columns:
                        entity_df["business_area"] = ""
                    entity_df.loc[source_mask, "business_area"] = (
                        self._extract_business_area(new_page_tree)
                    )
                    
                    # labels = self._normalize_labels_for_entity(meta_row.get("labels", []))
                    labels = meta_row.get("labels")

                    entity_df.loc[source_mask, "labels"] = pd.Series(
                        [labels for _ in range(int(source_mask.sum()))],
                        index=entity_df.index[source_mask],
                    )
    
                    patched_count = int(source_mask.sum())
                    patched_row_count += patched_count
                    patched_source_set.add(source)
    
                entity_df.to_parquet(filepath, engine="pyarrow", index=False)
    
        logger.info(
            "[metadata-only patch] completed | patched_sources=%s | patched_rows=%s",
            len(patched_source_set),
            patched_row_count,
        )

    def _normalize_labels_for_entity(self, value) -> dict:
        """
        labels를 entity parquet 저장용 dict로 정규화한다.
        입력이 이미 dict이면 그대로 key별 list로 변환하고,
        입력이 raw label list이면 parse_confluence_labels를 적용한다.
        """
        if isinstance(value, dict):
            labels = value
        else:
            labels = parse_confluence_labels(value)
    
        return {
            "region": list(labels.get("region", [])),
            "biz": list(labels.get("biz", [])),
            "corp": list(labels.get("corp", [])),
            "extra": list(labels.get("extra", [])),
        }

    def _extract_module(self, page_tree: str | None) -> str:
        """
        page_tree 첫 괄호 토큰 추출

        예시:
        NERP Knowledge Space Home / Master Data (MD) / ...
        -> MD
        """
        if not isinstance(page_tree, str):
            return ""

        m = re.search(r"\(([^)]+)\)", page_tree)
        return m.group(1).strip() if m else ""

    def _extract_business_area(self, page_tree: str | None) -> str:
        """
        page_tree 에서 법인명을 뽑는다. GPPM 전용.

        예시:
        ... / GPPM(Subsidiary) / SEA / SEA B/M / ... -> SEA
        ... / GPPM(GSTD) / B/M / Finance / ... -> '' (공통)
        ... / 경영관리기준 / ... -> '' (공통)
        """
        BUSINESS_AREA_ROOT_TITLE = "GPPM(Subsidiary)"

        if not isinstance(page_tree, str):
            return ""
        parts = [p.strip() for p in page_tree.split(" / ")]
        try:
            # GPPM(Subsidiary) 트리 하위에 법인 별 폴더 존재
            i = parts.index(BUSINESS_AREA_ROOT_TITLE)
        except ValueError:
            return ""
        return parts[i + 1] if i + 1 < len(parts) else ""

def main():
    import os
    import glob
    import shutil
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    # =====================================================
    # 1. 경로 직접 설정
    # =====================================================
    captioned_dump_dir = "./pipeline_storage/document_processor_gppm"
    dump_pattern = os.path.join(
        captioned_dump_dir,
        "v_new_confluence_dump_*.parquet",
    )

    download_dir = "./pipeline_storage/downloader/downloaded_gppm"
    abandoned_dir = "./pipeline_storage/downloader/abandoned_gppm"

    # embed_and_update_dir = "./pipeline_storage/embed_and_update_new_embedding_voc_chlee"
    # db_source_dir = "./pipeline_storage/embed_and_update_new_embedding_voc_chlee/batch_folder"

    embed_and_update_dir = "./pipeline_storage/embed_and_update_gppm"
    db_source_dir = "./pipeline_storage/embed_and_update_gppm/batch_folder"

    
    source_batch_num_info_filename = "db_source_batch_num_info.parquet"
    source_batch_len_info_filename = "db_source_batch_len_info.parquet"

    # 최초 base embedding 생성이면 True
    reset_existing_embedding_outputs = False

    # =====================================================
    # 2. embedding / chunking 설정 직접 할당
    # =====================================================
    source_baseurl = "https://confluence.sec.samsung.net/pages/viewpage.action?pageId="

    embedder_url = "http://10.196.37.2:8001/embed/text/v1"
    embedder_url_prod = "http://10.196.35.2:8001/embed/text/v1"

    stopword_path = "./asset/stopword.txt"
    user_tokenizer_path = "./asset/user_dict_v2.txt"
    norm_dict_path = "./asset/woojin_norm_dict_v2.json"
    normalization_mode = "NONE"

    n_local_overlap_words = 1
    n_local_dense_embeddings = 8

    tokenizer_url = "http://10.196.37.2:8001/embed/text/tokenize"
    use_token_based_splitter = True
    text_chunk_size = 192
    text_chunk_overlap_size = 50

    # =====================================================
    # 3. captioning 완료 dump parquet 자동 선택
    # =====================================================
    dump_files = glob.glob(dump_pattern)

    if not dump_files:
        raise FileNotFoundError(f"No captioned dump parquet found: {dump_pattern}")

    dump_filepath = max(dump_files, key=os.path.getmtime)

    logger.info("[Base embedding] selected dump parquet: %s", dump_filepath)

    # =====================================================
    # 4. 최초 base embedding이면 기존 산출물 초기화
    # =====================================================
    if reset_existing_embedding_outputs:
        for d in [embed_and_update_dir, db_source_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
                logger.info("[Base embedding] removed existing directory: %s", d)

            os.makedirs(d, exist_ok=True)
            logger.info("[Base embedding] created directory: %s", d)

    # =====================================================
    # 5. dump parquet 기준 page_id 추출
    # =====================================================
    dump_df = pd.read_parquet(dump_filepath)

    if "source" not in dump_df.columns:
        raise KeyError("dump parquet must contain 'source' column")

    # all_page_ids = (
    #     dump_df["source"]
    #     .dropna()
    #     .astype(str)
    #     .apply(lambda x : x.split("=")[-1])
    #     .unique()
    #     .tolist()
    # )
    all_page_ids = ['2024261856']
    # print(len(all_page_ids))
    # all_page_ids = all_page_ids[:3]
    # print(all_page_ids)
    # return
    
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

    logger.info(
        "[Base embedding] targets from captioned dump parquet: %s pages",
        len(all_page_ids),
    )

    # =====================================================
    # 6. Embedder 실행
    # =====================================================
    embedder = DocumentEmbedder(
        embedder_url=embedder_url,
        embedder_url_prod=embedder_url_prod,
        stopword_path=stopword_path,
        user_tokenizer_path=user_tokenizer_path,
        norm_dict_path=norm_dict_path,
        normalization_mode=normalization_mode,
        n_local_overlap_words=n_local_overlap_words,
        n_local_dense_embeddings=n_local_dense_embeddings,
        source_baseurl=source_baseurl,
        tokenizer_url=tokenizer_url,
        use_token_based_splitter=use_token_based_splitter,
        text_chunk_size=text_chunk_size,
        text_chunk_overlap_size=text_chunk_overlap_size,
        embed_and_update_dir=embed_and_update_dir,
        download_dir=download_dir,
        abandoned_dir=abandoned_dir,
        db_source_dir=db_source_dir,
        source_batch_num_info_filename=source_batch_num_info_filename,
        source_batch_len_info_filename=source_batch_len_info_filename,
    )

    embedder.embed_document_all(
        dump_filepath=dump_filepath,
        changed_urls=changed_urls,
    )

    logger.info("[Base embedding] embedding pipeline completed.")

if __name__ == "__main__":
    main()