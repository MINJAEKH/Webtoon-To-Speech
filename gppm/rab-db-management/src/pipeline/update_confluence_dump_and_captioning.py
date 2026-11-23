"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : 다운로드 산출물을 dump parquet으로 통합하고 이미지 caption을 생성하는 3단계
 *               변경 유형에 따라 markdown 재파싱, dump에서 삭제, metadata/labels patch를 수행하고,
 *               신규/변경 page의 이미지에 대해 Gauss API로 caption을 생성해 dump에 반영한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

import os
import re
import html
import glob
import logging
from pathlib import Path
import shutil

from datetime import datetime
from typing import List, Dict, Optional, Any, Set, Tuple
from tqdm import tqdm
import numpy as np

import pandas as pd
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.utils import sanitize_file_name, manage_version_files, extract_timestamp, measure_time
from src.utils.label_utils import parse_confluence_labels

from src.components.captioning.context_extractor import EnhancedContextExtractor
from src.components.captioning.gauss_api_client import GaussCaptionClient

logger = logging.getLogger(__name__)

class ConfluenceDocumentProcessor:
    """
    다운로드된 Confluence 산출물을 dump parquet으로 통합하고 이미지 caption을 생성하는 클래스.

    처리 흐름:
        1. created / updated page는 markdown을 다시 파싱한다.
        2. deleted page는 기존 dump에서 제거한다.
        3. moved / ancestor / structure 변경은 metadata만 patch한다.
        4. labels_changed는 labels만 patch한다.
        5. created / updated page의 image caption을 병렬 생성한다.
        6. 이미지 caption이 하나라도 실패한 page는 abandoned 처리한다.
        7. abandoned page는 parquet에서 제외하고 downloaded 폴더를 abandoned_dir로 이동한다.
    """

    def __init__(
        self,
        download_dir: str,
        abandoned_dir: str,
        document_processor_dir: str,
        source_baseurl: str,
        captioning_url: str,
        gauss_text_model_id: str,
        gauss_i2t_model_id: str,
        gauss_api_token: str,
        gauss_client_id: str,
        gauss_user_email: str,
        tokenizer_url: str,
        filename_prefix: str = "confluence_dump",
        replace_existing_captions: bool = False,
        tree_store=None,
        excluded_ids: Optional[List[str]] = None,
        tree_path: Optional[str] = None,
        max_workers: int = 4,
    ):
        """
        document processor 실행에 필요한 경로와 tree metadata를 초기화한다.

        Args:
            download_dir:
                ConfluenceDocumentExtractor가 저장한 다운로드 산출물 디렉터리.

            document_processor_dir:
                confluence_dump parquet을 저장할 디렉터리.

            source_baseurl:
                page_id를 source URL로 변환할 때 사용하는 base URL.

            captioning_url:
                이미지 caption 생성 API endpoint.

            filename_prefix:
                저장되는 dump parquet 파일 prefix.

            replace_existing_captions:
                False이면 기존 dump에 존재하는 caption을 재사용한다.
                True이면 기존 caption이 있어도 새로 생성한다.

            tree_store:
                최신 Confluence tree.
                label, title, page_tree 등 최신 metadata patch에 사용한다.

            excluded_ids:
                처리에서 제외할 page_id 목록.

            tree_path:
                최신 tree 파일 경로.
                dump 파일 timestamp를 tree snapshot과 맞추기 위해 사용한다.
        """

        
        self.download_dir = download_dir
        self.abandoned_dir = abandoned_dir
        self.document_processor_dir = document_processor_dir
        self.source_baseurl = source_baseurl
        self.captioning_url = captioning_url
        self.filename_prefix = filename_prefix
        self.replace_existing_captions = replace_existing_captions
        self.tree_store = tree_store
        self.excluded_ids = set(map(str, excluded_ids or []))
        self.tree_path = tree_path
        self.null_pages = []
        
        self.max_workers = max_workers
        self.captioning_status_list: List[Dict[str, Any]] = []
        
        self.caption_client = GaussCaptionClient(
            captioning_url=captioning_url,
            gauss_text_model_id=gauss_text_model_id,
            gauss_i2t_model_id=gauss_i2t_model_id,
            gauss_api_token=gauss_api_token,
            gauss_client_id=gauss_client_id,
            gauss_user_email=gauss_user_email,
            tokenizer_url=tokenizer_url,
        )
        self.context_extractor = EnhancedContextExtractor()

    
    @measure_time
    def process_documents(self, changed_urls: Dict[str, List[str]]) -> str:
        """
        changed_urls를 기준으로 dump parquet을 full build 또는 incremental update한다.

        처리 흐름:
            1. changed_urls를 full_process / structure_patch / label_patch / deleted로 분류한다.
            2. 기존 dump가 없으면 download_dir 전체를 기준으로 full build한다.
            3. 기존 dump가 있으면 created / updated page만 markdown에서 다시 추출한다.
            4. 기존 dump와 새 문서를 merge하면서 deleted source는 제거한다.
            5. moved / ancestor / structure 변경은 title/page_tree만 patch한다.
            6. labels_changed는 labels만 patch한다.
            7. full_process 대상에 대해서만 caption을 새로 생성한다.
            8. versioned parquet으로 저장한다.

        Args:
            changed_urls:
                변경 유형별 page_id 목록.

        Returns:
            str:
                저장된 최신 confluence_dump parquet 경로.
        """
        logger.info("Starting document processing pipeline...")
    
        # 본문을 다시 읽고 dump record를 새로 만들어야 하는 대상.
        # created/updated는 page_content 또는 image_info가 바뀌었을 가능성이 있으므로 full process가 필요하다.

        full_process_ids = sorted(
            (
                set(changed_urls.get("created", []))
                | set(changed_urls.get("updated", []))
            )
            - self.excluded_ids
        )

        structure_patch_ids = sorted(
            (
                set(changed_urls.get("moved", []))
                | set(changed_urls.get("ancestor_path_changed", []))
                | set(changed_urls.get("ancestor_title_changed", []))
                | set(changed_urls.get("structure_changed", []))
            )
            - set(full_process_ids)
            - set(changed_urls.get("deleted", []))
            - self.excluded_ids
        )

        label_patch_ids = sorted(
            set(changed_urls.get("labels_changed", []))
            - set(full_process_ids)
            - set(changed_urls.get("deleted", []))
            - self.excluded_ids
        )

        deleted_ids = sorted(set(changed_urls.get("deleted", [])) - self.excluded_ids)

        logger.info(
            "Processing targets | full_process=%s, structure_patch=%s, label_patch=%s, deleted=%s",
            len(full_process_ids),
            len(structure_patch_ids),
            len(label_patch_ids),
            len(deleted_ids),
        )

        latest_dump = self._get_latest_dump_path()

        # =====================================================
        # bootstrap: 기존 dump가 없으면 download_dir 전체 기준 full build
        # =====================================================
        if latest_dump is None:
            logger.info("No existing dump found. Run full build from download_dir.")

            all_page_ids = self._collect_all_page_ids_from_download_dir()
            logger.info("Full build targets from download_dir: %s", len(all_page_ids))

            documents = self.extract_documents_by_page_ids(all_page_ids)
            logger.info("Extracted %s document records for full build", len(documents))

            merged_df = pd.DataFrame(documents)

            merged_df, abandoned_page_ids = self.generate_captions(
                merged_df,
                all_page_ids,
            )

            merged_df = self._exclude_and_move_abandoned_pages(
                merged_df=merged_df,
                abandoned_page_ids=abandoned_page_ids,
            )

            filepath = self._save_with_version_control(merged_df)
            logger.info("Document processing pipeline completed (full build)")
            return filepath

        # =====================================================
        # incremental: created / updated만 다시 추출
        # =====================================================
        documents = self.extract_documents_by_page_ids(full_process_ids)
        logger.info("Extracted %s document records for full processing", len(documents))

        merged_df = self._merge_with_existing_data(
            documents=documents,
            changed_urls=changed_urls,
            latest_dump=latest_dump,
        )
        logger.info("Merged data: %s total records", len(merged_df))

        latest_tree_pages = None

        if structure_patch_ids:
            latest_tree_pages = self._load_latest_tree_pages()
            merged_df = self._apply_structure_patches(
                merged_df=merged_df,
                page_ids=structure_patch_ids,
                latest_tree_pages=latest_tree_pages,
            )
            logger.info("Structure patch completed for %s pages", len(structure_patch_ids))

        if label_patch_ids:
            if latest_tree_pages is None:
                latest_tree_pages = self._load_latest_tree_pages()

            merged_df = self._apply_label_patches(
                merged_df=merged_df,
                page_ids=label_patch_ids,
                latest_tree_pages=latest_tree_pages,
            )
            logger.info("Label patch completed for %s pages", len(label_patch_ids))

        merged_df, abandoned_page_ids = self.generate_captions(
            merged_df,
            full_process_ids,
        )

        merged_df = self._exclude_and_move_abandoned_pages(
            merged_df=merged_df,
            abandoned_page_ids=abandoned_page_ids,
        )

        filepath = self._save_with_version_control(merged_df)
        logger.info("Document processing pipeline completed")
        return filepath

    
    # =====================================================
    # dump helpers
    # =====================================================
    def _get_latest_dump_path(self) -> Optional[str]:
        """
        document_processor_dir에서 가장 최신 v_new confluence dump parquet 경로를 반환한다.

        Returns:
            Optional[str]:
                최신 dump 파일 경로.
                아직 dump가 없으면 None.
        """
        dump_pattern = os.path.join(
            self.document_processor_dir,
            f"v_new_{self.filename_prefix}_*.parquet",
        )
        dump_files = glob.glob(dump_pattern)
        if not dump_files:
            return None
        return max(dump_files, key=os.path.getmtime)

    
    def _collect_all_page_ids_from_download_dir(self) -> List[str]:
        """
        download_dir 하위 폴더명에서 page_id를 추출해 전체 page_id 목록을 만든다.

        최초 full build 또는 누락 검사에서 사용된다.
        excluded_ids에 포함된 page는 제외한다.

        Returns:
            List[str]:
                download_dir에 존재하는 page_id 목록.
        """
        page_ids = []
        if not os.path.exists(self.download_dir):
            return page_ids

        for subdir in os.listdir(self.download_dir):
            page_id = self._extract_page_id_from_name(subdir)
            if page_id and page_id not in self.excluded_ids:
                page_ids.append(page_id)

        return sorted(set(page_ids))

    
    # =====================================================
    # extract markdown
    # =====================================================
    def extract_all_documents(self) -> List[Dict]:
        """
        download_dir에 존재하는 모든 page의 markdown을 추출한다.

        Returns:
            List[Dict]:
                dump parquet으로 변환 가능한 document record 목록.
        """
        return self.extract_documents_by_page_ids(self._collect_all_page_ids_from_download_dir())

    
    def extract_documents_by_page_ids(self, target_page_ids: List[str]) -> List[Dict]:
        """
        지정된 page_id 목록에 해당하는 output.md 파일만 읽어 document record로 변환한다.

        다운로드 산출물 경로 구조:
            download_dir/
            └── {sanitized_folder}/
                └── {sanitized_folder}/
                    └── output.md

        Args:
            target_page_ids:
                markdown을 추출할 page_id 목록.

        Returns:
            List[Dict]:
                page_tree, title, page_content, source, image_info, labels를 포함한 record 목록.
        """
        documents = []
        if not os.path.exists(self.download_dir):
            logger.warning("download_dir does not exist: %s", self.download_dir)
            return documents

        target_page_ids = set(map(str, target_page_ids or []))
        if not target_page_ids:
            logger.info("No target page ids for extraction.")
            return documents

        subdirs = os.listdir(self.download_dir)
        for subdir in tqdm(subdirs, desc="Extracting markdown texts"):
            page_id = self._extract_page_id_from_name(subdir)
            if page_id not in target_page_ids:
                continue

            markdown_path = Path(self.download_dir) / subdir / subdir / "output.md"
            if not markdown_path.is_file():
                self.null_pages.append(subdir)
                logger.warning("Markdown file not found: %s", markdown_path)
                continue

            document = self._parse_markdown_document(markdown_path)
            if document:
                documents.append(document)

        if self.null_pages:
            logger.warning("[WARNING] %s pages without markdown files", len(self.null_pages))

        return documents

    def _parse_markdown_document(self, markdown_path: Path) -> Dict:
        """
        output.md 파일 하나를 dump parquet record로 변환한다.

        처리 내용:_parse_markdown_document
            1. markdown text를 읽는다.
            2. 첫 번째 heading에서 title을 추출한다.
            3. <tree>...</tree> 태그에서 page_tree를 추출하고 본문에서 제거한다.
            4. 폴더명에서 page_id를 추출해 source URL을 만든다.
            5. tree_store에서 최신 labels를 가져온다.
            6. markdown 내 img tag를 image_info 목록으로 변환한다.

        Args:
            markdown_path:
                parser가 생성한 output.md 파일 경로.

        Returns:
            Dict:
                dump parquet 한 row에 해당하는 document record.
        """
        try:
            with open(markdown_path, "r", encoding="utf-8") as f:
                text = f.read().replace("\xa0", " ")
        except Exception as e:
            logger.error("[ERROR] Failed to read markdown file: %s - %s", markdown_path, e)
            return {}

        title = self._extract_title(text)
        tree_info, text = self._extract_tree(text)

        sanitized_name = markdown_path.parent.name
        page_id = self._extract_page_id_from_name(sanitized_name)
        source_url = f"{self.source_baseurl}{page_id}"
        
        # label은 markdown 자체가 아니라 최신 tree_store metadata에서 가져온다.
        raw_labels = []
        if self.tree_store is not None:
            node = self.tree_store.get(str(page_id))
            if node is not None:
                raw_labels = list(getattr(node, "labels", []) or [])
        
        labels = parse_confluence_labels(raw_labels)

        if self._is_empty_page(title, text):
            logger.warning("[WARNING] Empty page at %s", sanitized_name)

        image_infos = self._extract_image_info(text, sanitized_name)

        document = {
            "page_tree": tree_info,
            "title": title.replace("\\", ""),
            "page_content": text,
            "source": source_url,
            "image_source": sanitized_name,
            "image_info": image_infos,
            "labels": labels,
        }
        return document

    
    def _extract_image_info(self, text: str, image_source: str) -> List[Dict]:
        """
        markdown text 내 img tag를 image_info 목록으로 변환한다.

        image_info는 이후 caption 생성과 embedding 단계에서 이미지 정보를 참조하는 데 사용된다.

        Args:
            text:
                markdown 본문.

            image_source:
                download_dir 하위 문서 폴더명.
                실제 이미지 경로를 조합하는 데 사용한다.

        Returns:
            List[Dict]:
                alt_text, image_path, not_sanitized_image_path, image_title, image_id, caption 정보를 담은 목록.
        """
        image_infos = []
        extracted_contexts = self.context_extractor.extract_image_contexts(text)
    
        alt_pat = re.compile(r'alt="(.*?)"')
        title_pat = re.compile(r'title="(.*?)"')
    
        for idx, ctx in enumerate(extracted_contexts):
            full_tag = ctx.get("raw_html_tag", "")
    
            alt_match = alt_pat.search(full_tag)
            title_match = title_pat.search(full_tag)
    
            alt = alt_match.group(1) if alt_match else None
            img_title = title_match.group(1) if title_match else None
    
            src = ctx["src_attr"]
            unescaped_src = html.unescape(src)

            non_image_exts = {
                ".ppt",
                ".pptx",
                ".doc",
                ".docx",
                ".pdf"
            }
            
            ext = os.path.splitext(
                unescaped_src.split("?")[0]
            )[1].lower()
            
            if ext in non_image_exts:
                logger.warning(
                    "[SKIP NON-IMAGE ATTACHMENT] "
                    "image_source=%s | idx=%s | ext=%s | src=%s",
                    image_source,
                    idx,
                    ext,
                    unescaped_src,
                )
                continue
            
            image_infos.append({
                "image_id": idx,
                "alt_text": alt,
                "image_title": img_title,
                "image_path": os.path.join(
                    image_source,
                    "images",
                    sanitize_file_name(unescaped_src.split("/")[-1]),
                ),
                "not_sanitized_image_path": os.path.join(
                    image_source,
                    src.replace("&amp;", "&"),
                ),
                "pre_text": ctx.get("pre_text", ""),
                "post_text": ctx.get("post_text", ""),
                "detail": None,
                "caption": None,
            })
    
        return image_infos

    
    # def _extract_image_patterns(self, text: str) -> list[tuple]:
    #     """
    #     markdown/HTML text에서 <img .../> 태그의 alt, src, title 값을 추출한다.

    #     Args:
    #         text:
    #             markdown 본문.

    #     Returns:
    #         list[tuple]:
    #             (alt, src, title) tuple 목록.
    #             src가 없는 img tag는 caption 대상이 될 수 없으므로 제외한다.
    #     """
    #     alt_pat = re.compile(r'alt="(.*?)"')
    #     src_pat = re.compile(r'src="(.*?)"')
    #     title_pat = re.compile(r'title="(.*?)"')
    #     img_tags = re.findall(r"<img .*?/>", text)

    #     matches = []
    #     for tag in img_tags:
    #         alt_match = alt_pat.search(tag)
    #         src_match = src_pat.search(tag)
    #         title_match = title_pat.search(tag)

    #         alt = alt_match.group(1) if alt_match else "None"
    #         src = src_match.group(1) if src_match else "None"
    #         title = title_match.group(1) if title_match else "None"

    #         if src_match:
    #             matches.append((alt, src, title))
    #     return matches

        
    def _extract_title(self, text: str) -> str:
        """
        markdown text에서 문서 제목을 추출한다.

        우선순위:
            1. 첫 번째 '# ...' heading
            2. '#' 기준 split 후 마지막 조각
            3. 첫 번째 line
            4. Untitled

        Args:
            text:
                markdown 본문.

        Returns:
            str:
                추출된 title.
        """
        title_pattern = re.compile(r"#(.*?)\n")
        matches = title_pattern.findall(text)

        if matches:
            return matches[0].strip()

        parts = text.split("#")
        if len(parts) > 1:
            return parts[-1].strip()[:100]

        first_line = text.split("\n")[0].strip()[:100]
        return first_line if first_line else "Untitled"

    
    def _extract_tree(self, text: str) -> tuple:
        """
        markdown text에서 <tree>...</tree> 경로 정보를 추출한다.

        <tree> 태그는 문서의 Confluence 계층 경로를 나타내며,
        embedding 대상 본문에서는 제거한다.

        Args:
            text:
                markdown 본문.

        Returns:
            tuple:
                (page_tree, text_without_tree)
        """
        tree_pattern = re.compile(r"<tree>(.*?)</tree>")
        matches = tree_pattern.findall(text)

        if matches:
            tree = matches[0]
            new_text = text.replace(f"<tree>{tree}</tree>\n", "")
            return tree, new_text
        return "", text

    
    def _is_empty_page(self, title: str, content: str) -> bool:
        """
        title을 제외했을 때 본문이 비어 있는 page인지 판단한다.

        Args:
            title:
                문서 제목.

            content:
                markdown 본문.

        Returns:
            bool:
                제목 외 내용이 없으면 True.
        """
        cleaned = content.replace(f"#{title}", "").replace(" ", "").replace("\n", "")
        return len(cleaned) == 0

    
    def _extract_page_id_from_name(self, text: str) -> str:
        """
        폴더명 또는 문자열 끝의 숫자열을 page_id로 추출한다.

        download_dir 폴더명은 hash와 page_id를 조합한 형태이므로,
        마지막 숫자열을 page_id로 사용한다.

        Args:
            text:
                폴더명 또는 page_id가 포함된 문자열.

        Returns:
            str:
                추출된 page_id. 숫자열이 없으면 원본 문자열.
        """
        match = re.search(r"(\d+)$", str(text))
        return match.group(1) if match else str(text)

    
    # =====================================================
    # merge / incremental update
    # =====================================================
    def _merge_with_existing_data(
        self,
        documents: List[Dict],
        changed_urls: Dict[str, List[str]],
        latest_dump: str,
    ) -> pd.DataFrame:
        """
        기존 dump parquet과 새로 추출한 documents를 병합한다.

        처리 방식:
            1. 기존 dump를 로드한다.
            2. updated / created / deleted source URL 목록을 만든다.
            3. 기존 dump에서 changed/deleted source를 제거한다.
            4. 새로 추출한 documents를 붙인다.
            5. replace_existing_captions=False이면 기존 caption을 재사용한다.

        Args:
            documents:
                full_process 대상에서 새로 추출한 document record 목록.

            changed_urls:
                변경 유형별 page_id 목록.

            latest_dump:
                기존 최신 dump parquet 경로.

        Returns:
            pd.DataFrame:
                변경사항이 반영된 dump dataframe.
        """
        logger.info("Loading existing dump: %s", latest_dump)

        existing_df = pd.read_parquet(latest_dump)
        updated_docs_df = pd.DataFrame(documents)

        all_changed_page_ids = list(
            (
                set(changed_urls.get("updated", []))
                | set(changed_urls.get("created", []))
                | set(changed_urls.get("deleted", []))
            ) - self.excluded_ids
        )
        source_url_list = [
            f"{self.source_baseurl}{page_id}" for page_id in all_changed_page_ids
        ]

        logger.info(
            "Changed pages: %s updated, %s created, %s deleted",
            len(changed_urls.get("updated", [])),
            len(changed_urls.get("created", [])),
            len(changed_urls.get("deleted", [])),
        )

        unchanged_df = existing_df[~existing_df["source"].isin(source_url_list)].copy()

        if updated_docs_df.empty:
            updated_df = unchanged_df.reset_index(drop=True)
            logger.info(
                "No newly extracted docs to merge. Remaining unchanged rows=%s",
                len(updated_df),
            )
            return updated_df

        updated_docs_df = updated_docs_df[updated_docs_df["source"].isin(source_url_list)].copy()

        if not self.replace_existing_captions and not updated_docs_df.empty:
            for idx, row in updated_docs_df.iterrows():
                same_source_df = existing_df[existing_df["source"] == row["source"]]
                if same_source_df.empty:
                    continue
        
                unchanged_image_infos = same_source_df.iloc[0]["image_info"]
                updated_image_infos = row["image_info"]
        
                for updated_image_info in updated_image_infos:
                    for unchanged_image_info in unchanged_image_infos:
                        if (
                            unchanged_image_info["image_id"] == updated_image_info["image_id"]
                            and unchanged_image_info["image_path"] == updated_image_info["image_path"]
                            and unchanged_image_info["image_title"] == updated_image_info["image_title"]
                            and unchanged_image_info["not_sanitized_image_path"]
                            == updated_image_info["not_sanitized_image_path"]
                        ):
                            old_caption = unchanged_image_info.get("caption")
                            old_detail = unchanged_image_info.get("detail")

                            updated_image_info["caption"] = old_caption
                            updated_image_info["detail"] = old_detail
        
                updated_docs_df.at[idx, "image_info"] = updated_image_infos
        updated_df = pd.concat(
            [unchanged_df, updated_docs_df],
            axis=0,
            ignore_index=True,
        ).reset_index(drop=True)

        logger.info(
            "Data merge: %s unchanged + %s changed = %s total",
            len(unchanged_df),
            len(updated_docs_df),
            len(updated_df),
        )
        return updated_df

    
    # =====================================================
    # structure patch
    # =====================================================
    def _apply_structure_patches(
        self,
        merged_df: pd.DataFrame,
        page_ids: List[str],
        latest_tree_pages: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        본문 재처리 없이 title/page_tree metadata만 patch한다.

        사용 대상:
            - moved
            - ancestor_path_changed
            - ancestor_title_changed
            - structure_changed

        Args:
            merged_df:
                기존 dump와 full_process 문서를 merge한 dataframe.

            page_ids:
                structure metadata patch 대상 page id 목록.

            latest_tree_pages:
                tree_store에서 추출한 최신 page metadata dict.

        Returns:
            pd.DataFrame:
                title/page_tree가 patch된 dataframe.
        """
        if merged_df.empty:
            logger.warning("Merged dataframe is empty. Structure patch is skipped.")
            return merged_df

        patched_count = 0
        for page_id in page_ids:
            page_meta = latest_tree_pages.get(str(page_id))
            if not page_meta:
                logger.warning("Latest tree metadata not found for page_id=%s", page_id)
                continue

            new_title = (page_meta.get("title") or "").replace("\\", "")
            new_tree = page_meta.get("page_tree") or ""
            source = f"{self.source_baseurl}{page_id}"

            if source in merged_df["source"].values:
                merged_df.loc[merged_df["source"] == source, "title"] = new_title
                merged_df.loc[merged_df["source"] == source, "page_tree"] = new_tree
                patched_count += 1
            else:
                logger.warning(
                    "Source not found in existing dump for structure patch page_id=%s",
                    page_id,
                )

            markdown_path = self._find_markdown_path_by_page_id(page_id)
            if markdown_path:
                self._patch_markdown_file(
                    markdown_path,
                    new_title=new_title,
                    new_tree=new_tree,
                )
            else:
                logger.warning("Markdown path not found for structure patch page_id=%s", page_id)

        logger.info("Applied structure metadata patch to %s rows", patched_count)
        return merged_df

    
    def _find_markdown_path_by_page_id(self, page_id: str) -> Optional[Path]:
        """
        download_dir에서 특정 page_id에 해당하는 output.md 경로를 찾는다.

        Args:
            page_id:
                찾을 Confluence page id.

        Returns:
            Optional[Path]:
                output.md 경로. 없으면 None.
        """
        if not os.path.exists(self.download_dir):
            return None

        page_id = str(page_id)
        for subdir in os.listdir(self.download_dir):
            if self._extract_page_id_from_name(subdir) != page_id:
                continue
            markdown_path = Path(self.download_dir) / subdir / subdir / "output.md"
            if markdown_path.is_file():
                return markdown_path
        return None

    
    def _patch_markdown_file(self, markdown_path: Path, new_title: str, new_tree: str) -> None:
        """
        output.md 내부의 <tree> 경로와 첫 번째 heading title을 최신 metadata로 patch한다.

        patch 대상:
            - <tree>...</tree>
            - 첫 번째 '# ...' heading

        Args:
            markdown_path:
                patch할 output.md 경로.

            new_title:
                최신 page title.

            new_tree:
                최신 page_tree 경로.
        """
        try:
            text = markdown_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Failed to read markdown for patch: %s - %s", markdown_path, e)
            return

        patched_text = text

        if re.search(r"^<tree>.*?</tree>$", patched_text, flags=re.MULTILINE):
            patched_text = re.sub(
                r"^<tree>.*?</tree>$",
                f"<tree>{new_tree}</tree>",
                patched_text,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            patched_text = f"<tree>{new_tree}</tree>\n" + patched_text

        if re.search(r"^#.*$", patched_text, flags=re.MULTILINE):
            patched_text = re.sub(
                r"^#.*$",
                f"# {new_title}",
                patched_text,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            patched_text = patched_text + f"\n# {new_title}\n"

        if patched_text != text:
            markdown_path.write_text(patched_text, encoding="utf-8")
            logger.info("Patched markdown metadata: %s", markdown_path)

    
    def _load_latest_tree_pages(self) -> Dict[str, Dict]:
        """
        tree_store의 TreeNode들을 page_id 기준 dict로 변환한다.

        structure patch와 label patch는 최신 tree metadata를 기준으로 수행된다.
        tree_store가 없으면 patch를 진행할 수 없으므로 빈 dict를 반환한다.

        Returns:
            Dict[str, Dict]:
                page_id -> page metadata dict.
        """
        if self.tree_store is None:
            logger.warning("tree_store is None. Structure patch will be skipped.")
            return {}

        pages = {}
        for page_id, node in self.tree_store.nodes_by_id.items():
            pages[str(page_id)] = {
                "page_id": str(node.page_id),
                "title": node.title,
                "page_tree": node.page_tree,
                "parent_id": node.parent_id,
                "ancestor_ids": list(node.ancestor_ids),
                "ancestor_titles": list(node.ancestor_titles),
                "page_url": node.page_url,
                "page_url_tiny": node.page_url_tiny,
                "last_modified_date": node.last_modified_date,
                "labels": parse_confluence_labels(list(getattr(node, "labels", []) or [])),
            }
        return pages

    
    # =====================================================
    # apply label patch
    # =====================================================
    def _apply_label_patches(
        self,
        merged_df: pd.DataFrame,
        page_ids: List[str],
        latest_tree_pages: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        본문 재처리 없이 labels 컬럼만 최신 tree metadata 기준으로 patch한다.

        Args:
            merged_df:
                기존 dump와 full_process 문서를 merge한 dataframe.

            page_ids:
                label patch 대상 page id 목록.

            latest_tree_pages:
                tree_store에서 추출한 최신 page metadata dict.

        Returns:
            pd.DataFrame:
                labels가 patch된 dataframe.
        """
        if merged_df.empty:
            logger.warning("Merged dataframe is empty. Label patch is skipped.")
            return merged_df
    
        if "labels" not in merged_df.columns:
            merged_df["labels"] = [
                parse_confluence_labels([])
                for _ in range(len(merged_df))
            ]
    
        patched_count = 0
    
        for page_id in page_ids:
            page_meta = latest_tree_pages.get(str(page_id))
            if not page_meta:
                logger.warning(
                    "Latest tree metadata not found for label patch page_id=%s",
                    page_id,
                )
                continue
    
            source = f"{self.source_baseurl}{page_id}"
            mask = merged_df["source"] == source
    
            if not mask.any():
                logger.warning(
                    "Source not found in existing dump for label patch page_id=%s",
                    page_id,
                )
                continue
    
            new_labels = parse_confluence_labels(page_meta.get("labels"))
    
            merged_df.loc[mask, "labels"] = pd.Series(
                [new_labels for _ in range(mask.sum())],
                index=merged_df.index[mask],
            )
    
            patched_count += int(mask.sum())
    
        logger.info("Applied label patch to %s rows", patched_count)
        return merged_df

    def _normalize_filename_for_match(self, name: str) -> str:
        return (
            str(name)
            .replace("\xa0", " ")
            .replace("%C2%A0", " ")
            .strip()
        )

    def _resolve_image_path(
        self,
        image_source: str,
        rel_image_path: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(rel_image_path, str) or not rel_image_path.strip():
            return None, None
    
        image_path = os.path.join(
            self.download_dir,
            image_source or "",
            rel_image_path,
        )
    
        alt_path = image_path.replace(" ", "\xa0")
    
        # 1. exact path
        if os.path.exists(image_path):
            return image_path, alt_path
    
        # 2. all-space-to-NBSP fallback
        if os.path.exists(alt_path):
            return alt_path, alt_path
    
        # 3. filename normalize fallback
        images_dir = os.path.dirname(image_path)
        target_filename = os.path.basename(rel_image_path)
        target_norm = self._normalize_filename_for_match(target_filename)
    
        if os.path.isdir(images_dir):
            for actual_filename in os.listdir(images_dir):
                actual_norm = self._normalize_filename_for_match(actual_filename)
    
                if actual_norm == target_norm:
                    resolved_path = os.path.join(images_dir, actual_filename)
    
                    logger.info(
                        "[IMAGE PATH RESOLVED BY NORMALIZE] target=%r | actual=%r",
                        target_filename,
                        actual_filename,
                    )
    
                    return resolved_path, alt_path
    
        return None, alt_path
        
    
    def _process_single_image(
        self,
        page_id: str,
        image_source: str,
        page_tree: str,
        page_title: str,
        image: Dict,
    ) -> Dict:
        """
        단일 이미지 captioning 작업을 수행한다.
    
        병렬 실행에서 worker 함수로 사용된다.
        파일 존재 여부, 기존 caption 재사용 여부, API 호출 결과를 status로 반환한다.
        """
        rel_image_path = image.get("not_sanitized_image_path")

        result_record = {
            "page_id": page_id,
            "image_id": image.get("image_id"),
            "image_path": None,
            "status": "fail",
            "error_msg": None,
            "vlm_res": None,
        }

        if not isinstance(rel_image_path, str) or not rel_image_path.strip():
            result_record["status"] = "skip"
            result_record["error_msg"] = "invalid not_sanitized_image_path"
            return result_record

    
        image_path, _ = self._resolve_image_path(
            image_source=image_source,
            rel_image_path=rel_image_path,
        )
        
        if image_path is None:
            result_record["status"] = "not_found"
            result_record["error_msg"] = "image file not found"
            return result_record
        
        result_record["image_path"] = image_path

        
        
        # image_path = os.path.join(self.download_dir, image_source, rel_image_path)
        # result_record["image_path"] = image_path

        # if not os.path.exists(image_path):
        #     alt_path = image_path.replace(" ", "\xa0")

        #     if os.path.exists(alt_path):
        #         image_path = alt_path
        #         result_record["image_path"] = image_path
        #     else:
        #         result_record["status"] = "not_found"
        #         result_record["error_msg"] = "image file not found"
        #         return result_record

        existing_caption = image.get("caption")

        if not self.replace_existing_captions and existing_caption:
            result_record["status"] = "skip"
            return result_record

        full_context = {
            "tree_info": page_tree,
            "document_title": page_title,
            "pre_text": image.get("pre_text", ""),
            "post_text": image.get("post_text", ""),
        }

        time.sleep(random.uniform(0.1, 0.5))

        try:
            vlm_res = self.caption_client.generate_caption(
                image_path=image_path,
                context_dict=full_context,
            )
        
            if vlm_res.get("status") != "success":
                result_record["status"] = "fail"
                result_record["error_msg"] = vlm_res.get(
                    "error_msg",
                    "caption generation failed",
                )
                result_record["vlm_res"] = vlm_res
                return result_record
        
            result_record["status"] = "success"
            result_record["vlm_res"] = vlm_res
            return result_record
        
        except Exception as e:
            result_record["status"] = "fail"
            result_record["error_msg"] = str(e)
            return result_record

    
        # try:
        #     vlm_res = self.caption_client.generate_caption(
        #         image_path=image_path,
        #         context_dict=full_context,
        #     )

        #     summary = str(vlm_res.get("summary", "")).strip()
        #     summary_lower = summary.lower()

        #     if (
        #         not summary
        #         or summary == "Image Not Found"
        #         or summary == "Unknown Error"
        #         or summary.startswith("Error after")
        #         or summary.startswith("Unexpected Error")
        #         or "cannot analysis image file" in summary_lower
        #     ):
        #         result_record["status"] = "fail"
        #         result_record["error_msg"] = summary or "empty summary"
        #         return result_record

        #     result_record["status"] = "success"
        #     result_record["vlm_res"] = vlm_res
        #     return result_record

        # except Exception as e:
        #     result_record["status"] = "fail"
        #     result_record["error_msg"] = str(e)
        #     return result_record

    def generate_captions(
        self,
        df: pd.DataFrame,
        full_caption_target_ids: List[str],
    ) -> Tuple[pd.DataFrame, Set[str]]:
        """
        created / updated page에 대해서만 이미지 caption을 병렬 생성한다.

        이미지 하나라도 실패하면 해당 page_id를 abandoned_page_ids에 추가한다.
        """
        target_ids = set(map(str, full_caption_target_ids or []))

        tasks = []
        abandoned_page_ids: Set[str] = set()

        skip_count = 0
        image_not_found_count = 0
        failed_caption_count = 0
        new_caption_count = 0

        for idx, row in df.iterrows():
            page_id = str(row["source"]).replace(self.source_baseurl, "")

            if page_id not in target_ids:
                continue

            image_infos = row.get("image_info", [])

            if isinstance(image_infos, np.ndarray):
                image_infos = image_infos.tolist()
                df.at[idx, "image_info"] = image_infos

            if image_infos is None or len(image_infos) == 0:
                continue

            for image in image_infos:
                existing_caption = image.get("caption")

                if not self.replace_existing_captions and existing_caption:
                    image.setdefault("detail", None)
                    skip_count += 1
                    continue

                tasks.append(
                    {
                        "idx": idx,
                        "page_id": page_id,
                        "image_source": row.get("image_source"),
                        "page_tree": row.get("page_tree", ""),
                        "page_title": row.get("title", ""),
                        "image": image,
                    }
                )

        if not tasks:
            logger.info(
                "Caption generation completed: %s new, %s skipped, %s images not found, %s failed",
                new_caption_count,
                skip_count,
                image_not_found_count,
                failed_caption_count,
            )
            return df, abandoned_page_ids

        logger.info(
            "Starting parallel captioning with %s workers for %s images",
            self.max_workers,
            len(tasks),
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._process_single_image,
                    task["page_id"],
                    task["image_source"],
                    task["page_tree"],
                    task["page_title"],
                    task["image"],
                ): task
                for task in tasks
            }

            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Generating Image Captions",
            ):
                task = future_to_task[future]
                idx = task["idx"]
                image_id = task["image"].get("image_id")

                try:
                    result = future.result()
                except Exception as e:
                    failed_caption_count += 1
                    abandoned_page_ids.add(str(task["page_id"]))
                    logger.warning(
                        "Caption worker failed: page_id=%s, error=%s",
                        task["page_id"],
                        e,
                    )
                    continue

                status = result.get("status")

                if status == "not_found":
                    image_not_found_count += 1
                    abandoned_page_ids.add(str(result.get("page_id")))
                    logger.warning("Image not found: %s", result.get("image_path"))
                    continue

                if status == "skip":
                    skip_count += 1
                    continue

                if status != "success":
                    failed_caption_count += 1
                    abandoned_page_ids.add(str(result.get("page_id")))
                    logger.warning(
                        "Caption generation failed: page_id=%s, image_id=%s, error=%s",
                        result.get("page_id"),
                        result.get("image_id"),
                        result.get("error_msg"),
                    )
                    continue

                vlm_res = result["vlm_res"]
                summary = str(vlm_res.get("summary", "")).strip()

                image_infos = df.at[idx, "image_info"]

                if isinstance(image_infos, np.ndarray):
                    image_infos = image_infos.tolist()

                target_image = next(
                    (img for img in image_infos if img.get("image_id") == image_id),
                    None,
                )

                if target_image is None:
                    failed_caption_count += 1
                    abandoned_page_ids.add(str(result.get("page_id")))
                    logger.warning(
                        "Target image not found in dataframe: page_id=%s, image_id=%s",
                        result.get("page_id"),
                        image_id,
                    )
                    continue

                # target_image["caption"] = summary
                # target_image["detail"] = vlm_res.get("detail") or None
                target_image["caption"] = self._normalize_caption_field(summary)
                target_image["detail"] = self._normalize_caption_field(vlm_res.get("detail"))
 
                df.at[idx, "image_info"] = image_infos
                new_caption_count += 1

        logger.info(
            "Caption generation completed: %s new, %s skipped, %s images not found, %s failed",
            new_caption_count,
            skip_count,
            image_not_found_count,
            failed_caption_count,
        )

        if abandoned_page_ids:
            logger.warning(
                "Abandoned page ids due to caption failure: %s",
                sorted(abandoned_page_ids),
            )

        return df, abandoned_page_ids

    
    # =====================================================
    # abandoned
    # =====================================================
    def _exclude_and_move_abandoned_pages(
        self,
        merged_df: pd.DataFrame,
        abandoned_page_ids: Set[str],
    ) -> pd.DataFrame:
        """
        caption 실패 page를 최종 parquet 저장 대상에서 제외하고,
        해당 downloaded 폴더를 abandoned_dir로 이동한다.
        """
        abandoned_page_ids = set(map(str, abandoned_page_ids or set()))

        if not abandoned_page_ids:
            return merged_df

        abandoned_sources = {
            f"{self.source_baseurl}{page_id}"
            for page_id in abandoned_page_ids
        }

        filtered_df = merged_df[
            ~merged_df["source"].isin(abandoned_sources)
        ].reset_index(drop=True)

        logger.warning(
            "Excluded abandoned pages from parquet: %s",
            sorted(abandoned_page_ids),
        )

        if self.abandoned_dir:
            self._move_abandoned_downloads(abandoned_page_ids)
        else:
            logger.warning(
                "abandoned_dir is not set. Abandoned downloads were not moved: %s",
                sorted(abandoned_page_ids),
            )

        return filtered_df

    def _move_abandoned_downloads(self, abandoned_page_ids: Set[str]) -> None:
        """
        caption 실패 page의 downloaded/{hash}_{page_id} 폴더 전체를 abandoned_dir로 이동한다.

        abandoned_dir에 동일 폴더가 이미 있으면 기존 폴더를 삭제한 뒤 새로 이동한다.
        """
        if not self.abandoned_dir:
            logger.warning("abandoned_dir is not set.")
            return

        os.makedirs(self.abandoned_dir, exist_ok=True)

        abandoned_page_ids = set(map(str, abandoned_page_ids or set()))

        if not os.path.isdir(self.download_dir):
            logger.warning("download_dir does not exist: %s", self.download_dir)
            return

        for subdir in os.listdir(self.download_dir):
            page_id = self._extract_page_id_from_name(subdir)

            if page_id not in abandoned_page_ids:
                continue

            src_path = os.path.join(self.download_dir, subdir)
            dst_path = os.path.join(self.abandoned_dir, subdir)

            if not os.path.isdir(src_path):
                continue

            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
                logger.warning("Removed existing abandoned directory: %s", dst_path)

            shutil.move(src_path, dst_path)

            logger.warning(
                "Moved abandoned download directory: %s -> %s",
                src_path,
                dst_path,
            )

    
    
    # =====================================================
    # save
    # =====================================================
    def _save_with_version_control(self, df: pd.DataFrame) -> str:
        """
        dump dataframe을 versioned parquet 파일로 저장한다.

        파일명 timestamp 정책:
            - tree_path가 있으면 tree snapshot timestamp를 재사용한다.
            - tree_path에서 timestamp 추출에 실패하면 현재 시간을 사용한다.

        저장 전 manage_version_files를 호출하여 기존 v_new 파일을 v_old로 이동한다.

        Args:
            df:
                저장할 dump dataframe.

        Returns:
            str:
                저장된 parquet 파일 경로.
        """
        timestamp = None

        if self.tree_path:
            try:
                timestamp = extract_timestamp(
                    file_name=self.tree_path,
                    file_prefix="confluence_tree",
                )
            except Exception as e:
                logger.warning(
                    "Failed to extract timestamp from tree_path=%s | %s",
                    self.tree_path,
                    e,
                )

        if not timestamp:
            timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")

        filename = f"v_new_{self.filename_prefix}{timestamp}.parquet"
        filepath = os.path.join(self.document_processor_dir, filename)

        manage_version_files(self.document_processor_dir, self.filename_prefix)
        os.makedirs(self.document_processor_dir, exist_ok=True)
        df = df.copy()
        
        if "image_info" in df.columns:
            df["image_info"] = df["image_info"].apply(self._normalize_image_info_for_parquet)
        
                
        df.to_parquet(filepath, engine="pyarrow", index=False)

        logger.info("[SUCCESS] File saved successfully: %s (%s records)", filepath, len(df))
        return filepath

        # df.to_parquet(filepath, engine="pyarrow", index=False)

        # logger.info("[SUCCESS] File saved successfully: %s (%s records)", filepath, len(df))
        # return filepath

    def _normalize_caption_field(self, value):
        import json
    
        if value is None:
            return None
    
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
    
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
    
        return str(value)
    
    
    def _normalize_image_info_for_parquet(self, image_infos):
        if isinstance(image_infos, np.ndarray):
            image_infos = image_infos.tolist()
    
        if image_infos is None:
            return []
    
        normalized = []
    
        for image in image_infos:
            if not isinstance(image, dict):
                continue
    
            item = dict(image)
            item["caption"] = self._normalize_caption_field(item.get("caption"))
            item["detail"] = self._normalize_caption_field(item.get("detail"))
    
            normalized.append(item)
    
        return normalized

# def validate_labels(labels, context=""):
#     import numpy as np

#     assert isinstance(labels, dict), f"{context} labels must be dict, got {type(labels)}"
#     assert set(labels.keys()) == {"region", "biz", "corp", "extra"}, f"{context} invalid keys: {labels}"

#     for key in ["region", "biz", "corp", "extra"]:
#         assert isinstance(labels[key], (list, np.ndarray)), (
#             f"{context} {key} must be list-like, got {type(labels[key])}"
#         )

#     return True

# def validate_page_in_parquet(df, source_baseurl, page_id, context=""):
#     source = f"{source_baseurl}{page_id}"
#     target_df = df[df["source"] == source]

#     print(f"\n[{context}]")
#     print("page_id:", page_id)
#     print("source :", source)
#     print("found rows:", len(target_df))

#     assert not target_df.empty, f"{context} target source not found: {source}"

#     labels = target_df.iloc[0]["labels"]

#     print(target_df[["source", "title", "labels"]].to_dict("records"))
#     print("labels type:", type(labels))

#     validate_labels(labels, context=context)
#     return target_df.iloc[0]

# def run_one_case(processor, config, case_name, changed_urls, page_id):
#     print("\n" + "=" * 100)
#     print(f"[RUN CASE] {case_name}")
#     print("changed_urls:", changed_urls)
#     print("=" * 100)

#     output_path = processor.process_documents(changed_urls)

#     print("\n[OUTPUT PARQUET]")
#     print(output_path)

#     df = pd.read_parquet(output_path)

#     row = validate_page_in_parquet(
#         df=df,
#         source_baseurl=config.source_baseurl,
#         page_id=page_id,
#         context=case_name,
#     )

#     print(f"[OK] {case_name} passed")
#     return output_path, row

def main():
    import os
    import pandas as pd
    import logging

    from src.pipeline.update_confluence_dump_and_captioning import ConfluenceDocumentProcessor
    from src.components.tree.tree_io import load_tree
    from src.utils.config import init_config
    
    logging.basicConfig(level=logging.INFO)

    config = init_config("config/params_config_dev_gppm.yaml")
    tree_store = load_tree(config.tree_path)

    processor = ConfluenceDocumentProcessor(
        download_dir='./pipeline_storage/downloader/downloaded_gppm',
        abandoned_dir='./pipeline_storage/downloader/abandoned_gppm',
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
        tree_store=tree_store,
        excluded_ids=config.get("excluded_ids", []),
        tree_path=None,
        max_workers=4,
    )

    created_ids = ['2192377929']
    changed_urls = {
        "created": created_ids,
        "updated": [],
        "deleted": [],
        "moved": [],
        "ancestor_path_changed": [],
        "ancestor_title_changed": [],
        "structure_changed": [],
        "labels_changed": [],
    }

    output_path = processor.process_documents(changed_urls)

    print("[DONE] captioned dump saved:")
    print(output_path)

if __name__ == "__main__":
    main()