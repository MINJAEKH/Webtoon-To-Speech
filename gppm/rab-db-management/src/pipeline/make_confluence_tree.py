gppm-rag-db-management
"""
# make_confluence_tree.py
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : Confluence 페이지 계층 트리를 수집하고 변경 사항을 탐지하는 파이프라인 1단계
 *               bootstrap root부터 하위 page를 순회해 tree snapshot을 만들고,
 *               이전 snapshot과 비교하여 created/updated/deleted/moved/labels_changed 이벤트를
 *               담은 compare_results를 생성한다
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
import sys
import time
import re
sys.path.append('./site-packages')

from html import unescape
from collections import deque
from pathlib import Path
from datetime import datetime

from atlassian import Confluence

from src.utils.utils import RetryOnException, manage_version_files, measure_time
from src.utils.confluence_patch import patch_confluence_request

from src.components.tree.tree_models import TreeNode, TreeStore
from src.components.tree.tree_io import load_tree, save_tree

logger = logging.getLogger(__name__)
patch_confluence_request()

class ConfluenceTreeDiscovery:
    """
    Confluence 페이지 계층 구조를 수집하고, 이전 tree snapshot과 비교하여 변경 사항을 탐지하는 클래스.

    이 클래스는 DB 업데이트 파이프라인의 첫 번째 단계에서 사용된다.
    Confluence 전체 문서를 매번 다시 처리하지 않고, tree 구조와 page metadata를 비교하여
    어떤 문서를 새로 다운로드/처리/삭제해야 하는지 판단하는 데 필요한 compare_results를 생성한다.

    주요 역할:
        1. bootstrap root를 기준으로 Confluence page tree를 BFS 방식으로 수집한다.
        2. 저장되어 있던 이전 tree snapshot을 로드한다.
        3. 현재 Confluence 상태와 이전 tree를 비교해 변경 계획(change_plan)을 만든다.
        4. 변경 계획을 기반으로 기존 tree를 patch / refresh / rebuild 한다.
        5. old_tree와 new_tree를 비교하여 downstream 단계에서 사용할 compare_results를 만든다.

    변경 감지 범위:
        - created: 이전 tree에는 없고 새 tree에는 있는 page
        - deleted: 이전 tree에는 있고 새 tree에는 없는 page
        - updated: last_modified_date가 변경된 page
        - moved: parent_id가 변경된 page
        - children_changed: direct child 구성이 변경된 parent page
        - ancestor_path_changed: ancestor id 경로가 변경된 page
        - ancestor_title_changed: ancestor title 경로가 변경된 page
        - labels_changed: label 목록이 변경된 page

    동작 모드:
        - Initial full build:
            기존 tree snapshot이 없을 때 bootstrap root 전체를 새로 수집한다.
        - Incremental update:
            기존 tree를 기준으로 현재 상태를 비교하고, 필요한 부분만 갱신한다.

    Incremental update의 핵심 전략:
        1. parent별 direct child 목록 비교로 구조 변경을 감지한다.
        2. added / removed child 정보를 이용해 deleted와 moved를 구분한다.
        3. 수정일 또는 label이 바뀐 page는 detail fetch로 최신화한다.
        4. 기존 leaf 아래에 새 subtree가 생긴 경우에는 subtree rebuild로 보정한다.
    """

    # parent의 child 수가 이 값 이상이면 CQL 결과 대신 generator fetch를 사용한다.
    # child 수가 많은 parent에서는 CQL 응답 크기/페이지네이션 이슈를 줄이기 위한 fallback 기준이다.
    
    GENERATOR_THRESHOLD = 200

    
    def __init__(
        self,
        page_root: str,
        confluence_token: str,
        tree_path: str | None = None,
        max_bfs_depth: int | None = None,
        bootstrap_roots: dict | None = None,
        tree_filename_prefix: str = "confluence_tree",
    ):
        """
        Confluence tree discovery 실행에 필요한 경로, 인증 정보, 탐색 root를 초기화한다.

        Args:
            page_root: Confluence base URL.
            confluence_token: Confluence API token.
            tree_path: tree snapshot이 저장될 디렉터리.
            max_bfs_depth: BFS 탐색 최대 depth. None이면 제한하지 않는다.
            bootstrap_roots: tree 탐색 시작점으로 사용할 root page 목록.
                일반적으로 {root_name: page_id} 형태를 기대한다.
            tree_filename_prefix: 저장되는 tree snapshot 파일 prefix.
        """

        
        self.page_root = str(page_root)
        # bootstrap_roots는 config에서 넘어오며, key/value 타입이 섞일 수 있으므로 문자열로 통일한다.
        # allowed_root_ids는 실제 BFS 시작점으로 사용할 page_id 집합이다.
        self.bootstrap_roots = {str(k): str(v) for k, v in (bootstrap_roots or {}).items()}
        self.allowed_root_ids = set(self.bootstrap_roots.values())

        self.confluence_token = confluence_token
        if not tree_path:
            raise ValueError("tree_path is required")
        self.tree_dir = tree_path 
        self.max_bfs_depth = max_bfs_depth
        self.tree_filename_prefix = tree_filename_prefix
        
        # 최신 tree는 tree_dir에 저장하고, 이전 버전은 v_old 하위에 보관한다.
        os.makedirs(self.tree_dir, exist_ok=True)
        os.makedirs(os.path.join(self.tree_dir, "v_old"), exist_ok=True)

        self.confluence_api = None
        self._initialize_confluence_api()

    
    # =========================================================
    # public
    # =========================================================
    @measure_time
    def prepare(self) -> dict:

        """
        Confluence tree를 생성 또는 증분 업데이트하고, 변경 비교 결과를 반환한다.

        처리 흐름:
            1. 저장된 기존 tree snapshot을 로드한다.
            2. 기존 tree가 없으면 initial full build를 수행한다.
            3. 기존 tree가 있으면 현재 Confluence 상태를 기준으로 change_plan을 만든다.
            4. change_plan의 tree_changes를 기존 tree에 patch한다.
            5. pages_to_refresh는 detail fetch로 최신 metadata를 반영한다.
            6. subtree_roots_to_rebuild는 BFS로 subtree를 다시 생성한 뒤 기존 tree에 병합한다.
            7. old_tree와 new_tree를 비교하여 compare_results를 생성한다.
            8. new_tree를 versioned snapshot으로 저장한다.

        Returns:
            dict:
                tree_path: 저장된 최신 tree 파일 경로. 변경이 없으면 None.
                scan_stats: discovery 실행 통계.
                changed_root_ids: subtree rebuild 대상 root 목록.
                compare_results: page_id별 변경 이벤트.
                old_tree: 이전 tree snapshot.
                new_tree: 최신 tree snapshot.
                change_plan: tree patch / detail refresh / subtree rebuild 계획.
        """

        
        logger.info("[TREE DISCOVERY] loading tree from dir: %s", self.tree_dir)
        
        old_tree = load_tree(self.tree_dir, self.tree_filename_prefix)
        has_old_tree = len(old_tree) > 0
    
        root_ids = sorted(self.allowed_root_ids)
        if not root_ids:
            raise ValueError("bootstrap_roots is required for tree-based prepare()")
    
        # -----------------------------------------------------
        # initial full build
        # -----------------------------------------------------
        # 기존 tree snapshot이 없는 최초 실행 모드.
        # bootstrap root 전체를 BFS로 수집하여 기준 tree를 만든다.
        if not has_old_tree:
            logger.info("[TREE DISCOVERY] initial full build mode")
            logger.info("[TREE DISCOVERY] root_ids=%s", root_ids)

            new_tree = self.build_tree_from_roots(
                root_ids=root_ids,
                include_body=True,
            )
           
            # 최초 생성 시에는 비교 대상 old_tree가 없으므로 compare_results는 비워 둔다.
            # 이후 pipeline에서는 이 tree snapshot을 기준으로 다음 실행부터 증분 비교한다.
            compare_results = {}
            scan_mode = "initial_tree_build"
            
            # 기존 최신 tree 파일을 v_old로 이동한 뒤 새 tree를 저장한다.
            manage_version_files(self.tree_dir, self.tree_filename_prefix)
    
            saved_tree_path = save_tree(
                self.tree_dir,
                new_tree,
                metadata={
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "root_source": scan_mode,
                },
                filename_prefix=self.tree_filename_prefix,
            )
    
            logger.info(
                "[BRANCH DISCOVERY DONE] mode=%s | node_count=%s | compare_count=%s",
                scan_mode,
                len(new_tree),
                len(compare_results),
            )
    
            return {
                "tree_path": saved_tree_path,
                "scan_stats": {
                    "scan_mode": scan_mode,
                    "children_updated_count": 0,
                    "subtrees_deleted_count": 0,
                    "subtrees_moved_count": 0,
                    "page_refresh_count": 0,
                    "subtree_rebuild_count": len(root_ids),
                    "node_count": len(new_tree),
                },
                "changed_root_ids": root_ids,
                "compare_results": compare_results,
                "old_tree": old_tree,
                "new_tree": new_tree,
                "change_plan": {},
            }
    
        # -----------------------------------------------------
        # incremental tree update mode
        # -----------------------------------------------------
        logger.info("[BRANCH DISCOVERY] incremental tree update mode")
        
        # 기존 tree와 현재 Confluence 상태를 비교하여 변경 계획을 만든다.
        # change_plan은 크게 두 영역으로 나뉜다.
        #   1. tree_changes: 기존 tree에 직접 patch할 수 있는 구조 변경
        #   2. refresh_targets: detail fetch 또는 subtree rebuild가 필요한 대상    
        change_plan = self._build_change_plan_from_old_tree(old_tree)
    
        # tree에 바로 반영 가능한 구조 변경 목록.
        # parent의 children 변경, subtree 삭제, subtree 이동을 포함한다.
        tree_changes = change_plan["tree_changes"]
        children_updated = tree_changes["children_updated"]
        subtrees_deleted = tree_changes["subtrees_deleted"]
        subtrees_moved = tree_changes["subtrees_moved"]
        
        # Confluence API를 다시 호출해야 최신화할 수 있는 대상.
        # pages_to_refresh는 page metadata 갱신용이고,
        # subtree_roots_to_rebuild는 구조 자체를 다시 BFS로 구성해야 하는 대상이다.
        refresh_targets = change_plan["refresh_targets"]
        pages_to_refresh = refresh_targets["pages_to_refresh"]
        subtree_roots_to_rebuild = refresh_targets["subtree_roots_to_rebuild"]
    
        logger.info(
            "[Tree DISCOVERY] children_updated=%s | subtrees_deleted=%s | subtrees_moved=%s | pages_to_refresh=%s | subtree_roots_to_rebuild=%s",
            len(children_updated),
            len(subtrees_deleted),
            len(subtrees_moved),
            len(pages_to_refresh),
            len(subtree_roots_to_rebuild),
        )
        
        # 구조 변경도 없고 metadata refresh/rebuild 대상도 없으면 tree를 저장하지 않는다.
        # downstream 단계에서 변경 없음으로 처리할 수 있도록 compare_results도 비워서 반환한다.
        no_tree_changes = (
            not children_updated
            and not subtrees_deleted
            and not subtrees_moved
            and not pages_to_refresh
            and not subtree_roots_to_rebuild
        )
    
        if no_tree_changes:
            logger.info("[Tree DISCOVERY] no changes detected")
    
            return {
                "tree_path": None,
                "scan_stats": {
                    "scan_mode": "tree_compare_no_change",
                    "children_updated_count": 0,
                    "subtrees_deleted_count": 0,
                    "subtrees_moved_count": 0,
                    "page_refresh_count": 0,
                    "subtree_rebuild_count": 0,
                    "node_count": len(old_tree),
                },
                "changed_root_ids": [],
                "compare_results": {},
                "old_tree": old_tree,
                "new_tree": old_tree,
                "change_plan": change_plan,
            }
        
        # 1) tree 구조에 대한 직접 patch 적용.
        # children_ids 변경, subtree 삭제, subtree 이동은 기존 tree 객체에 직접 반영 가능하다.
        patched_tree = self._apply_change_plan_to_tree(old_tree, change_plan)
    
        # 2) 메타데이터가 바뀐 페이지는 상세 조회로 최신화한다.
        # child row만으로는 title, body 존재 여부, URL 등 모든 값을 신뢰하기 어렵기 때문에
        # refresh 대상으로 잡힌 page는 fetch_page_node(include_body=True)로 다시 읽는다.
        if pages_to_refresh:
            detail_nodes = self._fetch_detail_nodes(pages_to_refresh)
            patched_tree = self._apply_detail_nodes(patched_tree, detail_nodes)
    
        # 3) 구조가 크게 변한 루트는 subtree 단위로 재빌드한다.
        # 예를 들어 기존 leaf 아래에 새 child가 생기면 direct children 비교만으로는
        # 하위 전체 구조를 안정적으로 복원하기 어렵기 때문에 해당 leaf를 root로 다시 BFS 탐색한다.
        if subtree_roots_to_rebuild:
            partial_tree = self.build_tree_from_roots(
                root_ids=subtree_roots_to_rebuild,
                include_body=True,
            )
    
            new_tree = self._merge_updated_subtrees(
                old_tree=patched_tree,
                partial_tree=partial_tree,
                changed_root_ids=subtree_roots_to_rebuild,
            )
        else:
            new_tree = patched_tree
        
        # 최종 old_tree와 new_tree를 비교하여 downstream changed_urls 생성에 사용할 이벤트를 만든다.
        # 실제 다운로드/삭제/DB sync 기준은 이 compare_results에서 파생된다.
        compare_results = self._compare_trees(old_tree, new_tree)
    
        manage_version_files(self.tree_dir, self.tree_filename_prefix)
    
        saved_tree_path = save_tree(
            self.tree_dir,
            new_tree,
            metadata={
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "root_source": "tree_delta_refresh",
            },
            filename_prefix=self.tree_filename_prefix,
        )
    
        logger.info(
            "[Tree DISCOVERY DONE] mode=%s | node_count=%s | children_updated=%s | subtrees_deleted=%s | subtrees_moved=%s | pages_to_refresh=%s | subtree_rebuild_count=%s | compare_count=%s",
            "tree_delta_refresh",
            len(new_tree),
            len(children_updated),
            len(subtrees_deleted),
            len(subtrees_moved),
            len(pages_to_refresh),
            len(subtree_roots_to_rebuild),
            len(compare_results),
        )
    
        return {
            "tree_path": saved_tree_path,
            "scan_stats": {
                "scan_mode": "tree_delta_refresh",
                "children_updated_count": len(children_updated),
                "subtrees_deleted_count": len(subtrees_deleted),
                "subtrees_moved_count": len(subtrees_moved),
                "page_refresh_count": len(pages_to_refresh),
                "subtree_rebuild_count": len(subtree_roots_to_rebuild),
                "node_count": len(new_tree),
            },
            "changed_root_ids": subtree_roots_to_rebuild,
            "compare_results": compare_results,
            "old_tree": old_tree,
            "new_tree": new_tree,
            "change_plan": change_plan,
        }
    
    
    # =========================================================
    # init
    # =========================================================
    @RetryOnException()
    def _initialize_confluence_api(self):
        """
        Confluence API client를 초기화한다.

        네트워크/API 연결 실패가 발생할 수 있으므로 RetryOnException을 적용한다.
        """
        self.confluence_api = Confluence(
            url=self.page_root,
            token=self.confluence_token,
        )
        logger.info("Confluence API connection successful (branch discovery)")

    
    # =========================================================
    # utils
    # =========================================================
    def _make_page_tree(self, ancestor_titles: list[str], title: str) -> str:
        """
        ancestor title 목록과 현재 page title을 사람이 읽기 쉬운 경로 문자열로 만든다.

        Example:
            ["A", "B"], "C" -> "A > B > C"
        """
        return " > ".join(list(ancestor_titles) + [title])

    
    def extract_meaningful_text_from_storage(self, body_storage: str) -> str:
        """
        Confluence storage format HTML에서 실제 의미 있는 텍스트만 추출한다.

        Confluence body.storage에는 macro, ri 태그, HTML 태그 등이 섞여 있다.
        본문 존재 여부를 판단할 때 이런 구조적 태그만 있는 page를 본문 있는 page로 오판하지 않도록
        macro와 태그를 제거한 뒤 남는 텍스트를 기준으로 판단한다.
        """
        if not body_storage:
            return ""

        text = body_storage
        text = text.replace("\xa0", " ")

        # self-closing macro(toc 등)를 먼저 제거
        text = re.sub(r"<ac:structured-macro\b[^>]*/>", " ", text)
        # 여는 태그 + 닫는 태그 쌍인 macro만 제거: 여는 태그를 [^>]*> 로 한정해 태그 경계를 넘어 매칭되지 않게 한다.
        text = re.sub(
            r"<ac:structured-macro\b[^>]*>.*?</ac:structured-macro>",
            " ",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(r"<ac:plain-text-body\b.*?</ac:plain-text-body>", " ", text, flags=re.DOTALL)

        # Confluence resource identifier 태그와 일반 HTML 태그를 제거한다.
        text = re.sub(r"<ri:[^>]+/>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)

        # unescape는 태그 제거 뒤에 수행: <1>와 같은 경우 태그로 오인되지 않도록 한다.
        text = unescape(text)

        # 공백을 정리해 최종 텍스트만 남긴다.
        text = re.sub(r"\s+", " ", text).strip()

        return text

    
    def has_meaningful_body(self, body_storage: str) -> bool:
        """
        Confluence page가 실제 본문을 가지는지 판단한다.

        - 매크로/태그 제거 후 텍스트가 남으면 True
        - 텍스트가 없어도 첨부 이미지(<ac:image><ri:attachment>)가 있으면 True
          (조직도처럼 이미지만으로 구성된 문서는 캡션으로 검색 가능하므로 본문으로 본다)
        - toc, children 같은 구조용 매크로만 있으면 False
        """
        if self.extract_meaningful_text_from_storage(body_storage):
            return True
        return bool(re.search(r"<ac:image\b[^>]*>\s*<ri:attachment\b", body_storage or ""))

    
    # =========================================================
    # direct child fetch
    # =========================================================
    def _get_direct_children_from_confluence(
        self,
        parent_id: str,
        page_limit: int = 200,
    ) -> tuple[list[dict], dict]:
        
        """
        특정 parent page의 direct child 목록을 Confluence API에서 조회한다.

        이 함수는 tree build 단계에서 child page_id 목록을 얻기 위한 lightweight 조회 함수다.
        page_id와 title만 정규화해서 반환하며, pagination과 중복 제거를 함께 처리한다.

        Args:
            parent_id: direct child를 조회할 parent page id.
            page_limit: Confluence API pagination limit.

        Returns:
            tuple:
                normalized_children:
                    [{"page_id": str, "title": str}, ...]
                meta:
                    parent_id, elapsed_seconds, error 정보를 담은 dict.
        """
        
        
        parent_id = str(parent_id)
        normalized_children = []
        seen_child_ids = set()

        meta = {
            "parent_id": parent_id,
            "elapsed_seconds": 0.0,
            "error": None,
        }

        started_at = time.perf_counter()

        try:
            start = 0

            while True:
                results = self.confluence_api.get_page_child_by_type(
                    parent_id,
                    type="page",
                    start=start,
                    limit=page_limit,
                )
                # atlassian-python-api 버전/호출 방식에 따라 dict 또는 list/generator 형태가 올 수 있다.
                # 두 형태를 모두 batch list로 맞춰 아래 로직을 공통 처리한다.
                if isinstance(results, dict):
                    batch = results.get("results", []) or []
                    size = results.get("size")
                else:
                    batch = results or []
                    size = len(batch)

                if not batch:
                    break

                for child in batch:
                    child_id = str(child.get("id"))
                    if not child_id or child_id in seen_child_ids:
                        continue

                    title = child.get("title", child_id)
                    seen_child_ids.add(child_id)

                    normalized_children.append({
                        "page_id": child_id,
                        "title": title,
                    })
                # size가 없으면 batch 길이를 기준으로 마지막 페이지 여부를 판단한다.
                if size is None:
                    if len(batch) < page_limit:
                        break
                    start += page_limit
                    continue

                if len(batch) < page_limit:
                    break

                start += page_limit

        except Exception as e:
            meta["error"] = str(e)
            logger.error(
                "[branch discovery] Failed to get direct children for %s: %s",
                parent_id,
                e,
            )

        meta["elapsed_seconds"] = time.perf_counter() - started_at
        return normalized_children, meta

    
    def fetch_children_ids(self, page_id: str) -> list[str]:
        """
        특정 page의 direct child page_id 목록만 반환한다.

        fetch_page_node에서 children_ids를 채우기 위해 사용한다.
        """
        page_id = str(page_id)
        children, _ = self._get_direct_children_from_confluence(page_id)
        return [str(child["page_id"]) for child in children if child.get("page_id")]

    
    # =========================================================
    # page fetch
    # =========================================================
    def fetch_page_meta(self, page_id: str) -> dict:
        """
        특정 page의 metadata만 조회한다.

        move 판별처럼 본문 정보가 필요 없는 경우에 사용한다.
        body.storage를 expand하지 않기 때문에 fetch_page_node(include_body=True)보다 가볍다.

        Args:
            page_id: 조회할 Confluence page id.

        Returns:
            dict:
                title, parent_id, ancestor_ids, ancestor_titles, depth,
                last_modified_date, page_tree, page_url, page_url_tiny, labels.
        """        

        
        page_id = str(page_id)
        page = self.confluence_api.get_page_by_id(
            page_id,
            expand="ancestors,version,metadata.labels",
        )
        if not page:
            raise ValueError(f"page not found: {page_id}")

        title = str(page.get("title", page_id))

        ancestors = page.get("ancestors", []) or []
        ancestor_ids = [str(x.get("id")) for x in ancestors if x.get("id")]
        ancestor_titles = [str(x.get("title", x.get("id"))) for x in ancestors if x.get("id")]
        parent_id = ancestor_ids[-1] if ancestor_ids else None

        version_info = page.get("version", {}) or {}
        last_modified_date = self._normalize_datetime(version_info.get("when"))

        page_links = page.get("_links", {}) or {}
        webui = page_links.get("webui")
        tinyui = page_links.get("tinyui")
        base = page_links.get("base") or self.page_root.rstrip("/")

        page_url = f"{base}{webui}" if webui else None
        page_url_tiny = f"{base}{tinyui}" if tinyui else None

        depth = len(ancestor_ids) + 1
        page_tree = self._make_page_tree(ancestor_titles, title)
        labels = self._extract_labels_from_page(page)

        return {
            "page_id": page_id,
            "title": title,
            "parent_id": parent_id,
            "ancestor_ids": ancestor_ids,
            "ancestor_titles": ancestor_titles,
            "depth": depth,
            "last_modified_date": last_modified_date,
            "page_tree": page_tree,
            "page_url": page_url,
            "page_url_tiny": page_url_tiny,
            "labels": labels,
        }

    
    def fetch_page_node(self, page_id: str, include_body: bool = False) -> TreeNode:
        """
        특정 Confluence page를 TreeNode 형태로 조회한다.

        TreeNode는 tree snapshot 저장과 비교에 사용하는 표준 데이터 구조다.
        include_body=True이면 body.storage를 함께 조회하여 has_body와 is_container를 계산한다.

        Args:
            page_id: 조회할 Confluence page id.
            include_body: 본문 존재 여부를 함께 계산할지 여부.

        Returns:
            TreeNode: page metadata, children, ancestor, label 정보를 담은 node 객체.
        """

        
        page_id = str(page_id)

        expand_fields = ["ancestors", "version", "metadata.labels"]
        if include_body:
            expand_fields.append("body.storage")

        page = self.confluence_api.get_page_by_id(
            page_id,
            expand=",".join(expand_fields),
        )
        if not page:
            raise ValueError(f"page not found: {page_id}")

        title = str(page.get("title", page_id))

        ancestors = page.get("ancestors", []) or []
        ancestor_ids = [str(x.get("id")) for x in ancestors if x.get("id")]
        ancestor_titles = [str(x.get("title", x.get("id"))) for x in ancestors if x.get("id")]
        parent_id = ancestor_ids[-1] if ancestor_ids else None

        version_info = page.get("version", {}) or {}
        last_modified_date = self._normalize_datetime(version_info.get("when"))
        # include_body=True일 때만 본문 존재 여부를 계산한다.
        # body 조회는 상대적으로 무겁기 때문에 metadata만 필요한 경우에는 생략한다.
        has_body = None
        if include_body:
            body_storage = (((page.get("body") or {}).get("storage") or {}).get("value")) or ""
            has_body = self.has_meaningful_body(body_storage)

        children_ids = self.fetch_children_ids(page_id)
        has_children = bool(children_ids)
        
        # 본문은 없지만 child가 있는 page는 문서라기보다 구조용 container로 본다.
        # downstream downloader/processor에서 본문 없는 container page를 제외하는 기준으로 사용한다.
        if include_body:
            is_container = bool(has_children and not has_body)
        else:
            is_container = False

        page_links = page.get("_links", {}) or {}
        webui = page_links.get("webui")
        tinyui = page_links.get("tinyui")
        base = page_links.get("base") or self.page_root.rstrip("/")

        page_url = f"{base}{webui}" if webui else None
        page_url_tiny = f"{base}{tinyui}" if tinyui else None

        depth = len(ancestor_ids) + 1
        page_tree = self._make_page_tree(ancestor_titles, title)
        labels = self._extract_labels_from_page(page)

        return TreeNode(
            page_id=page_id,
            title=title,
            parent_id=parent_id,
            children_ids=children_ids,
            ancestor_ids=ancestor_ids,
            ancestor_titles=ancestor_titles,
            depth=depth,
            last_modified_date=last_modified_date,
            has_body=has_body,
            has_children=has_children,
            is_container=is_container,
            page_tree=page_tree,
            page_url=page_url,
            page_url_tiny=page_url_tiny,
            labels=labels,
        )

    
    # =========================================================
    # current row collection
    # =========================================================
    def _minimize_seed_roots(self, candidate_ids: set[str], tree: TreeStore) -> set[str]:
        """
        subtree rebuild 후보 중 중첩되는 하위 root를 제거한다.

        예를 들어 A와 A/B가 동시에 rebuild 후보라면 A만 rebuild해도 A/B가 포함된다.
        따라서 ancestor가 이미 후보에 포함된 page는 제거하여 중복 BFS를 방지한다.
        """        
        
        minimal = set()

        for page_id in candidate_ids:
            node = tree.get(page_id)
            if not node:
                minimal.add(page_id)
                continue

            ancestor_ids = set(map(str, node.ancestor_ids))
            has_ancestor_in_candidates = any(ancestor_id in candidate_ids for ancestor_id in ancestor_ids)

            if not has_ancestor_in_candidates:
                minimal.add(page_id)

        return minimal

    
    # =========================================================
    # tree build
    # =========================================================
    def build_tree_from_roots(self, root_ids: list[str], include_body: bool = False) -> TreeStore:
        """
        주어진 root page 목록을 기준으로 Confluence tree를 BFS 방식으로 생성한다.

        처리 방식:
            1. root_ids를 queue에 넣는다.
            2. queue에서 page_id를 하나씩 꺼내 TreeNode로 조회한다.
            3. 조회된 node를 TreeStore에 추가한다.
            4. node.children_ids를 queue에 추가하여 하위 page를 계속 탐색한다.
            5. 탐색 완료 후 parent-child link와 bottom parent 정보를 계산한다.

        Args:
            root_ids: tree 생성 시작점이 되는 page id 목록.
            include_body: 본문 존재 여부까지 조회할지 여부.

        Returns:
            TreeStore: page_id 기준으로 TreeNode를 저장한 tree 객체.
        """
        
        tree = TreeStore()
        queue = deque(str(x) for x in root_ids if x)
        visited = set()

        started_at = time.perf_counter()

        while queue:
            page_id = str(queue.popleft())
            if page_id in visited:
                continue
            visited.add(page_id)

            try:
                node = self.fetch_page_node(page_id, include_body=include_body)
            except Exception as e:
                # 일부 page 조회 실패가 전체 tree build 실패로 이어지지 않도록 해당 page만 skip한다.
                logger.error("[build_tree_from_roots] skip page_id=%s | error=%s", page_id, e)
                continue
            # max_bfs_depth가 설정된 경우 지정 depth보다 깊은 node는 tree에 포함하지 않는다.
            if self.max_bfs_depth is not None and node.depth is not None and node.depth > self.max_bfs_depth:
                continue

            tree.add_node(node)

            for child_id in node.children_ids:
                child_id = str(child_id)
                if child_id not in visited:
                    queue.append(child_id)
        # TreeStore 내부의 parent-child 관계와 bottom parent 플래그를 갱신한다.
        tree.build_links()
        tree.mark_bottom_parents()

        logger.info(
            "[build_tree_from_roots] root_count=%s | node_count=%s | elapsed=%.3f",
            len(root_ids),
            len(tree),
            time.perf_counter() - started_at,
        )
        return tree

    
    # =========================================================
    # patch / detail apply
    # =========================================================
    def _clone_tree(self, tree: TreeStore) -> TreeStore:
        """
        TreeStore를 복제한다.

        patch 과정에서 old_tree를 직접 수정하면 이후 compare 기준이 깨진다.
        따라서 변경 적용 전에는 항상 clone을 만들고, clone에만 patch를 적용한다.
        """        
        cloned = TreeStore()
        for node in tree.iter_nodes():
            cloned.add_node(self._clone_node(node))
        cloned.build_links()
        cloned.mark_bottom_parents()
        return cloned

    
    def _clone_node(self, node: TreeNode) -> TreeNode:
        """
        TreeNode를 새 객체로 복제한다.

        list 타입 필드는 shallow copy가 공유 참조를 만들 수 있으므로 list(...)로 새 리스트를 만든다.
        """
        return TreeNode(
            page_id=node.page_id,
            title=node.title,
            parent_id=node.parent_id,
            children_ids=list(node.children_ids),
            ancestor_ids=list(node.ancestor_ids),
            ancestor_titles=list(node.ancestor_titles),
            depth=node.depth,
            last_modified_date=node.last_modified_date,
            has_body=node.has_body,
            has_children=node.has_children,
            is_container=node.is_container,
            is_bottom_parent=getattr(node, "is_bottom_parent", False),
            page_tree=node.page_tree,
            page_url=node.page_url,
            page_url_tiny=node.page_url_tiny,
            labels=list(node.labels),
        )

    
    def _fetch_detail_nodes(self, page_ids: list[str]) -> dict[str, TreeNode]:
        """
        metadata refresh 대상 page들을 최신 TreeNode로 조회한다.

        각 page는 include_body=True로 조회한다.
        이 단계에서 title, label, last_modified_date, has_body, children_ids 등을 최신 상태로 가져온다.
        실패한 page는 로그만 남기고 제외하여 전체 discovery가 중단되지 않게 한다.
        """
        fetched = {}

        for page_id in page_ids:
            try:
                node = self.fetch_page_node(page_id, include_body=True)
                fetched[str(page_id)] = node
            except Exception as e:
                logger.error("[detail fetch skipped] failed page_id=%s | error=%s", page_id, e)

        return fetched

    def _apply_detail_nodes(self, tree: TreeStore, detail_nodes: dict[str, TreeNode]) -> TreeStore:
        """
        detail fetch로 가져온 최신 TreeNode 정보를 기존 tree에 반영한다.

        갱신 대상:
            - title
            - parent_id
            - ancestor_ids / ancestor_titles
            - depth
            - last_modified_date
            - has_body / has_children / is_container
            - page_tree / page_url / page_url_tiny
            - children_ids
            - labels

        중요한 이유:
            child row 비교만으로는 page의 모든 metadata가 정확히 최신이라고 보기 어렵다.
            그래서 수정일/label/added/moved 등으로 refresh 대상이 된 page는 detail fetch 결과로 덮어쓴다.

        title 변경 처리:
            특정 page의 title이 바뀌면 descendant의 ancestor_titles와 page_tree도 함께 바뀌어야 한다.
            현재 node를 먼저 갱신한 뒤, title_changed_items를 기반으로 descendant에 경로 변경을 전파한다.
        """
        
        new_tree = self._clone_tree(tree)

        # title 변경은 현재 node뿐 아니라 모든 descendant의 page_tree에 영향을 준다.
        # node 갱신 중에 즉시 전파하면 중간 상태가 섞일 수 있으므로 변경 목록을 모아두고 마지막에 처리한다.
        title_changed_items = []
    
        for page_id, new_node in detail_nodes.items():
            old_node = new_tree.get(page_id)
            # 기존 tree에 없던 page라면 새 node로 추가한다.
            # parent가 이미 tree에 있으면 parent.children_ids에도 연결해준다.
            if old_node is None:
                new_tree.add_node(self._clone_node(new_node))
    
                if new_node.parent_id:
                    parent = new_tree.get(new_node.parent_id)
                    if parent and page_id not in parent.children_ids:
                        parent.children_ids.append(page_id)
                continue
            # detail fetch 결과를 기준으로 metadata를 최신화한다.    
            old_title = old_node.title
    
            old_node.title = new_node.title
            old_node.parent_id = new_node.parent_id
            old_node.ancestor_ids = list(new_node.ancestor_ids)
            old_node.ancestor_titles = list(new_node.ancestor_titles)
            old_node.depth = new_node.depth
            old_node.last_modified_date = new_node.last_modified_date
            if new_node.has_body is not None:
                old_node.has_body = new_node.has_body
                
            old_node.has_children = new_node.has_children
            old_node.is_container = new_node.is_container
            old_node.page_tree = new_node.page_tree
            old_node.page_url = new_node.page_url
            old_node.page_url_tiny = new_node.page_url_tiny
            old_node.children_ids = list(new_node.children_ids)
            old_node.labels = list(new_node.labels)
            
            # title이 바뀐 경우 descendant의 ancestor_titles/page_tree에 전파해야 한다.
            if old_title != new_node.title:
                title_changed_items.append({
                    "page_id": str(page_id),
                    "old_title": str(old_title),
                    "new_title": str(new_node.title),
                })
        # title 변경 propagation.
        # descendant의 구조 정보(parent_id, ancestor_ids)는 그대로 두고 title 기반 경로 문자열만 보정한다.
        for item in title_changed_items:
            self._propagate_title_change_to_descendants(
                tree=new_tree,
                page_id=item["page_id"],
                old_title=item["old_title"],
                new_title=item["new_title"],
            )
    
        new_tree.build_links()
        new_tree.mark_bottom_parents()
        return new_tree

    def _apply_change_plan_to_tree(self, old_tree: TreeStore, change_plan: dict) -> TreeStore:
        """
        change_plan["tree_changes"]에 담긴 구조 변경을 old_tree에 적용한다.

        이 함수는 '직접 반영 가능한 tree 구조 변경'만 처리한다.
        page metadata 최신화는 _apply_detail_nodes에서 처리한다.

        처리 순서:
            1. parent의 children_ids 목록 갱신
            2. 삭제된 subtree 제거
            3. 이동된 subtree의 parent / ancestor 경로 갱신

        Args:
            old_tree: 기준 tree snapshot.
            change_plan: _build_change_plan_from_old_tree에서 생성한 변경 계획.

        Returns:
            TreeStore: 구조 변경이 반영된 새 tree.
        """
        new_tree = self._clone_tree(old_tree)
    
        tree_changes = change_plan["tree_changes"]
    
        # -----------------------------------------------------
        # 1) parent의 children_ids 목록 갱신
        # -----------------------------------------------------
        # parent의 direct child 구성이 바뀐 경우 현재 Confluence 기준 children_ids로 교체한다.
        for item in tree_changes["children_updated"]:
            page_id = item["parent_id"]
            node = new_tree.get(page_id)
            if not node:
                continue
    
            node.children_ids = list(item["new_children_ids"])
    
        # -----------------------------------------------------
        # 2) 삭제된 subtree 제거
        # -----------------------------------------------------
        # 삭제 후보 root 아래의 모든 descendant를 tree에서 제거한다.
        for item in tree_changes["subtrees_deleted"]:
            self._apply_delete_subtree(new_tree, item["root_id"])
    
        # -----------------------------------------------------
        # 3) move된 subtree 반영
        # -----------------------------------------------------
        # page 자체와 descendant의 ancestor 경로를 새 parent 기준으로 갱신한다.
        for item in tree_changes["subtrees_moved"]:
            self._apply_pure_move_subtree(
                new_tree=new_tree,
                page_id=item["page_id"],
                new_parent_id=item["new_parent_id"],
                new_ancestor_ids=item["new_ancestor_ids"],
                new_ancestor_titles=item["new_ancestor_titles"],
            )
    
        new_tree.build_links()
        new_tree.mark_bottom_parents()
        return new_tree

    
    def _apply_delete_subtree(self, tree: TreeStore, root_id: str):
        """
        root_id를 기준으로 subtree 전체를 tree에서 제거한다.

        삭제 처리 시 주의할 점:
            1. root node만 제거하면 descendant가 orphan으로 남을 수 있다.
            2. parent.children_ids에 root_id가 남으면 존재하지 않는 child를 참조하게 된다.

        따라서 parent 연결을 먼저 끊고, root를 포함한 subtree 전체 node를 삭제한다.
        """
        
        root_id = str(root_id)
        root = tree.get(root_id)
        if not root:
            return

        subtree_ids = set(map(str, tree.get_subtree_ids(root_id)))

        if root.parent_id:
            parent = tree.get(root.parent_id)
            if parent:
                parent.children_ids = [cid for cid in parent.children_ids if cid != root_id]

        for page_id in subtree_ids:
            if page_id in tree.nodes_by_id:
                del tree.nodes_by_id[page_id]

    
    def _apply_pure_move_subtree(
        self,
        new_tree: TreeStore,
        page_id: str,
        new_parent_id: str | None,
        new_ancestor_ids: list[str],
        new_ancestor_titles: list[str],
    ):
        """
        subtree 이동(move)을 tree에 반영한다.

        변경 대상:
            - 이동한 node의 parent_id
            - 이전 parent의 children_ids
            - 새 parent의 children_ids
            - 이동한 node의 ancestor_ids / ancestor_titles / depth / page_tree
            - descendant들의 ancestor_ids / ancestor_titles / depth / page_tree

        중요한 점:
            subtree 내부의 parent-child 구조는 그대로 유지된다.
            바뀌는 것은 subtree가 외부 tree에서 연결되는 위치와 그에 따른 ancestor 경로다.
        """
        
        node = new_tree.get(page_id)
        if not node:
            return

        old_parent_id = node.parent_id
        old_prefix_ids = list(node.ancestor_ids) + [node.page_id]
        old_prefix_titles = list(node.ancestor_titles) + [node.title]
        # 기존 parent의 children_ids에서 이동 node를 제거한다.
        if old_parent_id:
            old_parent = new_tree.get(old_parent_id)
            if old_parent:
                old_parent.children_ids = [cid for cid in old_parent.children_ids if cid != page_id]
        # 새 parent의 children_ids에 이동 node를 추가한다.
        if new_parent_id:
            new_parent = new_tree.get(new_parent_id)
            if new_parent and page_id not in new_parent.children_ids:
                new_parent.children_ids.append(page_id)
        # 이동 node 자체의 경로 정보를 새 parent 기준으로 갱신한다.
        node.parent_id = new_parent_id
        node.ancestor_ids = list(new_ancestor_ids)
        node.ancestor_titles = list(new_ancestor_titles)
        node.depth = len(node.ancestor_ids) + 1
        node.page_tree = self._make_page_tree(node.ancestor_titles, node.title)
        # descendant들은 기존 ancestor prefix를 새 위치 prefix로 치환해야 한다.
        # 예: A/B/X/Y가 C/D/X/Y로 이동하면 X 하위의 Y도 ancestor 경로가 함께 바뀐다.
        subtree_ids = list(map(str, new_tree.get_subtree_ids(page_id)))
        for desc_id in subtree_ids:
            if desc_id == page_id:
                continue

            desc = new_tree.get(desc_id)
            if not desc:
                continue

            if desc.ancestor_ids[:len(old_prefix_ids)] == old_prefix_ids:
                suffix_ids = desc.ancestor_ids[len(old_prefix_ids):]
                suffix_titles = desc.ancestor_titles[len(old_prefix_titles):]

                desc.ancestor_ids = list(new_ancestor_ids) + [page_id] + suffix_ids
                desc.ancestor_titles = list(new_ancestor_titles) + [node.title] + suffix_titles
                desc.depth = len(desc.ancestor_ids) + 1
                desc.page_tree = self._make_page_tree(desc.ancestor_titles, desc.title)

    
    # =========================================================
    # partial merge
    # =========================================================
    def _collect_subtree_ids(self, tree: TreeStore, root_id: str) -> set[str]:
        """
        특정 root_id 아래 subtree의 모든 page_id를 수집한다.
        root가 tree에 없으면 빈 set을 반환한다.
        """
        root = tree.get(root_id)
        if not root:
            return set()
        return set(map(str, tree.get_subtree_ids(root_id)))

    
    def _merge_updated_subtrees(
        self,
        old_tree: TreeStore,
        partial_tree: TreeStore,
        changed_root_ids: list[str],
    ) -> TreeStore:
        """
        재빌드된 partial_tree를 기존 tree에 병합한다.

        처리 방식:
            1. changed_root_ids에 해당하는 기존 subtree id를 old_tree에서 수집한다.
            2. 해당 subtree node는 기존 tree에서 제외한다.
            3. 나머지 기존 node는 그대로 유지한다.
            4. 새로 BFS 조회한 partial_tree node를 추가한다.
            5. parent-child link와 bottom parent 정보를 다시 계산한다.

        이 방식은 subtree 내부 구조가 크게 바뀐 경우에도 해당 subtree만 안전하게 교체할 수 있다.
        """
        
        new_tree = TreeStore()

        remove_ids = set()
        for root_id in changed_root_ids:
            remove_ids |= self._collect_subtree_ids(old_tree, root_id)

        for node in old_tree.iter_nodes():
            if str(node.page_id) in remove_ids:
                continue
            new_tree.add_node(self._clone_node(node))

        for node in partial_tree.iter_nodes():
            new_tree.add_node(self._clone_node(node))

        new_tree.build_links()
        new_tree.mark_bottom_parents()
        return new_tree

    
    # =========================================================
    # compare
    # =========================================================
    def _compare_trees(self, old_tree: TreeStore, new_tree: TreeStore) -> dict:
        """
        old_tree와 new_tree를 비교하여 page_id별 변경 이벤트를 생성한다.

        비교 항목:
            - old에 없고 new에 있으면 created
            - old에 있고 new에 없으면 deleted
            - last_modified_date가 다르면 updated
            - parent_id가 다르면 moved
            - children_ids가 다르면 children_changed
            - ancestor_ids가 다르면 ancestor_path_changed
            - ancestor_titles가 다르면 ancestor_title_changed
            - labels가 다르면 labels_changed

        direct_events / change_flags / impact_events 구분:
            - direct_events: 해당 page 자체에 직접 발생한 이벤트
            - change_flags: metadata 변화처럼 downstream에서 별도 처리할 수 있는 flag
            - impact_events: ancestor 변경처럼 하위 문서 검색 경로에 영향을 주는 이벤트

        Returns:
            dict: changed_urls 생성 단계에서 사용하는 page_id별 변경 결과.
        """
        results = {}

        old_ids = set(map(str, old_tree.nodes_by_id.keys()))
        new_ids = set(map(str, new_tree.nodes_by_id.keys()))
        # 새 tree에만 존재하는 page는 생성된 page다.

        for page_id in new_ids - old_ids:
            node = new_tree.get(page_id)
            results[page_id] = {
                "direct_events": ["created"],
                "change_flags": [],
                "impact_events": [],
                "new_parent_id": node.parent_id if node else None,
                "old_parent_id": None,
                "old_last_modified_date": None,
                "new_last_modified_date": node.last_modified_date if node else None,
            }
        # old tree에만 존재하는 page는 삭제된 page다.

        for page_id in old_ids - new_ids:
            node = old_tree.get(page_id)
            results[page_id] = {
                "direct_events": ["deleted"],
                "change_flags": [],
                "impact_events": [],
                "old_parent_id": node.parent_id if node else None,
                "new_parent_id": None,
                "old_last_modified_date": node.last_modified_date if node else None,
                "new_last_modified_date": None,
            }
        # 양쪽 tree에 모두 있는 page는 field 단위로 변경 여부를 비교한다.

        for page_id in old_ids & new_ids:
            old_node = old_tree.get(page_id)
            new_node = new_tree.get(page_id)
            if not old_node or not new_node:
                continue

            direct_events = []
            change_flags = []
            impact_events = []

            if old_node.last_modified_date != new_node.last_modified_date:
                direct_events.append("updated")

            if old_node.parent_id != new_node.parent_id:
                direct_events.append("moved")

            if set(map(str, old_node.children_ids)) != set(map(str, new_node.children_ids)):
                direct_events.append("children_changed")

            if old_node.ancestor_ids != new_node.ancestor_ids:
                impact_events.append("ancestor_path_changed")
                
            if old_node.ancestor_titles != new_node.ancestor_titles:
                impact_events.append("ancestor_title_changed")

            if list(old_node.labels) != list(new_node.labels):
                change_flags.append("labels_changed")

            
            if direct_events or change_flags or impact_events:
                results[page_id] = {
                    "direct_events": direct_events,
                    "change_flags": change_flags,
                    "impact_events": impact_events,
                    "old_parent_id": old_node.parent_id,
                    "new_parent_id": new_node.parent_id,
                    "old_last_modified_date": old_node.last_modified_date,
                    "new_last_modified_date": new_node.last_modified_date,
                }

        return results

    
    def _is_old_leaf(self, node: TreeNode) -> bool:
        """
        기존 tree 기준 leaf node인지 판단한다.
        children_ids가 없으면 leaf로 본다.
        """
        return not bool(node.children_ids)
   
    
    def _get_children_nodes(self, tree: TreeStore, node: TreeNode) -> list[TreeNode]:
        """
        node.children_ids를 실제 TreeNode 목록으로 변환한다.
        tree에 존재하지 않는 child id는 무시한다.
        """
        children = []
        for cid in node.children_ids:
            child = tree.get(str(cid))
            if child:
                children.append(child)
        return children
    
    
    def _is_anchor_parent(self, tree: TreeStore, node: TreeNode) -> bool:
        """
        leaf child와 non-leaf child를 함께 가진 parent인지 판단한다.

        현재 코드에서는 bottom parent 기반 probe를 사용하고 있어 직접 호출되지 않을 수 있지만,
        anchor 성격의 parent를 정의하는 보조 함수로 유지한다.
        """
        children = self._get_children_nodes(tree, node)
        if not children:
            return False
    
        leaf_flags = [self._is_old_leaf(child) for child in children]
        return any(leaf_flags) and any(not x for x in leaf_flags)

    
    def _is_bottom_parent_node(self, node: TreeNode) -> bool:
        """
        TreeStore.mark_bottom_parents()에서 표시한 bottom parent 여부를 반환한다.

        bottom parent는 hidden 구조 변화 탐지를 위한 ancestor probe 시작점으로 사용된다.
        """
        return bool(getattr(node, "is_bottom_parent", False))

    
    def _get_old_parent_ids(self, old_tree: TreeStore) -> list[str]:
        """
        old_tree에서 children_ids를 가진 parent page_id 목록을 반환한다.

        direct child 비교는 기존에 parent였던 node를 기준으로 수행한다.
        """
        return sorted(
            str(node.page_id)
            for node in old_tree.iter_nodes()
            if node.children_ids
        )
    
    
    @RetryOnException()
    def _get_child_rows_by_parent_cql(self, parent_id: str):
        """
        CQL parent 조건으로 direct child row를 조회한다.

        child row의 ancestors와 metadata.labels를 함께 가져와서
        full detail fetch 없이도 parent, label, 일부 metadata 변경 여부를 판단하는 데 사용한다.
        """
        try:
            cql = f'parent={parent_id}'
            result = self.confluence_api.cql(
                cql,
                # expand="content.ancestors",
                expand="content.ancestors,content.metadata.labels",
                limit=2000,
            )
            return result.get("results", [])
        except Exception as e:
            logger.error(
                "[branch discovery] Failed to get child rows for parent %s: %s",
                parent_id,
                e
            )
            raise
    
    
    @RetryOnException()
    def _get_child_rows_by_parent_generator(self, parent_id: str) -> list[dict]:
        """
        child 수가 많은 parent에 대해 generator 방식으로 direct child row를 조회한다.

        CQL 결과가 많을 때 응답 크기나 pagination 안정성 문제가 생길 수 있으므로,
        GENERATOR_THRESHOLD 이상인 경우 fallback 경로로 사용한다.
        """
        rows = []
        try:
            pages_generator = self.confluence_api.get_page_child_by_type(
                parent_id,
                # expand="ancestors,version"
                expand="ancestors,version,metadata.labels",

            )
    
            for page in pages_generator:
                rows.append(page)
    
        except Exception as e:
            logger.error(
                "[branch discovery] Failed generator fetch for parent %s: %s",
                parent_id,
                e
            )
            raise
    
        return rows   
    
    
    def _normalize_child_row(self, row: dict) -> dict:
        """
        CQL row와 generator row를 동일한 schema로 정규화한다.

        Confluence API 응답은 조회 방식에 따라 구조가 다르다.
        CQL 응답은 row["content"] 아래에 page 정보가 들어가고,
        generator 응답은 page 정보가 row 최상위에 들어간다.

        이 함수는 이후 비교 로직이 응답 형태에 의존하지 않도록
        page_id, title, ancestor, parent, modified date, URL, labels를 공통 구조로 맞춘다.
        반환:
        {
            "page_id": str,
            "title": str,
            "ancestor_ids": list[str],
            "ancestor_titles": list[str],
            "parent_id": str | None,
            "last_modified_date": str | None,
            "page_url": str | None,
            "page_url_tiny": str | None,
        }
        """
        if "content" in row:
            content = row.get("content", {}) or {}
            page_id = str(content.get("id") or "")
            title = content.get("title", page_id)
    
            ancestors = content.get("ancestors", []) or []
            ancestor_ids = [str(x.get("id")) for x in ancestors if x.get("id")]
            ancestor_titles = [str(x.get("title", x.get("id"))) for x in ancestors if x.get("id")]
            parent_id = ancestor_ids[-1] if ancestor_ids else None
    
            page_links = content.get("_links", {}) or {}
            webui = page_links.get("webui")
            tinyui = page_links.get("tinyui")
            base = page_links.get("base") or self.page_root.rstrip("/")
    
            return {
                "page_id": page_id,
                "title": title,
                "ancestor_ids": ancestor_ids,
                "ancestor_titles": ancestor_titles,
                "parent_id": parent_id,
                "last_modified_date": self._normalize_datetime(row.get("lastModified")),
                "page_url": f"{base}{webui}" if webui else None,
                "page_url_tiny": f"{base}{tinyui}" if tinyui else None,
                "labels": self._extract_labels_from_child_row(row),
            }
    
        page_id = str(row.get("id") or "")
        title = row.get("title", page_id)
    
        ancestors = row.get("ancestors", []) or []
        ancestor_ids = [str(x.get("id")) for x in ancestors if x.get("id")]
        ancestor_titles = [str(x.get("title", x.get("id"))) for x in ancestors if x.get("id")]
        parent_id = ancestor_ids[-1] if ancestor_ids else None
    
        version = row.get("version", {}) or {}
        page_links = row.get("_links", {}) or {}
        webui = page_links.get("webui")
        tinyui = page_links.get("tinyui")
        base = page_links.get("base") or self.page_root.rstrip("/")
    
        return {
            "page_id": page_id,
            "title": title,
            "ancestor_ids": ancestor_ids,
            "ancestor_titles": ancestor_titles,
            "parent_id": parent_id,
            "last_modified_date": self._normalize_datetime(version.get("when")),
            "page_url": f"{base}{webui}" if webui else None,
            "page_url_tiny": f"{base}{tinyui}" if tinyui else None,
            "labels": self._extract_labels_from_child_row(row),

        }
    
    
    @RetryOnException()
    def _get_descendant_rows_by_ancestor_cql(self, ancestor_id: str, page_limit: int = 200) -> list[dict]:
        """
        특정 ancestor 아래의 모든 descendant row를 CQL로 조회한다.

        사용 목적:
            old tree에서 leaf였던 page 아래에 새 child가 생긴 경우,
            기존 parent 목록만 비교해서는 그 변화를 놓칠 수 있다.
            이를 보완하기 위해 bottom parent를 기준으로 descendant를 조회하고,
            old leaf가 현재 ancestor chain에 등장하는지 확인한다.
        """
        ancestor_id = str(ancestor_id)
        rows = []
        start = 0
    
        while True:
            cql = f'ancestor={ancestor_id}'
            result = self.confluence_api.cql(
                cql,
                expand="content.ancestors",
                limit=page_limit,
                start=start,
            )
            
            batch = result.get("results", []) or []
            if not batch:
                break
    
            rows.extend(batch)
    
            if len(batch) < page_limit:
                break   
            start += page_limit
 
        return rows

    
    def _build_change_plan_from_old_tree(self, old_tree: TreeStore) -> dict:
        """
        old_tree를 기준으로 현재 Confluence 상태를 비교해 change_plan을 만든다.

        change_plan은 실제 tree를 바로 수정하지 않고, 수정에 필요한 계획만 담는다.
        이후 _apply_change_plan_to_tree, _apply_detail_nodes, _merge_updated_subtrees에서 이 계획을 반영한다.

        핵심 판단 기준:
            1. parent별 direct child 목록 비교
                - old child 목록과 현재 child 목록이 다르면 children_updated로 기록한다.
                - old에는 없고 현재에 있으면 added child로 본다.
                - old에는 있고 현재에 없으면 removed child로 본다.

            2. deleted와 moved 구분
                - 어떤 page_id가 removed에도 있고 added에도 있으면 실제 삭제/생성이 아니라 move일 가능성이 높다.
                - 이 경우 fetch_page_meta로 현재 parent를 확인한 뒤 subtrees_moved로 전환한다.

            3. metadata refresh 판단
                - last_modified_date 또는 label이 다르면 page detail refresh 대상에 넣는다.
                - added/moved page도 최신 metadata가 필요하므로 refresh 대상에 넣는다.

            4. hidden 구조 변화 감지
                - old tree에서 leaf였던 page가 현재 ancestor chain에 등장하면
                  그 아래 새 subtree가 생긴 것이므로 subtree rebuild 대상으로 넣는다.

            5. 삭제 subtree 보정
                - 실제 삭제될 subtree 내부 page는 refresh/rebuild할 필요가 없으므로 대상에서 제거한다.

        Returns:
            dict:
                tree_changes:
                    children_updated: parent의 children_ids 변경 목록
                    subtrees_deleted: 삭제된 subtree root 목록
                    subtrees_moved: 이동된 subtree root 및 새 ancestor 정보

                refresh_targets:
                    pages_to_refresh: detail fetch가 필요한 page 목록
                    subtree_roots_to_rebuild: subtree rebuild가 필요한 root 목록
        """
    
        # -----------------------------------------------------
        # tree_changes:
        # 기존 tree에 직접 반영 가능한 구조 변경
        # -----------------------------------------------------
        children_updated = []
        subtrees_deleted = []
        subtrees_moved = []
    
        # -----------------------------------------------------
        # refresh_targets:
        # 별도 상세 조회 또는 subtree 재탐색이 필요한 대상
        # -----------------------------------------------------
        pages_to_refresh = set()
        subtree_roots_to_rebuild = set()
    
        # added/removed를 전체 단위로 모아두면
        # "삭제처럼 보였지만 사실 move였던 page"를 나중에 판별할 수 있다.
        added_ids_global = set()
        removed_ids_global = set()
    
        # 실제로 삭제된 subtree의 root 후보
        deleted_root_ids = set()
    
        # -----------------------------------------------------
        # 1) old tree에서 parent였던 노드들의 direct child 목록 비교
        # -----------------------------------------------------
        parent_ids = self._get_old_parent_ids(old_tree)
    
        for parent_id in parent_ids:
            parent_node = old_tree.get(parent_id)
            if not parent_node:
                continue
    
            rows = self._get_current_children_info(parent_id)
    
            current_child_ids = set()
    
            for row in rows:
                info = self._normalize_child_row(row)
                child_id = info["page_id"]
                if not child_id:
                    continue
    
                current_child_ids.add(child_id)
    
                # direct child row만으로도 수정 시각 차이를 확인할 수 있으면
                # page detail refresh 대상으로 등록한다.
                old_child = old_tree.get(child_id)
                if old_child:
                    if old_child.last_modified_date != info["last_modified_date"]:
                        pages_to_refresh.add(str(child_id))
                        
                    if list(old_child.labels) != list(info.get("labels", [])):
                        pages_to_refresh.add(str(child_id))
                                
            old_child_ids = set(map(str, parent_node.children_ids))
    
            # child 구성 변화가 없으면 skip
            if old_child_ids == current_child_ids:
                continue
    
            added_child_ids = sorted(current_child_ids - old_child_ids)
            removed_child_ids = sorted(old_child_ids - current_child_ids)
    
            logger.info(
                "[CHILDREN CHANGED] parent=%s | title=%s | added=%s | removed=%s",
                parent_id,
                parent_node.title,
                added_child_ids[:20],
                removed_child_ids[:20],
            )
    
            # parent의 child_ids 목록 자체가 바뀌었으므로 tree 변경 계획에 기록
            children_updated.append({
                "parent_id": str(parent_id),
                "new_children_ids": sorted(current_child_ids),
            })
    
            # removed child는 우선 삭제 후보로 잡아둔다.
            # 이후 added ∩ removed 를 이용해 move인지 재판별한다.
            for removed_id in removed_child_ids:
                removed_ids_global.add(str(removed_id))
                if old_tree.get(removed_id):
                    deleted_root_ids.add(str(removed_id))
                    subtrees_deleted.append({
                        "root_id": str(removed_id),
                    })
    
            # # added child는 새로 들어온 노드이므로 상세 조회 대상
            # for added_id in added_child_ids:
            #     added_ids_global.add(str(added_id))
            #     pages_to_refresh.add(str(added_id))

            # added child는 새로 들어온 노드이므로 상세 조회 대상
            # 단, added child 아래에 하위 page가 있을 수 있으므로 subtree rebuild 대상으로도 등록한다.
            # added child는 새로 들어온 노드이므로 상세 조회 대상
            # 단, added child 아래에 하위 page가 있을 수 있으므로 subtree rebuild 대상으로도 등록한다.
            for added_id in added_child_ids:
                added_id = str(added_id)
            
                logger.info(
                    "SUBTREE_REBUILD_BY_ADDED_CHILD | parent_id=%s | added_child_id=%s",
                    parent_id,
                    added_id,
                )
            
                added_ids_global.add(added_id)
                pages_to_refresh.add(added_id)
                subtree_roots_to_rebuild.add(added_id)
                    
        # -----------------------------------------------------
        # 2) move 판별
        # added와 removed에 동시에 잡힌 page는
        # "삭제 + 생성"이 아니라 부모 변경(move)일 가능성이 높다.
        # -----------------------------------------------------
        move_candidate_ids = added_ids_global & removed_ids_global
    
        for page_id in sorted(move_candidate_ids):
            old_node = old_tree.get(page_id)
            if not old_node:
                continue
    
            try:
                new_meta = self.fetch_page_meta(page_id)
            except Exception:
                continue
    
            # parent가 같으면 진짜 move가 아님
            if str(old_node.parent_id) == str(new_meta["parent_id"]):
                continue
    
            # delete 후보에서 제거
            subtrees_deleted = [
                item for item in subtrees_deleted
                if item["root_id"] != str(page_id)
            ]
            deleted_root_ids.discard(str(page_id))
    
            # move 계획으로 전환
            subtrees_moved.append({
                "page_id": str(page_id),
                "new_parent_id": str(new_meta["parent_id"]) if new_meta["parent_id"] else None,
                "new_ancestor_ids": list(new_meta["ancestor_ids"]),
                "new_ancestor_titles": list(new_meta["ancestor_titles"]),
            })
    
            # moved page의 메타는 다시 최신화
            pages_to_refresh.add(str(page_id))
    
        # -----------------------------------------------------
        # 3) anchor parent 기준 ancestor probe
        # old leaf였던 노드 아래에 새 자식이 생긴 경우를 잡기 위함
        # -----------------------------------------------------
        for node in old_tree.iter_nodes():
            if not self._is_bottom_parent_node(node):
                continue
            
            try:
                rows = self._get_descendant_rows_by_ancestor_cql(node.page_id)
            except Exception as e:
                logger.error(
                    "[branch discovery] anchor probe failed | page_id=%s | error=%s",
                    node.page_id,
                    e,
                )
                continue
    
            for row in rows:
                info = self._normalize_child_row(row)
    
                page_id = info["page_id"]
                if page_id:
                    old_same_page = old_tree.get(page_id)
    
                    # descendant row에서도 수정 시각 차이를 확인하면 refresh 대상으로 등록
                    if old_same_page and old_same_page.last_modified_date != info["last_modified_date"]:
                        pages_to_refresh.add(str(page_id))

                # ancestor chain 중 old tree에서는 leaf였던 노드가
                # 현재는 중간 parent가 되었을 수 있다.
                # 이런 경우 subtree 단위 재탐색이 필요하다.
                for ancestor_id in info["ancestor_ids"]:
                    old_ancestor = old_tree.get(str(ancestor_id))
                    if not old_ancestor:
                        continue
    
                    # 이미 old tree에서 parent였던 노드는 rebuild 대상이 아님
                    if old_ancestor.children_ids:
                        continue
    
                    subtree_roots_to_rebuild.add(str(ancestor_id))
    
        # -----------------------------------------------------
        # 4) bootstrap root 자체 수정 여부 확인
        # root page의 메타가 바뀌었으면 page refresh 대상에 포함
        # -----------------------------------------------------
        for root_id in sorted(self.allowed_root_ids):
            old_root = old_tree.get(str(root_id))

            try:
                new_meta = self.fetch_page_meta(root_id)
            except Exception:
                continue

            # old tree에 없는 root = config에 새로 추가된 root.
            # 실존 확인(fetch 성공)된 경우에만 subtree rebuild로 편입한다.
            if not old_root:
                subtree_roots_to_rebuild.add(str(root_id))
                continue

            if old_root.last_modified_date != new_meta["last_modified_date"]:
                pages_to_refresh.add(str(root_id))
    
        # -----------------------------------------------------
        # 5) 실제 삭제 subtree 내부는 refresh/rebuild 대상에서 제외
        # 이미 tree에서 사라질 예정인 page를 다시 조회할 필요는 없다.
        # -----------------------------------------------------
        deleted_all_ids = set()
        for deleted_root_id in deleted_root_ids:
            deleted_all_ids |= set(map(str, old_tree.get_subtree_ids(deleted_root_id)))
    
        pages_to_refresh -= deleted_all_ids
        subtree_roots_to_rebuild -= deleted_all_ids

        for refresh_id in list(pages_to_refresh):
            old_node = old_tree.get(str(refresh_id))
            if not old_node:
                continue
        
            if old_node.children_ids:
                continue
        
            try:
                current_children = self.fetch_children_ids(refresh_id)
            except Exception as e:
                logger.error(
                    "[LEAF CHILD CHECK FAILED] page_id=%s | title=%s | error=%s",
                    refresh_id,
                    old_node.title,
                    e,
                )
                continue
        
            if current_children:
                logger.info(
                    "[LEAF BECAME PARENT] page_id=%s | title=%s | children=%s",
                    refresh_id,
                    old_node.title,
                    current_children,
                )
                subtree_roots_to_rebuild.add(str(refresh_id))

        
        # rebuild root는 최소 단위만 남긴다.
        # 예: A와 A/B가 동시에 rebuild 후보면 A만 남기면 충분하다.
        subtree_roots_to_rebuild = self._minimize_seed_roots(subtree_roots_to_rebuild, old_tree)
    
        logger.info(
            "[CHANGE PLAN SUMMARY] children_updated=%s | subtrees_deleted=%s | subtrees_moved=%s | pages_to_refresh=%s | subtree_roots_to_rebuild=%s | move_candidates=%s",
            len(children_updated),
            len(subtrees_deleted),
            len(subtrees_moved),
            len(pages_to_refresh),
            len(subtree_roots_to_rebuild),
            len(move_candidate_ids),
        )
    
        return {
            "tree_changes": {
                "children_updated": children_updated,
                "subtrees_deleted": subtrees_deleted,
                "subtrees_moved": subtrees_moved,
            },
            "refresh_targets": {
                "pages_to_refresh": sorted(pages_to_refresh),
                "subtree_roots_to_rebuild": sorted(subtree_roots_to_rebuild),
            },
        }
    
    
    def _get_current_children_info(self, parent_id: str) -> list[dict]:
        """
        parent의 현재 direct child row를 조회한다.

        기본적으로 CQL 조회를 사용한다.
        child 수가 많으면 generator 방식으로 다시 조회하여 대량 child parent를 안정적으로 처리한다.
        """

        rows = self._get_child_rows_by_parent_cql(parent_id)
    
        if len(rows) < self.GENERATOR_THRESHOLD:
            return rows
    
        logger.info(
            "[branch discovery] parent=%s has many children (%s). using generator mode",
            parent_id,
            len(rows),
        )
        return self._get_child_rows_by_parent_generator(parent_id)

    def _propagate_title_change_to_descendants(
        self,
        tree: TreeStore,
        page_id: str,
        old_title: str,
        new_title: str,
    ):
        """
        page_id 노드의 title 변경을 descendant의 경로 정보에 전파한다.

        page_tree와 ancestor_titles는 상위 page title에 의존한다.
        따라서 특정 page의 title이 변경되면 subtree 전체 descendant의
        ancestor_titles와 page_tree를 함께 갱신해야 한다.

        변경 대상:
            - descendant.ancestor_titles
            - descendant.page_tree

        변경하지 않는 대상:
            - parent_id
            - ancestor_ids
            - children_ids
        """
        page_id = str(page_id)
        root = tree.get(page_id)
        if not root:
            return
    
        subtree_ids = list(map(str, tree.get_subtree_ids(page_id)))
    
        for desc_id in subtree_ids:
            if desc_id == page_id:
                continue
    
            desc = tree.get(desc_id)
            if not desc:
                continue
    
            # ancestor_ids 안에서 page_id 위치를 찾아,
            # 같은 위치의 ancestor_titles만 new_title로 교체
            if page_id not in desc.ancestor_ids:
                continue
    
            idx = desc.ancestor_ids.index(page_id)
    
            if idx >= len(desc.ancestor_titles):
                continue
    
            # 안전하게 해당 조상 타이틀만 교체
            if desc.ancestor_titles[idx] == old_title:
                desc.ancestor_titles[idx] = new_title
            else:
                # 혹시 old_title이 달라도 page_id 위치 기준으로 덮어씀
                desc.ancestor_titles[idx] = new_title
    
            desc.page_tree = self._make_page_tree(desc.ancestor_titles, desc.title)

    
    def _normalize_datetime(self, iso_date: str | None) -> str | None:
        """
        Confluence ISO datetime 문자열을 비교 가능한 표준 문자열로 변환한다.

        Confluence API는 'Z' timezone 표현을 포함할 수 있다.
        이를 Python datetime에서 처리 가능한 '+00:00' 형태로 바꾼 뒤
        '%Y-%m-%d %H:%M:%S' 형식으로 통일한다.

        변환 실패 시:
            discovery 전체를 중단하지 않기 위해 warning만 남기고 원본 값을 반환한다.
        """
        if not iso_date:
            return None
    
        try:
            normalized = iso_date.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.warning("[datetime normalize failed] %s | %s", iso_date, e)
            return iso_date

    
    def _extract_labels_from_page(self, page: dict) -> list[str]:
        """
        page detail API 응답에서 label 목록을 추출한다.

        label은 downstream DB schema와 비교 로직에서 list[str]로 사용된다.
        빈 값과 중복을 제거하고 정렬하여 비교 가능한 형태로 반환한다.
        """
        results = (
            page.get("metadata", {})
                .get("labels", {})
                .get("results", [])
            or []
        )

        labels = []
        for item in results:
            name = item.get("name")
            if name is None:
                continue

            value = str(name).strip()
            if not value:
                continue

            labels.append(value)

        return sorted(set(labels))

    def _extract_labels_from_child_row(self, row: dict) -> list[str]:
        """
        CQL row 또는 generator row에서 label 목록을 추출한다.

        조회 방식에 따라 label 위치가 다르다.
            - CQL row: row["content"]["metadata"]["labels"]
            - generator row: row["metadata"]["labels"]

        반환값은 downstream 비교를 위해 정렬된 list[str]로 통일한다.
        """
        if "content" in row:
            label_results = (
                row.get("content", {})
                   .get("metadata", {})
                   .get("labels", {})
                   .get("results", [])
                or []
            )
        else:
            label_results = (
                row.get("metadata", {})
                   .get("labels", {})
                   .get("results", [])
                or []
            )
    
        labels = []
        for item in label_results:
            name = item.get("name")
            if name is None:
                continue
    
            value = str(name).strip()
            if value:
                labels.append(value)
    
        return sorted(set(labels))

if __name__ == "__main__":
    main()