import os
import time
import logging
from dataclasses import replace
from typing import Dict

from src.utils.utils import RetryOnException, manage_version_files
from src.components.subsidiary.subsidiary_io import load_subsidiaries, save_subsidiaries
from src.components.subsidiary.subsidiary_models import SubsidiaryStore, SubsidiaryState
from src.utils.confluence_patch import initialize_confluence_api, patch_confluence_request

logger = logging.getLogger(__name__)
patch_confluence_request()

class SubsidiaryRegistry:
    CQL_STRUCT_EXPAND = "content.ancestors"
    
    def __init__ (
        self,
        page_root: str,
        confluence_token: str,
        subsidiary_root_id: str,
        subsidiary_path: str | None = None,
        subsidiary_filename_prefix: str= 'subsidiary_master',
        cql_page_limit: int = 200,
    ):
        """
        Args:
            page_root: Confluence base URL.
            confluence_token: Confluence API token.
            subsidiary_root_id: GPPM(Subsidiary) 탐색 시작점.
            subsidiary_path: subsidiary state snapshot이 저장될 디렉터리
            filename_prefix: 저장되는 snapshot 파일 prefix
            cql_page_limit: 구조 조회 1회 요청 건수 (서버 한도 200)
        """
        self.page_root = str(page_root)
        self.confluence_token = confluence_token
        self.subsidiary_root_id = str(subsidiary_root_id)
        
        if not subsidiary_path :
            raise ValueError("subsidiary_path is required")
        self.subsidiary_dir = subsidiary_path
        self.subsidiary_filename_prefix = subsidiary_filename_prefix
        
        self.cql_page_limit = cql_page_limit
        
        # 최신 subsidiary는 subsidiary_dir에 저장하고, 이전 버전은 v_old 하위에 보관한다.
        os.makedirs(self.subsidiary_dir, exist_ok=True)
        os.makedirs(os.path.join(self.subsidiary_dir, "v_old"), exist_ok=True)
        
        self.confluence_api = initialize_confluence_api(self.page_root, self.confluence_token)
    
    # =========================================================
    # public 
    # =========================================================

    def sync(self, subsidiary_list: list[str]) -> dict :
        """
        수집 -> 로드 -> 비교 -> 저장 -> 검증 -> bootstrap_roots 반환
        """
        fetched_page_ids = self._fetch_page_ids()
        
        old_store = load_subsidiaries(self.subsidiary_dir, self.subsidiary_filename_prefix)
        has_old_store = len(old_store) > 0
        
        if has_old_store :
            results = self._compare_subsidiaries(old_store, fetched_page_ids)
            scan_mode = "delta_subsidiary_refresh"
        else :
            results = self._build_initial_store(fetched_page_ids)
            scan_mode = "initial_subsidiary_build"
            
            
        manage_version_files(self.subsidiary_dir, self.subsidiary_filename_prefix)  # v_new → v_old
        save_subsidiaries(
            self.subsidiary_dir, 
            results,
            metadata={
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "root_source": scan_mode
            },
            filename_prefix=self.subsidiary_filename_prefix,
        )
        
        local_bootstrap_roots = self._validate_inputs(results, subsidiary_list)
        print("[SUBSIDIARY REGISTER] local_bootstrap_roots=%s", local_bootstrap_roots)
        return local_bootstrap_roots
    
    # =========================================================
    # fetch from CQL 
    # =========================================================
    @RetryOnException()
    def _cql_all(self, cql: str, expand: str | None, limit: int) -> list[dict]:
        """
        CQL 결과를 끝까지 이어 받는다.

        요청한 limit가 아니라 서버가 실제로 돌려준 건수(len(batch))만큼 start를 늘리고,
        totalSize에 도달하거나 빈 batch가 오면 멈춘다.
        (서버가 limit을 200/50으로 줄여도 누락 없이 끝까지 받기 위함)
        """
        rows: list[dict] = []
        start = 0
        total = None

        while True:
            res = self.confluence_api.cql(cql, start=start, limit=limit, expand=expand)
            batch = res.get("results", []) or []
            if total is None :
                total = res.get("totalSize")
            if not batch:
                break

            rows.extend(batch)
            start += len(batch)

            if total is not None and start >= total:
                break

        if total is not None and len(rows) < total:
            print("[SUBSIDIARY REGISTER] %s | rows(%s) < totalSize(%s)", cql, len(rows), total) # logger.warning
        return rows
    
    def _fetch_page_ids(self) -> dict[str,str] :
        rows = self._cql_all(
            f"type=page AND parent={self.subsidiary_root_id}",
            self.CQL_STRUCT_EXPAND,
            self.cql_page_limit
        )
        
        fetched_page_ids = {}
        for row in rows :
            content = row.get("content") or {}
            title = content.get('title')
            page_id = content.get("id")
            if title and page_id :
                fetched_page_ids[title] = str(page_id)
        print("[SUBSIDIARY REGISTER] fetched=%s", len(fetched_page_ids)) # logger.info
        return fetched_page_ids
    
    # =========================================================
    # compare 
    # =========================================================
    def _build_initial_store(self, fetched_page_ids) -> SubsidiaryStore :
        '''
        기존 snapshot이 없어 새로 full build해야 하는 경우
        
        Args:
            fetched_page_ids: confluence cql로 가져온 {법인명: page_id}
        '''
        results = SubsidiaryStore()
        for subsidiary_name, page_id in fetched_page_ids.items() :
            results.add_subsidiary(subsidiary_name, SubsidiaryState(page_id=str(page_id)))
        return results
    
    def _compare_subsidiaries(
        self,
        old_store: SubsidiaryStore,
        fetched_page_ids: Dict[str, str]
    ) -> SubsidiaryStore:
        """
        법인 최신화 결과(new_store)를 기존 old_store(SubsidiaryStore)와 비교/병합한다. 파일은 건드리지 않는다.
        
        Args:
        new_store: 법인 최신화 결과 {법인명: page_id, ..., 법인명: page_id}.
        old_store: 기존 v_new에서 읽은 SubsidiaryStore.
        
        비교 항목 :
            - old에 없고 new에 있으면 pending
            - old에 있고 new에 없으면 deleted
            - old와 new에 둘 다 있고 page_id가 다르면 pending (재적재 대상)
            - 그 외 : old 값 그대로 유지
        """
        results = SubsidiaryStore()

        for subsidiary_name in sorted(new_store.keys() | old_store.subsidiaries.keys()):  # 합집합
            new_page_id = new_store.get(subsidiary_name)
            old_state = old_store.get(subsidiary_name)

            # 1. old에만 있음 → 삭제된 법인
            if new_page_id is None:
                if old_state.status == "deleted":  # 이미 삭제된 건
                    results.add_subsidiary(subsidiary_name, old_state)
                else:  # 이번에 삭제된 건
                    results.add_subsidiary(subsidiary_name, replace(old_state, status="deleted"))
                    print("[SUBSIDIARY REGISTER] deleted | name=%s, page_id=%s", subsidiary_name, old_state.page_id)
                continue

            # 2. new에만 있음 → 신규 법인
            if old_state is None:
                results.add_subsidiary(subsidiary_name, SubsidiaryState(page_id=str(new_page_id)))
                print("[SUBSIDIARY REGISTER] created | name=%s, page_id=%s", subsidiary_name, new_page_id)
                continue

            # 3. 양쪽에 있고 page_id가 바뀜
            if str(new_page_id) != str(old_state.page_id):
                results.add_subsidiary(subsidiary_name, SubsidiaryState(
                    page_id=str(new_page_id),
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                ))
                continue

            # 4. 변경 없음 → 기존 값 그대로
            results.add_subsidiary(subsidiary_name, old_state)

        logger.info("[SUBSIDIARY REGISTER] done | results=%d | %s", len(results), results.status_counts())
        print("[SUBSIDIARY REGISTER] done | results=%d | %s", len(results), results.status_counts())
        return results
    
    
    # =========================================================
    #  validate
    # =========================================================
    def _validate_inputs(self, new_store, subsidiary_list) :
        """입력한 로컬 법인이 모두 store에 있으며 deleted가 아닌지 확인한 후 {법인명: page_id}를 반환한다."""
        local_bootstrap_roots = {}
        for subsidiary in subsidiary_list :
            state = new_store.get(subsidiary)
            if state is None or state.status == 'deleted' :
                raise ValueError(f"unknown or deleted subsidiary: {subsidiary}")
            local_bootstrap_roots[subsidiary] = state.page_id
        return local_bootstrap_roots