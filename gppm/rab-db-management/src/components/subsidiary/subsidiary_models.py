"""
 ***********************************************************************************************************************
 * Creator     : minjae.cha
 * Create Date : 2026.11.20
 * Description : Confluence GPPM(Subsidiary) 하위 법인 페이지의 동기화 상태 모델
 *               SubsidiaryState(법인 1개 상태)와 SubsidiaryStore(법인명 기준 container)를 정의하고,
 *               법인 json snapshot 저장/로드를 위한 to_dict / from_dict 를 제공한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.11.20   minjae.cha                     최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

def now_str() -> str:
    """
    created_at / updated_at / deleted_at / 파일명에 쓰는 현재 시각 문자열을 반환한다.

    Returns:
        str: YYYYMMDD_HHMMSS 형식의 현재 시각.
    """
    return time.strftime("%Y-%m-%d_%H:%M:%S")


@dataclass
class SubsidiaryState:
    """
    Confluence 내 법인 페이지 동기화 상태 정의.

    필드 구성:
        - page_id:
            법인 ancestor page id. local 법인 tree discovery의 root로 사용된다.
        - status:
            처리 단계. pending(처리 대기) / deleted(Confluence에서 사라짐).
        - created_at / updated_at 
            법인이 처음 등록된 시각 / page_id가 바뀐 시각
        
    """
    page_id: str
    status: str = "pending"
    created_at: str = field(default_factory=now_str)
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        SubsidiaryState를 JSON 저장 가능한 dict로 변환한다.

        Returns:
            Dict[str, Any]:
                법인 json snapshot에 저장될 법인 상태.
        """
        return {
            "page_id": self.page_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubsidiaryState":
        """
        JSON에서 읽은 dict를 SubsidiaryState 객체로 복원한다.
        저장 파일에 없는 키는 기본값으로 채운다.

        Args:
            data:
                SubsidiaryState.to_dict()로 저장된 법인 상태.

        Returns:
            SubsidiaryState:
                복원된 법인 상태 객체.
        """
        events = data.get("events")
        return cls(
            page_id=str(data["page_id"]),
            status=data.get("status", "pending"),
            created_at=data.get("created_at") or now_str(),
            updated_at=data.get("updated_at"),
        )

class SubsidiaryStore:
    """
    SubsidiaryState들을 법인명 기준으로 관리하는 법인 container.

    SubsidiaryStore는 법인 json snapshot 파일 하나에 대응하며, JSON 저장/로드 가능한 구조를 제공한다.

    내부 구조:
        - subsidiaries: 법인명 -> SubsidiaryState 매핑.

    주요 역할:
        1. SubsidiaryState 추가/조회
        2. 상태별 / event별 개수 집계
        3. JSON 직렬화/역직렬화 (tree snapshot과 같은 metadata 형식)
    """

    def __init__(self) -> None:
        """
        빈 SubsidiaryStore를 초기화한다.
        """
        self.subsidiaries: Dict[str, SubsidiaryState] = {}

    def add_subsidiary(self, subsidiary_name: str, state: SubsidiaryState) -> None:
        """
        SubsidiaryState를 법인명 기준으로 저장한다.
        동일 법인명이 이미 있으면 새 상태로 덮어쓴다.
        """
        self.subsidiaries[subsidiary_name] = state

    def get(self, subsidiary_name: str) -> Optional[SubsidiaryState]:
        """
        법인명에 해당하는 SubsidiaryState를 반환한다.

        Args:
            subsidiary_name: 조회할 법인명 (json key).

        Returns:
            Optional[SubsidiaryState]: 존재하면 SubsidiaryState, 없으면 None.
        """
        return self.subsidiaries.get(subsidiary_name)

    def __contains__(self, subsidiary_name: str) -> bool:
        """
        법인명이 SubsidiaryStore에 존재하는지 확인한다.

        Example:
            if subsidiary_name in registry:
                ...
        """
        return subsidiary_name in self.subsidiaries

    def __len__(self) -> int:
        """
        SubsidiaryStore에 저장된 법인 개수를 반환한다.
        """
        return len(self.subsidiaries)

    def iter_subsidiaries(self) -> Iterator[tuple[str, SubsidiaryState]]:
        """
        저장된 모든 (법인명, SubsidiaryState)를 순회한다.

        Returns:
            Iterator[tuple[str, SubsidiaryState]]: subsidiaries.items().
        """
        return iter(self.subsidiaries.items())

    def status_counts(self) -> Dict[str, int]:
        """
        상태별 법인 개수를 반환한다.

        Returns:
            Dict[str, int]: 예: {"deleted": 1, "pending": 3}
        """
        return dict(sorted(Counter(state.status for state in self.subsidiaries.values()).items()))

    def to_dict(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        SubsidiaryStore 전체를 JSON 저장 가능한 dict로 변환한다.

        저장 내용:
            - metadata: saved_at, root_source, schema_version, subsidiary_count 등 부가 정보
            - subsidiaries: 법인명 -> SubsidiaryState.to_dict() (법인명 순 정렬)

        Args:
            metadata:
                저장 파일에 함께 포함할 메타데이터.
                없으면 기본 metadata(schema_version, subsidiary_count)만 생성한다.

        Returns:
            Dict[str, Any]: 법인 json snapshot payload.
        """
        meta = dict(metadata or {})
        meta.setdefault("schema_version", "1.0")
        meta.setdefault("subsidiary_count", len(self.subsidiaries))
        meta["subsidiaries"] = {
            subsidiary_name: state.to_dict()
            for subsidiary_name, state in sorted(self.subsidiaries.items())
        }
        return meta

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SubsidiaryStore":
        """
        JSON payload를 SubsidiaryStore 객체로 복원한다.

        처리 흐름:
            1. payload["subsidiaries"]에서 SubsidiaryState를 복원한다.
            2. 이전 형식 파일이면 payload["corps"]를 대신 읽는다.
               값이 page_id 문자열인 경우({법인명: page_id})도 page_id만 채운 상태로 복원한다.
            3. 모든 법인을 SubsidiaryStore에 추가한다.

        Args:
            payload: SubsidiaryStore.to_dict()로 저장된 JSON payload.

        Returns:
            SubsidiaryStore: 복원된 법인 container.
        """
        registry = cls()
        subsidiaries = payload.get("subsidiaries")
        if subsidiaries is None:
            subsidiaries = payload.get("corps", {}) or {}

        for subsidiary_name, data in subsidiaries.items():
            if not isinstance(data, dict):
                data = {"page_id": data}
            registry.add_subsidiary(subsidiary_name, SubsidiaryState.from_dict(data))
        return registry