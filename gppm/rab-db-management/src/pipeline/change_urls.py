"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : compare_results를 downstream 단계가 사용하는 changed_urls로 변환하는 모듈
 *               page_id별 상세 변경 이벤트를 변경 유형별 page_id 목록으로 재구성하고,
 *               버전 파일로 저장/최신본 로드/변경 내역 로깅을 수행한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

import json
import logging
import os
import re
from datetime import datetime

from src.utils.utils import manage_version_files

logger = logging.getLogger(__name__)

def build_changed_urls_from_compare_results(
    compare_results,
    delete_limit: int,
    excluded_ids: list[str] | None = None,
) -> dict:
    """
    tree discovery 단계에서 생성된 compare_results를 downstream 파이프라인에서 사용할
    changed_urls 형태로 변환한다.

    compare_results는 page_id별 상세 변경 이벤트를 담고 있고,
    changed_urls는 변경 유형별 page_id 목록을 담는다.

    변환 목적:
        - downloader 단계:
            created / updated page 다운로드
            deleted page 로컬 산출물 제거
        - document processor 단계:
            변경된 page 기준 dump / caption 갱신
        - embedding / DB sync 단계:
            변경 유형별 Milvus / MinIO 동기화 수행

    Args:
        compare_results:
            ConfluenceTreeDiscovery._compare_trees()에서 생성된 page_id별 변경 결과.
            각 result는 dict 또는 direct_events/change_flags/impact_events 속성을 가진 객체일 수 있다.

        delete_limit:
            한 번의 실행에서 삭제 처리할 최대 page 수.
            tree 오류나 설정 실수로 대량 삭제가 발생하는 것을 방지하기 위한 safety limit.

        excluded_ids:
            변경 처리에서 제외할 page_id 목록.
            config에서 제외 대상으로 지정한 page는 어떤 이벤트가 있어도 changed_urls에 포함하지 않는다.

    Returns:
        dict:
            {
                "created": [...],
                "updated": [...],
                "deleted": [...],
                "moved": [...],
                "ancestor_path_changed": [...],
                "ancestor_title_changed": [...],
                "structure_changed": [...],
                "labels_changed": [...],
            }
    """
    
    excluded_ids = set(map(str, excluded_ids or []))

    # 변경 유형별 page_id를 set으로 수집한다.
    # 중복 이벤트가 들어오더라도 최종 결과에서는 page_id가 한 번만 남도록 하기 위함이다.
    created_ids = set()
    updated_ids = set()
    deleted_ids = set()
    moved_ids = set()
    ancestor_path_changed_ids = set()
    ancestor_title_changed_ids = set()
    structure_changed_ids = set()
    labels_changed_ids = set()

    for page_id, result in (compare_results or {}).items():
        # compare_results의 result는 dict일 수도 있고,
        # dataclass/object 형태일 수도 있으므로 두 형태를 모두 처리한다.
        if isinstance(result, dict):
            direct_events = set(result.get("direct_events", []))
            change_flags = set(result.get("change_flags", []))
            impact_events = set(result.get("impact_events", []))
        else:
            direct_events = set(getattr(result, "direct_events", []))
            change_flags = set(getattr(result, "change_flags", []))
            impact_events = set(getattr(result, "impact_events", []))

        page_id = str(page_id)

        # config에서 제외한 page는 downstream 어느 단계에서도 처리하지 않는다.
        if page_id in excluded_ids:
            continue

        # deleted page는 다운로드/재처리 대상이 아니므로 다른 이벤트보다 우선 처리한다.
        # 삭제된 page에 updated/moved 같은 이벤트가 같이 들어와도 deleted만 남긴다.
        if "deleted" in direct_events:
            deleted_ids.add(page_id)
            continue

        # 신규 page는 downloader에서 새로 받아와야 한다.
        if "created" in direct_events:
            created_ids.add(page_id)
            continue
        # 수정된 page는 downloader, processor, embedding, DB update 대상이다.
        if "updated" in direct_events:
            updated_ids.add(page_id)

        # moved는 parent_id가 바뀐 page다.
        # page 본문 자체가 안 바뀌어도 ancestor/page_tree metadata가 달라질 수 있다.
        if "moved" in direct_events:
            moved_ids.add(page_id)

        # ancestor id 경로가 바뀐 page다.
        # 검색 결과에 노출되는 문서 경로 또는 metadata 갱신이 필요할 수 있다.
        if "ancestor_path_changed" in impact_events:
            ancestor_path_changed_ids.add(page_id)

        # ancestor title 경로가 바뀐 page다.
        # 부모 title 변경이 하위 page의 page_tree에 영향을 준 경우다.
        if "ancestor_title_changed" in impact_events:
            ancestor_title_changed_ids.add(page_id)

        # direct child 구성이 바뀐 parent page다.
        # tree 구조 변화가 있었음을 downstream에 전달하기 위한 별도 분류다.
        if "children_changed" in direct_events:
            structure_changed_ids.add(page_id)

        # title_changed만 있고 updated 이벤트가 없는 경우에도 page metadata 갱신이 필요하다.
        # 현재 compare 쪽에서 title_changed를 항상 쓰지 않더라도 이전 구조와 호환되도록 유지한다.
        if "title_changed" in change_flags and "updated" not in direct_events:
            updated_ids.add(page_id)

        # label 변경은 본문 수정과 별개로 DB metadata 갱신이 필요하므로 별도 분류한다.
        if "labels_changed" in change_flags:
            labels_changed_ids.add(page_id)

    # 삭제는 가장 위험한 작업이므로 safety limit을 적용한다.
    # delete_limit을 초과한 page는 이번 실행에서 삭제하지 않고 로그만 남긴다.
    all_deleted_ids = sorted(deleted_ids)
    deleted_overflow = []

    if len(all_deleted_ids) > delete_limit:
        deleted_overflow = all_deleted_ids[delete_limit:]
        deleted_ids = set(all_deleted_ids[:delete_limit])

        logger.warning(
            "[DELETE LIMIT APPLIED] deleted=%s > limit=%s | allowed=%s | skipped=%s",
            len(all_deleted_ids),
            delete_limit,
            len(deleted_ids),
            len(deleted_overflow),
        )
        logger.warning("[DELETE LIMIT SKIPPED IDS] %s", deleted_overflow[:100])
    else:
        deleted_ids = set(all_deleted_ids)

    # 이벤트 우선순위 정리
    # created page는 updated로도 잡힐 수 있으므로 created를 우선한다.
    # deleted page는 모든 변경보다 우선한다.
    updated_ids -= deleted_ids | created_ids
    
    # 모든 결과는 sorted list로 반환한다.
    # json 저장 결과의 재현성을 높이고, 로그 비교를 쉽게 하기 위함이다.
    return {
        "created": sorted(created_ids),
        "updated": sorted(updated_ids),
        "deleted": sorted(deleted_ids),
        "moved": sorted(moved_ids),
        "ancestor_path_changed": sorted(ancestor_path_changed_ids),
        "ancestor_title_changed": sorted(ancestor_title_changed_ids),
        "structure_changed": sorted(structure_changed_ids),
        "labels_changed": sorted(labels_changed_ids),
    }

def save_changed_urls_versioned(
    changed_urls: dict,
    output_dir: str,
    filename_prefix: str,
    meta: dict | None = None,
) -> str:
    """
    changed_urls를 timestamp가 포함된 versioned json 파일로 저장한다.

    저장 구조:
        output_dir/
        ├── v_new_{filename_prefix}_YYYYMMDD_HHMMSS.json
        └── v_old/
            └── 이전 changed_urls 파일들

    Args:
        changed_urls:
            변경 유형별 page_id 목록.

        output_dir:
            changed_urls json 파일을 저장할 디렉터리.

        filename_prefix:
            저장 파일명에 사용할 prefix.

        meta:
            실행 환경, tree_path, compare_count 등 부가 정보.
            추후 디버깅과 재실행 기준 확인을 위해 함께 저장한다.

    Returns:
        str:
            저장된 changed_urls json 파일 경로.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "v_old"), exist_ok=True)
    manage_version_files(output_dir, filename_prefix)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"v_new_{filename_prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    payload = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta or {},
        "changed_urls": changed_urls,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filepath

def load_latest_changed_urls_versioned(
    output_dir: str,
    filename_prefix: str,
) -> dict:
    """
    output_dir에서 가장 최신 changed_urls versioned json을 로드한다.

    사용 목적:
        - tree discovery를 다시 실행하지 않고 이후 단계만 재실행할 때
        - 특정 changed_urls 기준으로 downloader / processor / DB sync를 재현할 때
        - 디버깅 중 동일 변경 목록을 반복 사용해야 할 때

    Args:
        output_dir:
            changed_urls json 파일이 저장된 디렉터리.

        filename_prefix:
            찾을 파일의 prefix.
            파일명 형식은 v_new_{filename_prefix}_YYYYMMDD_HHMMSS.json 이다.

    Returns:
        dict:
            변경 유형별 page_id 목록.
            모든 page_id는 str로 변환하고 sorted list로 반환한다.
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"changed_urls dir not found: {output_dir}")

    pattern = re.compile(
        rf"^v_new_{re.escape(filename_prefix)}_\d{{8}}_\d{{6}}\.json$"
    )

    candidates = [
        filename
        for filename in os.listdir(output_dir)
        if pattern.match(filename)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No changed_urls file found in {output_dir} with prefix {filename_prefix}"
        )

    candidates.sort()
    latest_filename = candidates[-1]
    latest_filepath = os.path.join(output_dir, latest_filename)

    with open(latest_filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    changed_urls = payload.get("changed_urls", {})

    return {
        "created": sorted(map(str, changed_urls.get("created", []))),
        "updated": sorted(map(str, changed_urls.get("updated", []))),
        "deleted": sorted(map(str, changed_urls.get("deleted", []))),
        "moved": sorted(map(str, changed_urls.get("moved", []))),
        "ancestor_path_changed": sorted(map(str, changed_urls.get("ancestor_path_changed", []))),
        "ancestor_title_changed": sorted(map(str, changed_urls.get("ancestor_title_changed", []))),
        "structure_changed": sorted(map(str, changed_urls.get("structure_changed", []))),
        "labels_changed": sorted(map(str, changed_urls.get("labels_changed", []))),
    }

def log_changed_urls(changed_urls: dict):
    """
    changed_urls의 변경 유형별 page_id 목록을 로그로 출력한다.

    pipeline 실행 후 어떤 page가 어떤 이유로 처리 대상이 되었는지
    빠르게 확인하기 위한 디버깅/운영 로그 함수다.

    Args:
        changed_urls:
            build_changed_urls_from_compare_results 또는
            load_latest_changed_urls_versioned에서 반환된 변경 목록.
    """
    logger.info("created_ids: %s", changed_urls["created"])
    logger.info("updated_ids: %s", changed_urls["updated"])
    logger.info("deleted_ids: %s", changed_urls["deleted"])
    logger.info("moved_ids: %s", changed_urls["moved"])
    logger.info("ancestor_path_changed_ids: %s", changed_urls["ancestor_path_changed"])
    logger.info("ancestor_title_changed_ids: %s", changed_urls["ancestor_title_changed"])
    logger.info("structure_changed_ids: %s", changed_urls["structure_changed"])
    logger.info("labels_changed_ids: %s", changed_urls.get("labels_changed", []))