"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : tree snapshot 파일의 입출력 유틸
 *               tree 디렉터리에서 최신 snapshot 파일을 찾아 TreeStore로 로드하고,
 *               버전 관리 규칙에 따라 새 snapshot을 저장한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

import glob
import json
import os
from datetime import datetime
from typing import Any

from src.components.tree.tree_models import TreeStore
from src.utils.utils import manage_version_files

def get_latest_tree_file(tree_dir: str, filename_prefix: str = "confluence_tree") -> str:
    """
    tree_dir에서 가장 최신(confluence_tree) 파일 경로를 반환한다.

    파일명 규칙:
        v_new_{filename_prefix}_YYYYMMDD_HHMMSS.json

    동작:
        - 해당 패턴의 파일이 있으면 가장 최근 수정된 파일을 반환
        - 없으면 초기 파일 경로(v_new_*_initial.json)를 반환 (실제 파일이 없어도 fallback path 제공)

    Args:
        tree_dir: tree json 파일들이 저장된 디렉터리
        filename_prefix: 파일 prefix (기본값: confluence_tree)

    Returns:
        str: 최신 tree 파일 경로 또는 초기 파일 경로
    """
    pattern = os.path.join(tree_dir, f"v_new_{filename_prefix}_*.json")
    files = glob.glob(pattern)
    if not files:
        return os.path.join(tree_dir, f"v_new_{filename_prefix}_initial.json")
    return max(files, key=os.path.getmtime)

def load_tree(tree_dir: str, filename_prefix: str = "confluence_tree") -> TreeStore:
    """
    tree_dir에서 최신 tree snapshot을 로드하여 TreeStore 객체로 반환한다.

    동작:
        1. tree_dir가 없으면 생성
        2. v_new_{prefix}_*.json 파일 중 최신 파일 탐색
        3. 파일이 없으면 빈 TreeStore 반환
        4. 있으면 JSON을 읽어 TreeStore로 복원

    Args:
        tree_dir: tree snapshot 파일 디렉터리
        filename_prefix: 파일 prefix

    Returns:
        TreeStore:
            - 최신 tree snapshot
            - 파일이 없으면 빈 TreeStore
    """
    os.makedirs(tree_dir, exist_ok=True)

    pattern = os.path.join(tree_dir, f"v_new_{filename_prefix}_*.json")
    files = glob.glob(pattern)

    if not files:
        return TreeStore()

    latest_file = max(files, key=os.path.getmtime)

    with open(latest_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return TreeStore.from_dict(payload)

def save_tree(
    tree_dir: str,
    tree: TreeStore,
    metadata: dict[str, Any] | None = None,
    filename_prefix: str = "confluence_tree",
) -> str:
    """
    TreeStore를 versioned JSON 파일로 저장한다.

    저장 구조:
        tree_dir/
        ├── v_new_{prefix}_YYYYMMDD_HHMMSS.json   ← 최신
        └── v_old/                               ← 이전 버전들

    동작:
        1. 디렉터리 생성
        2. 기존 v_new 파일을 v_old로 이동 (버전 관리)
        3. 현재 tree를 timestamp 기반 파일로 저장

    Args:
        tree_dir: 저장 디렉터리
        tree: 저장할 TreeStore 객체
        metadata:
            저장 시 함께 포함할 메타데이터
            (예: saved_at, root_source 등)
        filename_prefix: 파일 prefix

    Returns:
        str: 저장된 파일 경로
    """

    
    os.makedirs(tree_dir, exist_ok=True)
    os.makedirs(os.path.join(tree_dir, "v_old"), exist_ok=True)

    manage_version_files(tree_dir, filename_prefix)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(tree_dir, f"v_new_{filename_prefix}_{timestamp}.json")

    payload = tree.to_dict(metadata=metadata)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filepath