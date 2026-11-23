"""
 ***********************************************************************************************************************
 * Creator     : minjae.cha
 * Create Date : 2026.11.20
 * Description : 법인(Subsidiary) json snapshot 파일 로드/저장
 *               v_new 최신 파일을 SubsidiaryStore로 읽고(없으면 v_old 최신 파일),
 *               manage_version_files로 이전 v_new를 v_old로 옮긴 뒤 새 v_new를 저장한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.11.20   minjae.cha                     최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

import os
import glob
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.utils.utils import manage_version_files
from src.components.subsidiary.subsidiary_models import SubsidiaryStore

logger = logging.getLogger(__name__)

def load_subsidiaries(sub_dir: str, filename_prefix: str = "subsidiary_master") -> SubsidiaryStore:
    """
    sub_dir에서 최신 법인 snapshot을 로드하여 SubsidiaryStore로 반환한다.

    동작:
        1. sub_dir가 없으면 생성
        2. v_new_{prefix}_*.json 중 최신 파일 탐색
        3. v_new가 없으면 v_old/v_old_{prefix}_*.json 중 최신 파일 탐색
        4. 둘 다 없으면 빈 SubsidiaryStore 반환 (첫 실행)
        5. 있으면 JSON을 읽어 SubsidiaryStore 복원

    Args:
        sub_dir: 법인 snapshot 파일 디렉터리
        filename_prefix: 파일 prefix

    Returns:
        SubsidiaryStore: 최신 법인 snapshot. 파일이 없으면 빈 SubsidiaryStore.
    """
    os.makedirs(sub_dir, exist_ok=True)

    files = glob.glob(os.path.join(sub_dir, f"v_new_{filename_prefix}_*.json"))
    if not files:
        logger.info("[SUBSIDIARY LOAD] no v_new file, try v_old | dir=%s", sub_dir)
        files = glob.glob(os.path.join(sub_dir, "v_old", f"v_old_{filename_prefix}_*.json"))

    if not files:
        logger.info("[SUBSIDIARY LOAD] no v_new / v_old file to compare (first run) | dir=%s", sub_dir)
        return SubsidiaryStore()

    latest_file = max(files, key=os.path.getmtime)

    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.error("[SUBSIDIARY LOAD][FAIL] file=%s | error=%s", latest_file, e)
        raise

    store = SubsidiaryStore.from_dict(payload)
    logger.info("[SUBSIDIARY LOAD] loaded %d subsidiaries | file=%s", len(store), latest_file)
    return store

def save_subsidiaries(
    sub_dir: str,
    store: SubsidiaryStore,
    metadata: Optional[Dict[str, Any]] = None,
    filename_prefix: str = "subsidiary_master",
) -> str:
    """
    SubsidiaryStore를 versioned JSON 파일로 저장한다.

    저장 구조:
        sub_dir/
        ├── v_new_{prefix}_YYYYMMDD_HHMMSS.json   ← 최신
        └── v_old/                               ← 이전 버전들

    동작:
        1. 디렉터리 생성
        2. 기존 v_new 파일을 v_old로 이동 (manage_version_files)
        3. 현재 store를 timestamp 기반 파일로 저장

    Args:
        sub_dir: 저장 디렉터리
        store: 저장할 SubsidiaryStore 객체
        metadata: 저장 시 함께 포함할 메타데이터 (예: saved_at, root_source)
        filename_prefix: 파일 prefix

    Returns:
        str: 저장된 파일 경로
    """
    os.makedirs(os.path.join(sub_dir, "v_old"), exist_ok=True)
    manage_version_files(sub_dir, filename_prefix)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(sub_dir, f"v_new_{filename_prefix}_{timestamp}.json")

    results = store.to_dict(metadata=metadata)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return filepath