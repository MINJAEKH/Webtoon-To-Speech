"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : 환경별 vectorize 파이프라인을 기동하고 진행 상태를 조회하는 서비스 로직
 *               설정 파일로 파이프라인을 백그라운드 실행한 뒤 로그를 파싱해 상태를 판별한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

from pathlib import Path
import subprocess
import sys
import json

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

from src.api.schemas.vectorize_schema import (
    VectorizeRunRequest,
    VectorizeRunResponse,
    VectorizeStatusResponse,
)
from src.api.utils.vectorize_log_utils import (
    find_log_by_start_date,
    parse_airflow_status,
)

REPO_ROOT = Path("/app")

ENV_TO_CONFIG = { 
    "test_gppm": "config/params_config_test_gppm.yaml",
    "dev_gppm": "config/params_config_dev_gppm.yaml",
    "prod_gppm": "config/params_config_prod_gppm.yaml",
}

KST = ZoneInfo("Asia/Seoul")

def now_kst_timestamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")

def run_vectorize_process(environment: str, request: Optional[VectorizeRunRequest]=None) -> VectorizeRunResponse:
    if environment not in ENV_TO_CONFIG:
        raise ValueError(f"Unsupported environment: {environment}")

    config_path = ENV_TO_CONFIG[environment]
    config_file = REPO_ROOT / config_path

    # if not config_file.exists():
    #     raise FileNotFoundError(f"Config file not found: {config_file}")

    start_date = now_kst_timestamp()

    command = [
            sys.executable,
            "main.py",
            "--config",
            config_path,
            "--airflow-start-date",
            start_date,
    ]
    local_corps = request.local_corps if request else []

    if local_corps:
        command += ["--local-corps", *local_corps] 

        
    subprocess.Popen(
        command,
        cwd=REPO_ROOT.as_posix(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    return VectorizeRunResponse(
            start_date=start_date,
            local_corps=local_corps,
        )

def get_vectorize_status(
    environment: str,
    start_date: str,
) -> VectorizeStatusResponse:
    if environment not in ENV_TO_CONFIG:
        raise ValueError(f"Unsupported environment: {environment}")

    log_file = find_log_by_start_date(
        environment=environment,
        start_date=start_date,
    )

    status = parse_airflow_status(
        environment=environment,
        log_file=log_file,
    )

    return VectorizeStatusResponse(status=status)