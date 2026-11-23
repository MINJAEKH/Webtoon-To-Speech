"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : 환경별 로그 디렉터리 매핑과 로그 파싱 유틸
 *               실행 시작일로 로그 파일을 찾고 내용에서 running/done/failed 상태를 판별한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

from pathlib import Path

ENV_TO_LOG_DIR = {
    "test_gppm": Path("/app/logs/TEST_GPPM"),
    "dev_gppm": Path("/app/logs/DEV_GPPM"),
    "prod_gppm": Path("/app/logs/PROD_GPPM"),

}

ENV_TO_STAGE = {
    "test_gppm": "TEST_GPPM",
    "dev_gppm": "DEV_GPPM",
    "prod_gppm": "PROD_GPPM",
}

def find_log_by_start_date(
    environment: str,
    start_date: str,
) -> Path | None:
    log_dir = ENV_TO_LOG_DIR[environment]
    log_file = log_dir / f"parameter_test_{start_date}.log"

    if log_file.exists():
        return log_file

    return None

def parse_airflow_status(
    environment: str,
    log_file: Path | None,
) -> str:
    if log_file is None:
        return "running"

    stage = ENV_TO_STAGE[environment]

    content = log_file.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if f"[Airflow {stage}][FAIL]" in content:
        return "failed"

    if f"[Airflow {stage}][SUCCESS]" in content:
        return "done"

    return "running"