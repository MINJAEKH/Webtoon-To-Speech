
"""
***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : 파이프라인 로깅 초기화와 로그 파일 보관 개수 관리
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
from src.utils.utils import file_managing

from datetime import datetime
# 한 번만 설정되도록 보장하기 위한 플래그
# _logger_configured = False

# 최대 log 파일 저장 갯수
MAX_LOG_FILES = 200

def setup_logging(
    level=logging.DEBUG,
    log_prefix="DEV",
    log_timestamp=None,
):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = f"./logs/{log_prefix}"
    os.makedirs(log_dir, exist_ok=True)

    if log_timestamp is None:
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filepath = f"{log_dir}/parameter_test_{log_timestamp}.log"

    file_handler = logging.FileHandler(log_filepath, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("로깅 설정이 완료되었습니다.")

def log_file_managing(log_prefix='DEV'):
    log_file_name = os.path.join(f"./logs/{log_prefix}/parameter_test_*.log")
    file_managing(log_file_name, MAX_LOG_FILES)