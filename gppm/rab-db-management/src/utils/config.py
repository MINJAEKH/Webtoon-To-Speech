"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : 파이프라인 설정 관리 모듈
 *               yaml 설정 로드와 디렉터리 검증, CLI 인자 파싱, 전역 config 제공을 담당한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

from easydict import EasyDict
import yaml
import logging
import os
import argparse
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def validate_directories(config):
    dir_keys = [
        "tree_path",
        "changed_urls_dir",
        "download_dir",
        "abandoned_dir",
        "document_processor_dir",
        "embed_and_update_dir",
        "db_source_dir",
        "local_image_dir",
    ]
    logger.info("validate directory")
    for dir_key in dir_keys:
        dir_path = config[dir_key]
        logger.info(f"{dir_key}: {dir_path}")
        if os.path.exists(dir_path):
            logger.info(f"use existing directory {dir_key}: {dir_path}")
        else:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"create new directory {dir_key}: {dir_path}")

def print_update_steps(config):
    update_steps = [
        "run_tree_discovery",
        "load_changed_urls",
        "run_downloader",
        "run_document_processor",
        "run_embed_and_update",
        "run_sync_db",
        "run_sync_db_all"
        ]
    update_status_strs = [ f"{update_step} : {config.get(update_step)}"  for update_step in update_steps]
    logger.info(" | ".join(update_status_strs))

_CONFIG: Optional[EasyDict] = None
_CONFIG_PATH: Optional[str] = None

# def _load_config_from_file(config_file_path: str | Path) -> EasyDict:
#     try:
#         with open(config_file_path, 'r') as file:
#             config = yaml.safe_load(file)
#         config = EasyDict(config)
#         validate_directories(config)
#     except FileNotFoundError:
#         logger.error(f"Error: The file '{config_file_path}' was not found in {os.getcwd()}.")
#     except yaml.YAMLError as exc:
#         logger.error(f"Error parsing YAML file: {exc}")
#     return config

def _load_config_from_file(config_file_path: str | Path) -> EasyDict:
    try:
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        config = EasyDict(config)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error: The file '{config_file_path}' was not found in {os.getcwd()}."
        ) from e
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Error parsing YAML file: {exc}"
        ) from exc
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="path of the config yaml file name")
    parser.add_argument('--subsidiary-list', nargs='*', default=[], help="list of subsidiary names")
    parser.add_argument("--airflow-start-date", type=str, default=None, help="Airflow API에서 전달한 start_date. Format: YYYYMMDD_HHMMSS")
    
    args = parser.parse_args()
    return args

def init_config(config_file_path: str | Path) -> EasyDict:
    """
    앱 시작 시 1회 호출해서 config를 초기화
    같은 path로 다시 호출 시 기존 값 반환
    다른 path로 재호출 시 log로 남김
    """
    global _CONFIG, _CONFIG_PATH

    config_file_path = str(config_file_path)

    if _CONFIG is not None:
        logger.debug(f"_CONFIG was not None... new _CONFIG is loaded from {config_file_path}")

    _CONFIG = _load_config_from_file(config_file_path)
    _CONFIG_PATH = config_file_path
    return _CONFIG

def get_config() -> EasyDict:
    """
    어디서든 불러서 쓰는 함수.
    init_config가 먼저 호출되지 않았으면 에러를 발생시켜줌.
    """
    if _CONFIG is None:
        raise RuntimeError(
            "Config is not initialized yet. "
            "Call init_config(config_file_path) once in main.py first."
        )
    return _CONFIG

def get_config_path() -> str:
    """
    어디서든 불러서 쓰는 함수.
    init_config가 먼저 호출되지 않았으면 에러를 발생시켜줌.
    """
    if _CONFIG_PATH is None:
        raise RuntimeError("Config is not initialized yet. Call init_config() first.")
    return _CONFIG_PATH