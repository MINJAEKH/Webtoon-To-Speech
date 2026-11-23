"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : 프로젝트 전역 공용 유틸 모음
 *               JSON 안전 로드, 파일명 정제, 버전 파일 관리, 최신 파일 추출, 배치 분할,
 *               이미지 개수 집계, 실행 시간 측정, 배치 폴더 관리, 재시도 데코레이터를 제공한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *    2       2026.11.02   soyeone.park              NERP&SEGAM 제거
 *********** ************ ************ ***************************************************** ************* *************
"""

import asyncio
import functools
import time
import os
import shutil
from datetime import datetime
from pathlib import Path
from easydict import EasyDict
import re
import hashlib
import base64
import json
import logging
import glob
from functools import wraps
from typing import Optional
from typing import Any
from IPython.display import Image
from langchain_core.runnables.graph import MermaidDrawMethod

from src.utils.config import get_config

logger = logging.getLogger(__name__)

def safe_load_json(json_file_path):
    """
    JSON 형식의 텍스트 파일에서 치환 딕셔너리를 로드합니다.
    """
    path = Path(json_file_path)
    if not path.exists():
        logger.error(f"File does not exist: {path}")
    if path.suffix.lower() != ".json":
        logger.error(f"File extension for load json is not JSON! extension = {json_file_path}")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_file = json.load(f)
        logger.info(f"Successfully loaded filepath = {json_file_path}")
        return json_file
    except Exception as e:
        logger.error(f"WARNING: Failed to load dictionary from JSON file '{json_file_path}': {e}")
        return {}

def format_context(docs):
    context = ''
    for i, doc in enumerate(docs):
        context += f'context {i + 1}:{doc.page_content}\n'
        if doc.metadata['include_image']:
            if isinstance(doc.metadata['image_info'], str):
                image_info = json.loads(doc.metadata['image_info'])  # child_chunk
            else:
                image_info = doc.metadata['image_info']  # parent_chunk
            for j in range(len(image_info)):
                img_path = image_info[j]['image_path']
                context += f'images {j}: {img_path} \n'
        else:
            context += '\n'
    return context

def pretty_print_docs(docs):
    print(
        f'\n{"-" * 100}\n'.join(
            [
                f'Document {i + 1}:\n\n'
                #  + f"source {d.metadata['source']}\n"
                + f'source {d.metadata["title"]}\n'
                #  + f"Page {d.metadata['page']}\n"
                + d.page_content
                for i, d in enumerate(docs)
            ]
        )
    )

def save_output(state, top_k=30, output_file='./output.json'):
    output = dict()
    if os.path.exists(output_file):
        # with open(output_file, 'r', encoding='utf-8') as f:
        #     output = json.load(f)
        otuput = safe_load_json(output_file)
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output.update(
        {
            state['question']: {
                # 'answer': state['answer'],
                'documents': [
                    {
                        'rank': i,
                        'title': d.metadata['title'],
                        'source': d.metadata['source'],
                        'page': d.metadata['page'],
                        'header1': d.metadata['header1'],
                        'header2': d.metadata['header2'],
                        'header3': d.metadata['header3'],
                        'image_info': d.metadata['image_info'],
                        'text': d.page_content,
                    }
                    for i, d in enumerate(state['documents'][:top_k])
                ]
            }
        }
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

def sanitize_file_name(name: str) -> str:
    FORBIDDEN_CHARS = r'[\s%&?!<>/=$#\\`^|{}\[\]+]'  # regex 패턴 (특수문자 제거)
    MAX_LENGTH = 40
    # url 처리
    if name.startswith('http'):
        short_hash = name.split("=")[-1]
        name = base64.urlsafe_b64encode(name.encode('utf-8')).decode('utf-8')
    else :
        # 너무 짧아지는 것 방지 및 충돌 방지용 해시 추가
        short_hash = hashlib.md5(name.encode()).hexdigest()[:6]

    # 확장자 분리
    base, ext = os.path.splitext(name)

    # 금지 문자 제거
    cleaned = re.sub(FORBIDDEN_CHARS, '', base)

    # 40자 제한을 맞추되, 해시를 고려해 자르기
    trimmed = cleaned[: MAX_LENGTH - len(short_hash) - 1]  # -1 for underscore
    sanitized_name = f'{trimmed}_{short_hash}{ext}'
    return sanitized_name

def get_api_header(api_token):
    """
    If using the SHUTTLE model, headers are required.
    """
    return {'Authorization': f'Bearer {api_token}'} if api_token else None

def convert_pipes_to_xml(input_str):
    if not isinstance(input_str, str):
        return input_str

    # XML 특수 문자 이스케이프
    escaped = input_str.replace("&", "&amp;")  # & → &amp;

    return escaped

def save_prompt(state, output_file='./output_prompt.json'):
    output_file = output_file.replace('.json', '_prompt.json')
    output = dict()
    if os.path.exists(output_file):
        output = safe_load_json(output_file)
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output.update(
        {
            state['question']: state['prompt']
        }
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

# def measure_time(_func=None, *, name: Any = None):
#     def decorator(func):
#         # 비동기 함수용 async wrapper
#         if asyncio.iscoroutinefunction(func):
#             @functools.wraps(func)
#             async def async_wrapper(*args, **kwargs):
#                 display_name = name
#                 # self를 인자로 받아 동적으로 name을 생성할 수 있도록 수정
#                 if callable(name) and args:
#                     display_name = name(args[0]) # args[0]이 self 입니다.
#                 elif name is None:
#                     display_name = func.__name__

#                 start_time = time.time()
#                 result = await func(*args, **kwargs)
#                 end_time = time.time()
#                 print(f"(async) '{display_name}' 실행 시간: {end_time - start_time:.5f} 초")
#                 return result
#             return async_wrapper
#         # 동기 함수용 sync wrapper
#         else:
#             @functools.wraps(func)
#             def sync_wrapper(*args, **kwargs):
#                 display_name = name
#                 # self를 인자로 받아 동적으로 name을 생성할 수 있도록 수정
#                 if callable(name) and args:
#                     display_name = name(args[0]) # args[0]이 self 입니다.
#                 elif name is None:
#                     display_name = func.__name__

#                 start_time = time.time()
#                 result = func(*args, **kwargs)
#                 end_time = time.time()
#                 print(f"'{display_name}' 실행 시간: {end_time - start_time:.5f} 초")
#                 return result
#             return sync_wrapper

#     if _func is None:
#         return decorator
#     else:
#         return decorator(_func)

def read_text(file_path) :
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def count_unique_image_tags_in_html(zip_output_dir: str, html_filename: str = 'input.html') -> int:
    '''HTML 파일 내 고유한 이미지만 카운트하는 함수 (중복 제거)

    Args:
        zip_output_dir (str): HTML 파일이 있는 디렉토리 경로
        html_filename (str): HTML 파일명 (기본값: 'input.html')

    Returns:
        int : (고유 이미지 개수)
    '''
    html_path = os.path.join(zip_output_dir, html_filename)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    try:
        if not os.path.exists(html_path):
            print(f"존재하지 않는 HTML 파일입니다: {html_path}")
            return 0

        html_content = read_text(html_path)

        image_count = {}  # 파일명별 등장 횟수

        # 1. <ac:image> 태그 처리
        ac_image_pattern = r'<ri:attachment\s+ri:filename="([^"]+)"[^>]*>'
        ac_matches = re.finditer(ac_image_pattern, html_content, re.IGNORECASE)

        for match in ac_matches:
            filename = match.group(1).strip()
            if any(filename.lower().endswith(ext) for ext in image_extensions) :
                image_count[filename] = image_count.get(filename, 0) + 1

        # 2. <img> 태그 처리
        img_pattern = r'<img[^>]+src="([^"]+)"[^>]*/?>'
        img_matches = re.finditer(img_pattern, html_content, re.IGNORECASE)

        for match in img_matches:
            src = match.group(1).strip()

            # Confluence UI 아이콘은 첨부가 아니므로 제외
            if re.match(r"^https?://", src) and "/download/attachments/" not in src:
                continue

            filename = os.path.basename(src.split("?")[0])
            if any(filename.lower().endswith(ext) for ext in image_extensions) :
                image_count[filename] = image_count.get(filename, 0) + 1

        unique_count = len(image_count)
        return unique_count

    except Exception as e:
        print(f"Error while count image tags in html documents  : {e}")
        return 0

def count_images_in_folder(zip_output_dir : str) -> int :
    '''컨플루언스 이미지 파일 개수 count'''

    img_counts = 0
    image_folder = os.path.join(zip_output_dir, 'images')
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    try :
        if not os.path.exists(image_folder):
            print('경로가 존재하지 않습니다.')
            return 0

        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)

            if not os.path.isfile(image_path):
                continue

            if not any(filename.lower().endswith(ext) for ext in image_extensions):
                print(f"이미지가 아닌 파일 삭제됨 : {filename}")
                os.remove(image_path)
            else :
                img_counts += 1
        return img_counts

    except Exception as e :
        print(f"Error while counting images in image folders  : {e}")
        return 0

def sorted_backuped_batch_folders(source_dir):
    # 3) backup 폴더들 전부 찾기
    backup_prefix = f"{str(source_dir)}_backup_*"
    backup_dirs = [
        Path(p) for p in glob.glob(backup_prefix)
        if Path(p).is_dir()
    ]
    # 4) 이름 기준 정렬 (YYYYMMDD_HHMMSS 형태 → 문자열 정렬 = 시간순)
    backup_dirs_sorted = sorted(backup_dirs, key=lambda p: p.name)
    logger.info(
        f"[backuped_batch_folders] Found {len(backup_dirs_sorted)} backup folders: "
        + ", ".join(str(p) for p in backup_dirs_sorted)
    )
    return backup_dirs_sorted

def manage_version_files(output_dir, filename_prefix):
    """기존 파일을 outdated 파일명으로 변경"""
    # v_old로 시작하는 파일 찾기
    os.makedirs(os.path.join(output_dir, "v_old"), exist_ok=True)
    v_old_pattern = os.path.join(output_dir, "v_old", f"v_old_{filename_prefix}_*")
    v_new_pattern = os.path.join(output_dir, f"v_new_{filename_prefix}_*")

    v_old_files = glob.glob(v_old_pattern)
    v_new_files = glob.glob(v_new_pattern)

    if len(v_old_files) == 1 and len(v_new_files) == 0:
        logger.info("Version management skipped: v_old files <= 1, no v_new files")
        return

    # v_new -> v_old 변경
    for v_new_file in v_new_files:
        try:
            filename = os.path.basename(v_new_file)
            new_filename = filename.replace('v_new_', 'v_old_', 1)
            go_to_path = os.path.join(output_dir, "v_old", f"{new_filename}")

            os.rename(v_new_file, go_to_path)
            logger.info(f"[SUCCESS] Renamed file v_new -> v_old: {filename} -> {new_filename}")
        except Exception as e:
            logger.error(f"[confluence list extractor] Failed to rename file: {v_new_file} - {e}")

def extract_newest_file(dir_path:str, file_prefix:str, file_extension:str, is_multiple_error = False) -> str:
    files_path = os.path.join(dir_path, f"{file_prefix}_*.{file_extension}")
    files = glob.glob(files_path)
    if len(files) == 0:
        logger.error("No such file exists!")
    if len(files) > 1 and is_multiple_error:
        logger.info(f"More than one {file_prefix}_*.{file_extension} files exist.")
    return max(files, key=os.path.getmtime) if files else None

    
def extract_timestamp(file_name:str, file_prefix:str):
    timestamp = file_name.split(file_prefix)[-1].split(".")[0]
    return timestamp

    
def batched_by_size(seq, size: int):
    """seq를 size 크기의 연속 청크로 잘라서 반환"""
    for start in range(0, len(seq), size):
        yield seq[start:start + size]

    
def batched(source_urls, batch_num_df):
    """source_urls를 같은 entity parquet에 들어있는 기준으로 잘라서 반환 (한 파일에 40여개 정도)"""
    source_url_batch_num = batch_num_df[batch_num_df['source'].isin(source_urls)]
    indices = source_url_batch_num['idx'].drop_duplicates().tolist()
    indices = sorted(indices)
    for idx in indices:
        batched_urls = source_url_batch_num[source_url_batch_num['idx'] == idx]['source'].tolist()
        yield batched_urls

    
def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed_min, elapsed_sec = divmod(end -start, 60)
        logger.info(f"{func.__name__} executed in {int(elapsed_min)} min {elapsed_sec: .2f} sec.")
        return result
    return wrapper

def all_file_managing(config: EasyDict):
    MAX_FILE_NUM = 200

    managed_patterns = [
        os.path.join(config.tree_path, "v_old", "v_old_confluence_tree_*.json"),
        os.path.join(config.changed_urls_dir, "v_old", "v_old_changed_urls_*.json"),
        os.path.join(config.document_processor_dir, "v_old", f"v_old_{config.confluence_dump_file_prefix}_*.parquet"),
        os.path.join(config.cron_log_dir, "cron_*.log"),
        # 추가 ♥️ 
        os.path.join(config.subsidiary_path, "v_old", "v_old_subsidiary_master_*.json"),
        os.path.join(config.config_path, "v_old", "v_old_params_config_dev_gppm_*.json"),
    ]
    # if config.get("subsidiary_mast_codes_dir"):
    #     managed_patterns.append(
    #         os.path.join(config.subsidiary_mast_codes_dir, "v_old", f"v_old_{config.subsidiary_mast_codes_file_prefix}_*.json")
    #     )

    for file_pattern in managed_patterns:
        file_managing(file_pattern, MAX_FILE_NUM)

def file_managing(file_name: str, MAX_FILE_NUM: int):
    file_path = os.path.join(file_name)
    all_files = glob.glob(file_path)
    all_files.sort()    # sort by file name

    logger.info(f"Managing file name with pattern: {file_name}")
    if (over_count := len(all_files) - MAX_FILE_NUM) > 0:
        remove_files = all_files[:over_count]
        for file in remove_files:
            os.remove(file)
        logger.info(f"Removed file names are {remove_files}")
        logger.info(f"Managing folder: {len(remove_files)} files are removed.")
        return
    logger.info(f"Managing folder: No files are removed.")

def batch_folder_managing(config, MAX_FOLDER_NUM: int = 3):
    """
    config.db_source_dir 폴더를 현재 날짜/시간 기준으로 백업하고,
    backup 폴더 개수가 MAX_FOLDER_NUM을 넘으면
    가장 오래된 폴더부터 삭제한다.
    """
    source_dir = Path(config.db_source_dir).resolve()

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source dir not found: {source_dir}")

    # 1) 날짜 + 시간 붙여서 백업 폴더 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"{str(source_dir)}_backup_{timestamp}")

    # 2) 백업 폴더 생성 + 내용 복사
    if backup_dir.exists():
        logger.info(f"[batch_folder_managing] Backup folder already exists: {backup_dir}")
    else:
        logger.info(f"[batch_folder_managing] Creating backup folder: {backup_dir}")
        shutil.copytree(source_dir, backup_dir)
        logger.info(f"[batch_folder_managing] Backup completed from {source_dir} to {backup_dir}")

    backup_dirs_sorted = sorted_backuped_batch_folders(source_dir)

    removed_dirs = []
    if (over_count := len(backup_dirs_sorted) - MAX_FOLDER_NUM) > 0:
        to_remove = backup_dirs_sorted[:over_count]
        for old_dir in to_remove:
            logger.info(f"[batch_folder_managing] Removing old backup folder: {old_dir}")
            shutil.rmtree(old_dir)
            removed_dirs.append(str(old_dir))
    logger.info(
        f"[batch_folder_managing] Done. Kept {len(backup_dirs_sorted) - len(removed_dirs)} "
        f"backups, removed {len(removed_dirs)}."
    )
    return removed_dirs

def build_hashed_dir_name(filepath: str, page_id: str) -> str:
    FORBIDDEN_CHARS = r'[\s%&?!<>/=$#\\`^|{}$begin:math:display$$end:math:display$+]'
    MAX_LENGTH = 40

    encoded = base64.urlsafe_b64encode(filepath.encode("utf-8")).decode("utf-8")
    base, ext = os.path.splitext(encoded)
    cleaned = re.sub(FORBIDDEN_CHARS, '', base)
    trimmed = cleaned[: MAX_LENGTH - len(str(page_id)) - 1]

    return f"{trimmed}_{page_id}{ext}"

class RetryOnException:
    max_retries = 3
    initial_delay = 1
    backoff_factor = 2
    exceptions=(Exception,)

    @classmethod
    def configure(cls, *, max_retries=None, initial_delay=None, backoff_factor=None):
        if max_retries is not None:
            cls.max_retries = max_retries
        if initial_delay is not None:
            cls.initial_delay = initial_delay
        if backoff_factor is not None:
            cls.backoff_factor = backoff_factor

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _max_retries = self.max_retries
            _initial_delay = self.initial_delay
            _backoff_factor = self.backoff_factor

            if _max_retries is None or _initial_delay is None or _backoff_factor is None:
                raise RuntimeError(
                    "RetryOnException is not configured. "
                    "Call RetryOnException.configure(...) in main after init_config()."
                )

            delay = _initial_delay
            for attempt in range(1, _max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    if attempt == _max_retries:
                        logging.error(f"[FINAL FAIL] {func.__name__} failed after {_max_retries} attempts")
                        raise
                    logging.warning(
                        f"[RETRY] {func.__name__} failed (attempt {attempt}/{_max_retries}) "
                        f"→ {delay} sec sleep. Error: {e}"
                    )
                    time.sleep(delay)
                    delay *= _backoff_factor
        return wrapper