import os

import re
import shutil
import logging
from datetime import datetime
from typing import Dict

def _find_block(lines: list[str], block_key: str) -> tuple[int, int, str]:
    """
    yaml 줄 목록에서 최상위 block_key 블록의 시작 줄, 마지막 내용 줄, 항목 들여쓰기를 찾는다.

    Returns:
        (start, last, indent)
            start : "bootstrap_roots:" 줄 index
            last  : 블록 안에서 내용이 있는 마지막 줄 index (항목이 없으면 start)
            indent: 기존 항목의 들여쓰기 (항목이 없으면 "  ")
    """
    pattern = re.compile(rf"^{re.escape(block_key)}:\s*(#.*)?$")
    start = next((i for i, l in enumerate(lines) if pattern.match(l.rstrip("\n"))), None)
    if start is None:
        raise ValueError(f"'{block_key}:' block not found")

    last = start
    indent = "  "
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():                      # 빈 줄은 판단에서 제외
            continue
        if not line.startswith((" ", "\t")):      # 들여쓰기 없는 줄 = 다음 섹션 시작
            break
        last = i
        if not line.lstrip().startswith("#"):
            indent = line[: len(line) - len(line.lstrip())]
    return start, last, indent

def upsert_bootstrap_roots(
    yaml_path: str,
    roots: Dict[str, str],
    backup_dir: str = "./pipeline_storage/config_gppm",
    block_key: str = "bootstrap_roots",
) -> bool:
    """
    yaml의 bootstrap_roots 블록에 {법인명: page_id}를 덮어쓰기/추가한다.
    yaml 파일의 나머지 줄(구조, 띄어쓰기, 주석)은 그대로 유지한다.

    동작:
        1. yaml을 줄 단위로 읽고 bootstrap_roots 블록 범위를 찾는다.
        2. 블록 안에 같은 key가 있으면 page_id 값만 바꾼다 (들여쓰기, 줄 끝 주석 유지).
        3. 없는 key는 블록 마지막 항목 아래에 기존 들여쓰기로 추가한다.
        4. 바뀐 게 없으면 백업/저장 없이 끝낸다.
        5. 원본 yaml을 backup_dir/v_old/v_old_{파일명}_{YYYYMMDD_HHMMSS}.yaml 로 복사한다.
        6. 임시 파일에 쓴 뒤 os.replace로 원본 yaml을 바꿔친다. (v_new 파일은 만들지 않음)

    Args:
        yaml_path: 수정할 config yaml 경로 (예: config/params_config_dev_gppm.yaml).
        roots: 붙일 {법인명: page_id}.
        backup_dir: 백업 폴더. 그 아래 v_old/ 에 저장한다.
        block_key: 대상 블록 이름.

    Returns:
        bool: yaml이 바뀌었으면 True, 바뀐 게 없으면 False.
    """
    with open(yaml_path, encoding="utf-8") as f:
        lines = f.readlines()

    try:
        start, last, indent = _find_block(lines, block_key)
    except ValueError as e:
        logger.error("[CONFIG ROOTS][failed] %s | file=%s", e, yaml_path)
        raise

    changed = False
    for name, page_id in roots.items():
        page_id = str(page_id)
        # 들여쓰기 + key + ":" + 값(따옴표 유무 상관없음) + 나머지(공백/주석)
        key_re = re.compile(
            rf"^(?P<indent>[ \t]+){re.escape(name)}(?P<sep>\s*:\s*)"
            rf"(?P<value>\"[^\"]*\"|'[^']*'|[^\s#]*)(?P<rest>.*)$"
        )

        hit = None
        for i in range(start + 1, last + 1):
            m = key_re.match(lines[i].rstrip("\n"))
            if m:
                hit = (i, m)
                break

        if hit:
            i, m = hit
            old_value = m.group("value").strip("\"'")
            if old_value == page_id:
                logger.info("[CONFIG ROOTS] unchanged | %s=%s", name, page_id)
                continue
            newline = "\n" if lines[i].endswith("\n") else ""
            lines[i] = f'{m.group("indent")}{name}{m.group("sep")}"{page_id}"{m.group("rest")}{newline}'
            logger.info("[CONFIG ROOTS] overwrite | %s: %s -> %s", name, old_value, page_id)
        else:
            if not lines[last].endswith("\n"):     # 파일 끝에 줄바꿈이 없는 경우 대비
                lines[last] += "\n"
            last += 1
            lines.insert(last, f'{indent}{name}: "{page_id}"\n')
            logger.info("[CONFIG ROOTS] append | %s=%s", name, page_id)
        changed = True

    if not changed:
        logger.info("[CONFIG ROOTS] nothing to change | file=%s", yaml_path)
        return False

    # 원본 백업: backup_dir/v_old/v_old_{파일명}_{timestamp}{확장자}
    v_old_dir = os.path.join(backup_dir, "v_old")
    os.makedirs(v_old_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(yaml_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(v_old_dir, f"v_old_{stem}_{timestamp}{ext}")
    shutil.copy2(yaml_path, backup_path)
    logger.info("[CONFIG ROOTS] backup | %s", backup_path)

    # 임시 파일에 쓰고 원본과 바꿔치기
    tmp_path = f"{yaml_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp_path, yaml_path)
    except Exception as e:
        logger.error("[CONFIG ROOTS][FAIL] write error | file=%s | error=%s", yaml_path, e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    logger.info("[CONFIG ROOTS][SUCCESS] updated | file=%s", yaml_path)
    return True