import os
import re
import shutil
import logging
import tempfile
from datetime import datetime
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


# =========================================================
# 줄 단위 helper (파일 IO 없음)
# =========================================================
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
    start = next((i for i, line in enumerate(lines) if pattern.match(line.rstrip("\r\n"))), None)
    if start is None:
        raise ValueError(f"'{block_key}:' block not found")

    last, indent = start, "  "
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


def _apply_roots(lines: list[str], roots: Dict[str, str], block_key: str, eol: str) -> tuple[list[str], bool]:
    """
    lines 복사본에 roots를 덮어쓰기/추가한다.

    Returns:
        (new_lines, changed)
    """
    lines = list(lines)
    start, last, indent = _find_block(lines, block_key)
    changed = False

    for name, page_id in roots.items():
        name, page_id = str(name).strip(), str(page_id).strip()
        # 들여쓰기 + key + ":" + 값(따옴표 유무 상관없음) + 나머지(공백/주석)
        key_re = re.compile(
            rf"^(?P<indent>[ \t]+){re.escape(name)}(?P<sep>\s*:\s*)"
            rf"(?P<value>\"[^\"]*\"|'[^']*'|[^\s#]*)(?P<rest>.*)$"
        )

        hit = next(
            ((i, m) for i in range(start + 1, last + 1)
             if (m := key_re.match(lines[i].rstrip("\r\n")))),
            None,
        )

        if hit:
            i, m = hit
            old_value = m.group("value").strip("\"'")
            if old_value == page_id:
                logger.info("[CONFIG ROOTS] unchanged | %s=%s", name, page_id)
                continue
            line_eol = lines[i][len(lines[i].rstrip("\r\n")):]
            lines[i] = f'{m.group("indent")}{name}{m.group("sep")}"{page_id}"{m.group("rest")}{line_eol}'
            logger.info("[CONFIG ROOTS] overwrite | %s: %s -> %s", name, old_value, page_id)
        else:
            if not lines[last].endswith(("\n", "\r")):    # 파일 끝에 줄바꿈이 없는 경우 대비
                lines[last] += eol
            last += 1
            lines.insert(last, f'{indent}{name}: "{page_id}"{eol}')
            logger.info("[CONFIG ROOTS] append | %s=%s", name, page_id)
        changed = True

    return lines, changed


def _verify_roots(text: str, roots: Dict[str, str], block_key: str) -> None:
    """바뀐 yaml 텍스트를 실제로 파싱해서 roots가 그대로 들어갔는지 확인한다."""
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"yaml parse failed after update: {e}") from e

    block = parsed.get(block_key) or {}
    wrong = {
        str(k): block.get(str(k))
        for k, v in roots.items()
        if str(block.get(str(k))) != str(v).strip()
    }
    if wrong:
        raise ValueError(f"yaml verify failed | {block_key} mismatch: {wrong}")


# =========================================================
# 파일 helper
# =========================================================
def _backup_file(path: str, backup_dir: str) -> str:
    """원본을 backup_dir/v_old/v_old_{파일명}_{YYYYMMDD_HHMMSS}{확장자} 로 복사한다."""
    v_old_dir = os.path.join(backup_dir, "v_old")
    os.makedirs(v_old_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(v_old_dir, f"v_old_{stem}_{timestamp}{ext}")
    shutil.copy2(path, backup_path)
    return backup_path


def _atomic_write(path: str, text: str) -> None:
    """같은 폴더의 임시 파일에 쓴 뒤 os.replace로 바꿔친다. (줄바꿈 문자는 text 그대로)"""
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)),
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# =========================================================
# public
# =========================================================
def update_bootstrap_roots(
    yaml_path: str,
    roots: Dict[str, str],
    backup_dir: str = "./pipeline_storage/config_gppm",
    block_key: str = "bootstrap_roots",
) -> None:
    """
    yaml의 bootstrap_roots 블록에 {법인명: page_id}를 덮어쓰기/추가한다.
    yaml 파일의 나머지 줄(구조, 띄어쓰기, 주석, 줄바꿈 CRLF/LF)은 그대로 유지한다.

    동작:
        1. yaml을 줄 단위로 읽는다. (줄바꿈 문자 보존)
        2. bootstrap_roots 블록에서 같은 key는 값만 바꾸고, 없는 key는 블록 끝에 추가한다.
        3. 바뀐 게 없으면 백업/저장 없이 끝낸다.
        4. 바뀐 텍스트를 yaml로 파싱해 roots가 제대로 들어갔는지 검증한다.
        5. 원본을 backup_dir/v_old/ 에 백업하고, 임시 파일 → os.replace로 원본을 바꿔친다.

    Args:
        yaml_path: 수정할 config yaml 경로 (예: config/params_config_dev_gppm.yaml).
        roots: 붙일 {법인명: page_id}.
        backup_dir: 백업 폴더. 그 아래 v_old/ 에 저장한다.
        block_key: 대상 블록 이름.

    """
    if not roots:
        logger.info("[CONFIG ROOTS] empty roots, skip")
        return

    try:
        with open(yaml_path, encoding="utf-8", newline="") as f:
            lines = f.readlines()
        eol = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"

        new_lines, changed = _apply_roots(lines, roots, block_key, eol)
        if not changed:
            logger.info("[CONFIG ROOTS] nothing to change | file=%s", yaml_path)
            return

        new_text = "".join(new_lines)
        _verify_roots(new_text, roots, block_key)

        backup_path = _backup_file(yaml_path, backup_dir)
        logger.info("[CONFIG ROOTS] backup | %s", backup_path)

        _atomic_write(yaml_path, new_text)

    except RuntimeError as e:
        logger.error("[CONFIG ROOTS][FAIL] %s | file=%s", e, yaml_path)
        raise
    except Exception as e:  # OSError, UnicodeDecodeError 등도 ValueError로 통일
        logger.error("[CONFIG ROOTS][FAIL] %s: %s | file=%s", type(e).__name__, e, yaml_path)
        raise RuntimeError(f"update_bootstrap_roots failed | file={yaml_path} | {type(e).__name__}: {e}") from e

    logger.info("[CONFIG ROOTS][SUCCESS] updated | file=%s", yaml_path)
