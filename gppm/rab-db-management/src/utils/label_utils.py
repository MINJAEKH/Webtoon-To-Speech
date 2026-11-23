"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : Confluence label 파싱 및 정규화 유틸
 *               다양한 형태의 raw label을 region/biz/corp/extra로 구조화하고
 *               결측 토큰 제거와 중복 제거를 거쳐 Milvus 입력 형식으로 변환한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *    2       2026.11.02   soyeone.park             NERP&SEGAM 제거
 *********** ************ ************ ***************************************************** ************* *************
"""

import json
from typing import Any
import ast

LABEL_KEYS = ("region", "biz", "corp", "extra")

# 결측치가 문자열로 변환되어 들어온 경우 걸러낼 토큰들
# (예: np.nan → "nan", pd.NA → "<na>", None 문자열 → "none")
# 주의: "none"이라는 실제 라벨을 쓰는 경우가 있다면 이 집합에서 제거할 것.
_INVALID_TOKENS = {"nan", "<na>", "none", "null", "[]"}

def empty_labels(include_structured: bool = True) -> dict[str, list[str]]:
    if include_structured:
        return {
            "region": [],
            "biz": [],
            "corp": [],
            "extra": [],
        }

    return {
        "extra": [],
    }

def _dedup_sorted(values: list[str]) -> list[str]:
    """
    중복 제거 후 정렬한다.
    label 결과를 deterministic하게 유지하기 위한 처리.
    """
    return sorted(set(values))

def normalize_label_values(value: Any) -> list[str]:
    """
    어떤 형태의 값이 들어와도 깨끗한 문자열 label list로 정규화한다.

    처리 가능한 입력:
        - None / NaN / pd.NA          → []
        - np.ndarray                  → list로 변환 후 처리
        - 단일 값 ("kr")              → ["kr"]
        - list / tuple / set          → 원소별 처리
        - 중첩 list ([["kr"]])        → 재귀로 평탄화
        - 오염 문자열 ("['kr']")      → ast.literal_eval로 복구
        - 이중 오염 ("[\"['kr']\"]")  → 재귀로 끝까지 복구 (멱등 보장)

    반환값은 항상 정렬 + 중복 제거된 list[str].
    """
    if value is None:
        return []

    # np.ndarray → list
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (tuple, set)):
        value = list(value)

    if not isinstance(value, list):
        value = [value]

    result: list[str] = []

    for x in value:
        if x is None:
            continue

        # 중첩된 list/tuple/set/ndarray → 재귀로 평탄화
        if hasattr(x, "tolist") or isinstance(x, (list, tuple, set)):
            result.extend(normalize_label_values(x))
            continue

        text = str(x).strip().lower()

        if not text or text in _INVALID_TOKENS:
            continue

        # "['kr']" / '["kr"]' 형태 오염 문자열 복구
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    # 재귀 처리라 몇 겹으로 오염되어 있어도 끝까지 복구된다
                    result.extend(normalize_label_values(list(parsed)))
                    continue
            except Exception:
                # 파싱 실패 시 원본 텍스트를 그대로 유지
                pass

        result.append(text)

    return _dedup_sorted(result)

def parse_confluence_labels(raw_labels: Any) -> dict[str, list[str]]:
    """
    Confluence label list를 Milvus JSON labels schema로 변환한다.

    입력 예:
        ["region-kr", "region-jp", "biz-sales", "corp-abc", "team-x"]

    출력 예:
        {
            "region": ["jp", "kr"],
            "biz": ["sales"],
            "corp": ["abc"],
            "extra": ["team-x"]
        }

    규칙:
        - labels는 절대 None이 될 수 없다.
        - label이 없어도 region/biz/corp/extra key는 반드시 존재한다.
        - region-, biz-, corp- prefix는 제거한다.
        - 그 외 label은 extra에 넣는다.
        - 이미 dict 형태면 key 누락만 보정한다.
        - JSON string이면 파싱 후 다시 변환한다.
        - ndarray/중첩 list/오염 문자열이 들어와도 복구한다 (멱등 보장).
    """
    result = empty_labels()

    if raw_labels is None:
        return result

    # 이미 표준 dict 형태이거나 유사 dict 형태인 경우
    # normalize_label_values가 None/ndarray/단일값/중첩/오염을 전부 처리한다.
    if isinstance(raw_labels, dict):
        for key in LABEL_KEYS:
            result[key] = normalize_label_values(raw_labels.get(key, []))

        return result

    # parquet/json 과정에서 문자열로 들어온 경우 방어
    if isinstance(raw_labels, str):
        stripped = raw_labels.strip()

        if not stripped:
            return result

        try:
            parsed = json.loads(stripped)
            return parse_confluence_labels(parsed)
        except Exception:
            raw_labels = [stripped]

    # list/tuple/set/np.ndarray/단일값 등 나머지 케이스:
    # 먼저 평탄한 문자열 list로 정규화한 뒤 prefix로 분류한다.
    # (중첩 list [["region-kr"]]도 여기서 풀려서 정상 분류된다)
    for label in normalize_label_values(raw_labels):
        if label.startswith("region-"):
            rest = label.removeprefix("region-")
            if rest:  # "region-" 단독 라벨로 빈 문자열이 들어가는 것 방지
                result["region"].append(rest)
        elif label.startswith("biz-"):
            rest = label.removeprefix("biz-")
            if rest:
                result["biz"].append(rest)
        elif label.startswith("corp-"):
            rest = label.removeprefix("corp-")
            if rest:
                result["corp"].append(rest)
        else:
            result["extra"].append(label)

    for key in LABEL_KEYS:
        result[key] = _dedup_sorted(result[key])

    return result

def normalize_labels_for_milvus(raw_labels: Any) -> dict[str, list[str]]:
    parsed = parse_confluence_labels(raw_labels)

    extra = []
    for key in ("region", "biz", "corp", "extra"):
        extra.extend(parsed.get(key, []))

    return {
        "extra": sorted(set(extra))
    }