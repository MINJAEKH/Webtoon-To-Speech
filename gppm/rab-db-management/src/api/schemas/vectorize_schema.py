"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : vectorize 실행 및 상태 조회 API의 응답 스키마
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *    2       2026.11..18  minjae.cha    Local Bootstrap Roots 입력 
 *********** ************ ************ ***************************************************** ************* *************
"""
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator

CorpName = Annotated[str, Field(pattern = r"^[A-Z0-9_]{2,50}$")]

# 기존 RootPageId 정의는 더 이상 안 쓰면 삭제
class VectorizeRunRequest(BaseModel):
    '''
    - local_corps가 비어 있으면([], null, 키 없음) 기존 파이프라인 실행
    - 값이 있으면 Local 파이프라인 실행
    '''
    local_corps: list[CorpName] = Field(default_factory=list)

    @field_validator("local_corps", mode="before")
    @classmethod
    def none_to_empty(cls, v):
        return [] if v is None else v

    # @field_validator("local_corps")
    # @classmethod
    # def dedupe(cls, v):
    #     return list(dict.fromkeys(v))   # 순서 유지 + 중복 제거

class VectorizeRunResponse(BaseModel):
    start_date: str
    local_corps: list[str] = []

class VectorizeStatusResponse(BaseModel):
    status: Literal["running", "done", "failed"]
    log_file: Optional[str] = None