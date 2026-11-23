"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : GPPM TEST/DEV/PROD 파이프라인 실행 및 상태 조회 API 라우터
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""
from typing import Optional

from fastapi import APIRouter, Query

from src.api.schemas.vectorize_schema import (
    VectorizeRunRequest,
    VectorizeRunResponse,
    VectorizeStatusResponse,
)
from src.api.services.vectorize_service import (
    run_vectorize_process,
    get_vectorize_status,
)

router = APIRouter(
    prefix="/sync",
    tags=["vectorize_gppm"],
)

@router.post(
    "/test_gppm/run",
    response_model=VectorizeRunResponse,
    summary="GPPM TEST Pipeline 실행",
)
def run_test(body: Optional[VectorizeRunRequest]=None):
    return run_vectorize_process("test_gppm", body)

@router.get(
    "/test_gppm/status",
    response_model=VectorizeStatusResponse,
    summary="GPPM TEST Pipeline 상태 조회",
)
def get_test_status(
    start_date: str = Query(
        ...,
        description="run API에서 반환된 start_date",
    ),
):
    return get_vectorize_status(
        environment="test_gppm",
        start_date=start_date,
    )

@router.post(
    "/dev_gppm/run",
    response_model=VectorizeRunResponse,
    summary="GPPM DEV Pipeline 실행",
)
def run_dev(body: Optional[VectorizeRunRequest]=None):
    return run_vectorize_process("dev_gppm", body)

@router.get(
    "/dev_gppm/status",
    response_model=VectorizeStatusResponse,
    summary="GPPM DEV Pipeline 상태 조회",
)
def get_dev_status(
    start_date: str = Query(
        ...,
        description="run API에서 반환된 start_date",
    ),
):
    return get_vectorize_status(
        environment="dev_gppm",
        start_date=start_date,
    )

@router.post(
    "/prod_gppm/run",
    response_model=VectorizeRunResponse,
    summary="GPPM PROD Pipeline 실행",
)
def run_prod(body: Optional[VectorizeRunRequest]=None):
    return run_vectorize_process("prod_gppm", body)

@router.get(
    "/prod_gppm/status",
    response_model=VectorizeStatusResponse,
    summary="GPPM PROD Pipeline 상태 조회",
)
def get_prod_status(
    start_date: str = Query(
        ...,
        description="run API에서 반환된 start_date",
    ),
):
    return get_vectorize_status(
        environment="prod_gppm",
        start_date=start_date,
    )