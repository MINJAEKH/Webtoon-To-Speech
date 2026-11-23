# ===== src/pipeline/subsidiary_registry.py 맨 아래(클래스 밖)에 추가 =====
from src.components.subsidiary.subsidiary_models import now_str


def mark_subsidiaries_result(
    subsidiary_path: str,
    subsidiary_list: list[str],
    success: bool,
    filename_prefix: str = "subsidiary_master",
) -> None:
    """
    pipeline 실행 결과를 법인 master json(v_new)에 반영한다.

    대상: 이번 실행에 입력된 법인(subsidiary_list)만.
        - 모든 입력 법인(deleted 제외): last_updated_at = now_str()
        - status가 pending인 법인만:   성공 → done / 실패 → failed
          (이미 done/failed/deleted인 법인의 status는 그대로 둔다)

    Args:
        subsidiary_path: 법인 master json 폴더 (config.subsidiary_path).
        subsidiary_list: 이번 실행의 --subsidiary-list.
        success: pipeline이 예외 없이 끝났으면 True.
        filename_prefix: master json 파일 prefix.
    """
    store = load_subsidiaries(subsidiary_path, filename_prefix)
    if len(store) == 0:
        logger.warning("[SUBSIDIARY RESULT] no master json, skip | dir=%s", subsidiary_path)
        return

    new_status = "done" if success else "failed"
    now = now_str()
    changed, skipped = [], []

    for name in subsidiary_list:
        state = store.get(name)
        if state is None or state.status == "deleted":
            skipped.append(name)
            continue
        state.last_updated_at = now
        if state.status == "pending":
            state.status = new_status
        changed.append(name)

    if not changed:
        logger.info("[SUBSIDIARY RESULT] nothing to update | skipped=%s", skipped)
        return

    save_subsidiaries(
        subsidiary_path,
        store,
        metadata={
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root_source": f"pipeline_{new_status}",
        },
        filename_prefix=filename_prefix,
    )
    logger.info(
        "[SUBSIDIARY RESULT] %s | updated=%s | skipped=%s | %s",
        new_status, changed, skipped, store.status_counts(),
    )

# ===== main.py import 추가 (34행 근처) =====
from src.pipeline.subsidiary_registry import SubsidiaryRegistry, mark_subsidiaries_result


# ===== main.py run() 전체 =====
def run(config, subsidiary_list):
    start_time = time.time()
    logger = logging.getLogger(__name__)
    stage = config.type

    # 법인 결과 기록 대상: Local 실행(입력 법인 있음 + registry 사용)이고 실제 pipeline을 탄 경우만
    track_subsidiary = bool(subsidiary_list) and config.get("run_subsidiary_registry", False)
    pipeline_started = False

    try:
        logger.info(f"[Airflow {stage}][START] process started")

        validate_directories(config)
        print_update_steps(config)

        if stage in ("TEST", "TEST_GPPM"):
            if not test_connection_all(config):
                raise RuntimeError("TEST connection check failed")
        else:
            pipeline_started = True
            pipeline(config, subsidiary_list)
            logger.info("The New Confluence pages are Successfully Saved!")

            # [추가] 성공: pending -> done, last_updated_at = now
            if track_subsidiary:
                mark_subsidiaries_result(config.subsidiary_path, subsidiary_list, success=True)

        all_file_managing(config)
        log_file_managing(stage)

        elapsed_min, elapsed_sec = divmod(time.time() - start_time, 60)
        logger.info(
            f"[{stage}] {config.collection_name} executed in "
            f"{int(elapsed_min)} min {elapsed_sec:.2f} sec."
        )
        logger.info("=" * 80)

        logger.info(f"[Airflow {stage}][SUCCESS] process completed")

    except Exception as e:
        logger.exception(f"[Airflow {stage}][FAIL] process failed: {e}")

        # [추가] 실패: pending -> failed, last_updated_at = now
        # 기록 중 에러가 나도 원래 에러(e)를 가리지 않도록 따로 감싼다.
        if track_subsidiary and pipeline_started:
            try:
                mark_subsidiaries_result(config.subsidiary_path, subsidiary_list, success=False)
            except Exception as mark_e:
                logger.error("[SUBSIDIARY RESULT][FAIL] could not mark failed: %s", mark_e)
        raise
