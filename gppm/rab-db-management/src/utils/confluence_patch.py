"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : atlassian Confluence 클라이언트 요청에 재시도와 쿨다운을 주입하는 monkey patch
 *               429 응답이 발생했을 때만 전체 요청을 지연시키는 적응형 쿨다운을 적용한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

import logging
import time
import threading
from json import dumps
from urllib.parse import urlencode

from atlassian import Confluence
from src.utils.utils import RetryOnException

logger = logging.getLogger(__name__)

class AdaptiveCooldown:
    """
    평소엔 대기 없음.
    429가 발생했을 때만 일정 시간 전체 요청을 늦춤.
    """
    def __init__(self):
        self.cooldown_until = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self) -> float:
        with self.lock:
            now = time.time()
            if now >= self.cooldown_until:
                return 0.0
            sleep_time = self.cooldown_until - now

        if sleep_time > 0:
            logger.warning("[AdaptiveCooldown] cooling down for %.2fs", sleep_time)
            time.sleep(sleep_time)
        return sleep_time

    def set_cooldown(self, seconds: float):
        if seconds <= 0:
            return
        with self.lock:
            new_until = time.time() + seconds
            if new_until > self.cooldown_until:
                self.cooldown_until = new_until
                logger.warning(
                    "[AdaptiveCooldown] cooldown set to %.2fs (until=%.2f)",
                    seconds,
                    self.cooldown_until,
                )

cooldown = AdaptiveCooldown()

@RetryOnException()
def _patched_request(
    self,
    method: str = "GET",
    path: str = "/",
    data=None,
    json=None,
    flags=None,
    params=None,
    headers=None,
    files=None,
    trailing=None,
    absolute: bool = False,
    advanced_mode: bool = False,
):
    cooldown_wait = cooldown.wait_if_needed()

    url = self.url_joiner(None if absolute else self.url, path, trailing)
    params_already_in_url = "?" in url

    if params or flags:
        url += "&" if params_already_in_url else "?"

    if params:
        url += urlencode((params or {}), safe=",")

    if flags:
        url += ("&" if params or params_already_in_url else "") + "&".join(flags or [])

    json_dump = None
    if files is None:
        data = None if not data else dumps(data)
        json_dump = None if not json else dumps(json)

    headers = headers or self.default_headers

    self.log_curl_debug(
        method=method,
        url=url,
        headers=headers,
        data=data or json_dump,
    )

    request_start = time.time()
    response = self._session.request(
        method=method,
        url=url,
        headers=headers,
        data=data,
        json=json,
        timeout=self.timeout,
        verify=self.verify_ssl,
        files=files,
        proxies=self.proxies,
        cert=self.cert,
    )
    request_elapsed = time.time() - request_start
    response.encoding = "utf-8"

    remaining = response.headers.get("X-RateLimit-Remaining")
    limit = response.headers.get("X-RateLimit-Limit")
    interval = response.headers.get("X-RateLimit-Interval-Seconds")
    retry_after = response.headers.get("Retry-After")

    logger.info(
        "[ConfluenceRequest] %s %s | cooldown_wait=%.2fs | request_elapsed=%.2fs | "
        "status=%s | remaining=%s | limit=%s | interval=%s | retry_after=%s",
        method,
        path,
        cooldown_wait,
        request_elapsed,
        response.status_code,
        remaining,
        limit,
        interval,
        retry_after,
    )

    if response.status_code == 429:
        cooldown.set_cooldown(3.0)
        logger.warning(
            "[ConfluenceRequest][429] cooldown=3.0s | remaining=%s | limit=%s | interval=%s | retry_after=%s",
            remaining,
            limit,
            interval,
            retry_after,
        )
        response.raise_for_status()

    logger.debug("HTTP: %s %s -> %s %s", method, path, response.status_code, response.reason)
    logger.debug("HTTP: Response text -> %s", response.text)
    print("HTTP: %s %s -> %s %s", method, path, response.status_code, response.reason)

    if self.advanced_mode or advanced_mode:
        return response

    self.raise_for_status(response)
    return response

def patch_confluence_request():
    Confluence.request = _patched_request
    _PATCHED = True
    logger.info("Confluence.request patched successfully.")
    
@RetryOnException()
def initialize_confluence_api(page_root, confluence_token):
    """
    Confluence API client를 초기화한다.

    네트워크/API 연결 실패가 발생할 수 있으므로 RetryOnException을 적용한다.
    """
    confluence_api = Confluence(
        url=page_root,
        token=confluence_token,
    )
    logger.info("Confluence API connection successful")
    return confluence_api