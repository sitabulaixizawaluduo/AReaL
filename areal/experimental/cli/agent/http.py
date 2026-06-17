# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class AgentCLIHTTPError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class AgentCLIUnreachable(Exception):
    pass


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    bearer: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise AgentCLIHTTPError(exc.code, body) from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        raise AgentCLIUnreachable(str(exc)) from exc


class AgentGatewayClient:
    def __init__(self, base_url: str, admin_api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.admin_api_key = admin_api_key

    def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        return _request(f"{self.base_url}/health", timeout=timeout)


class AgentRouterClient:
    def __init__(self, base_url: str, admin_api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.admin_api_key = admin_api_key

    def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        return _request(f"{self.base_url}/health", timeout=timeout)

    def register_proxy(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/register",
            method="POST",
            payload={"addr": addr},
            bearer=self.admin_api_key,
            timeout=timeout,
        )

    def route(self, session_key: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/route",
            method="POST",
            payload={"session_key": session_key},
            bearer=self.admin_api_key,
            timeout=timeout,
        )

    def remove_session(
        self, session_key: str, *, timeout: float = 10.0
    ) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/remove_session",
            method="POST",
            payload={"session_key": session_key},
            bearer=self.admin_api_key,
            timeout=timeout,
        )


class DataProxyClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        return _request(f"{self.base_url}/health", timeout=timeout)

    def close_session(
        self, session_key: str, *, timeout: float = 10.0
    ) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/session/{session_key}/close",
            method="POST",
            payload={},
            timeout=timeout,
        )


class InferenceClient:
    def __init__(self, base_url: str, admin_api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.admin_api_key = admin_api_key

    def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        return _request(f"{self.base_url}/health", timeout=timeout)

    def start_session(
        self,
        *,
        task_id: str,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        payload = {
            "task_id": task_id,
            "group_size": 1,
        }
        return _request(
            f"{self.base_url}/rl/start_session",
            method="POST",
            payload=payload,
            bearer=self.admin_api_key,
            timeout=timeout,
        )
