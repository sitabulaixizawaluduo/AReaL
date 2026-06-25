# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from areal.experimental.cli.client import BaseHTTPClient


class GatewayClient(BaseHTTPClient):
    """Gateway exposes only ``/health`` to the CLI today; everything else
    is reserved for end-user agent requests. Inherit from the scaffold
    base and add gateway-specific RPCs here when they appear."""


class RouterClient(BaseHTTPClient):
    def register_proxy(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._post("/register", payload={"addr": addr}, timeout=timeout)


class DataProxyClient(BaseHTTPClient):
    """Data-proxy exposes only ``/health`` to the CLI; admin paths live
    on the router. Kept as a distinct subclass so component-handling
    code can branch on type if it ever needs to."""
