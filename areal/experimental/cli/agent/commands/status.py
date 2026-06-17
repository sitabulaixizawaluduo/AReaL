# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time

from areal.experimental.cli.agent.http import (
    AgentCLIHTTPError,
    AgentCLIUnreachable,
    AgentGatewayClient,
    AgentRouterClient,
    DataProxyClient,
)
from areal.experimental.cli.agent.process import pid_alive
from areal.experimental.cli.agent.state import (
    ServiceState,
    SessionsState,
    resolve_service_name,
    service_state_path,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("status", help="Show agent service health")
    parser.add_argument("--service", default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.set_defaults(handler=handle)

    health = subparsers.add_parser("health", help="Alias for status")
    health.add_argument("--service", default=None)
    health.add_argument("--watch", action="store_true")
    health.add_argument("--interval", type=float, default=2.0)
    health.add_argument("--json", action="store_true", dest="as_json")
    health.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    service = resolve_service_name(args.service)
    return do_status(
        service=service,
        as_json=args.as_json,
        watch=args.watch,
        interval=args.interval,
    )


def do_status(*, service: str, as_json: bool, watch: bool, interval: float) -> int:
    while True:
        snapshot = _snapshot(service)
        if as_json:
            print(json.dumps(snapshot, indent=2))
        else:
            _print_table(snapshot)
        if not watch:
            return 0
        time.sleep(interval)


def _snapshot(service: str) -> dict:
    if not service_state_path(service).exists():
        return {"service": service, "running": False, "components": [], "sessions": []}
    try:
        state = ServiceState.load(service)
    except Exception as exc:
        return {
            "service": service,
            "running": False,
            "error": f"failed to read state: {exc}",
            "components": [],
            "sessions": [],
        }

    components = [
        _component_health(
            service,
            "gateway",
            state.gateway.url,
            state.gateway.pid,
            lambda: AgentGatewayClient(state.gateway.url, state.admin_api_key).health(),
        ),
        _component_health(
            service,
            "router",
            state.router.url,
            state.router.pid,
            lambda: AgentRouterClient(state.router.url, state.admin_api_key).health(),
        ),
    ]
    for pair in state.pairs:
        components.append(
            _component_health(
                service,
                pair.worker.component,
                pair.worker.url,
                pair.worker.pid,
                lambda url=pair.worker.url: DataProxyClient(url).health(),
            )
        )
        components.append(
            _component_health(
                service,
                pair.data_proxy.component,
                pair.data_proxy.url,
                pair.data_proxy.pid,
                lambda url=pair.data_proxy.url: DataProxyClient(url).health(),
            )
        )

    sessions_state = SessionsState.load(service)
    sessions = [
        {
            "key": session.key,
            "status": session.status,
            "current": session.key == sessions_state.current_session,
            "rl_negotiated": session.rl_negotiated,
        }
        for session in sessions_state.sessions.values()
    ]
    return {
        "service": service,
        "running": any(component["pid_alive"] for component in components),
        "gateway_url": state.gateway.url,
        "router_url": state.router.url,
        "components": components,
        "sessions": sessions,
    }


def _component_health(service: str, component: str, url: str, pid: int, fn) -> dict:
    http_status = "down"
    detail = ""
    try:
        data = fn()
        http_status = "ok"
        detail = json.dumps(data, sort_keys=True)
    except AgentCLIHTTPError as exc:
        http_status = f"http-{exc.status}"
        detail = exc.body
    except AgentCLIUnreachable as exc:
        detail = str(exc)
    return {
        "service": service,
        "component": component,
        "status": http_status,
        "addr": url,
        "pid": pid,
        "pid_alive": pid_alive(pid),
        "details": detail,
    }


def _print_table(snapshot: dict) -> None:
    components = snapshot.get("components") or []
    if not components:
        print(f"service {snapshot['service']!r} is not running")
        return
    rows = [
        (
            row["service"],
            row["component"],
            row["status"],
            row["addr"],
            row["details"],
        )
        for row in components
    ]
    cols = ("SERVICE", "COMPONENT", "STATUS", "ADDR", "DETAILS")
    widths = [max(len(str(row[i])) for row in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*cols))
    for row in rows:
        print(fmt.format(*row))
