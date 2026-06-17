# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.experimental.cli.agent import http as http_mod
from areal.experimental.cli.agent import session_ops
from areal.experimental.cli.agent.config import (
    load_config,
    resolve_admin_api_key,
    resolve_inf_addr,
)
from areal.experimental.cli.agent.session_ops import create_session
from areal.experimental.cli.agent.state import (
    ProcessState,
    ServiceState,
    SessionsState,
    SessionState,
    service_state_path,
)


def test_service_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    state = ServiceState(
        service="svc",
        launch_mode="detached",
        agent="pkg.Agent",
        admin_api_key="key",
        gateway=ProcessState(
            component="gateway",
            pid=11,
            url="http://127.0.0.1:1",
            log_file="gateway.log",
        ),
        router=ProcessState(
            component="router",
            pid=12,
            url="http://127.0.0.1:2",
            log_file="router.log",
        ),
        pairs=[],
    )

    state.save()

    loaded = ServiceState.load("svc")
    assert loaded.service == "svc"
    assert loaded.gateway.pid == 11
    assert loaded.all_pids() == [11, 12]
    assert service_state_path("svc").exists()


def test_sessions_state_tracks_current_session(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    sessions = SessionsState(service="svc")
    session = SessionState.create(key="s1", session_timeout=60.0)
    sessions.sessions[session.key] = session
    sessions.current_session = session.key

    sessions.save()

    loaded = SessionsState.load("svc")
    assert loaded.current_session == "s1"
    assert loaded.require_active("s1").key == "s1"
    assert "s1" in loaded.active_sessions()


def test_config_merges_user_and_extra(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path / "home"))
    config_dir = tmp_path / "home" / "agent"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        "[default]\nadmin_api_key = 'base'\n[inference]\naddr = 'http://base'\n"
    )
    extra = tmp_path / "extra.toml"
    extra.write_text("[default]\nadmin_api_key = 'override'\n")

    config = load_config(extra)

    assert resolve_admin_api_key(config, None) == "override"
    assert resolve_inf_addr(config, None) == "http://base"


def test_inference_start_session_uses_current_schema(monkeypatch):
    captured = {}

    def fake_request(url, *, method="GET", payload=None, bearer=None, timeout=5.0):
        captured.update(
            url=url,
            method=method,
            payload=payload,
            bearer=bearer,
            timeout=timeout,
        )
        return {"group_id": "grp", "sessions": []}

    monkeypatch.setattr(http_mod, "_request", fake_request)

    http_mod.InferenceClient("http://inf", "admin").start_session(task_id="task-1")

    assert captured["url"] == "http://inf/rl/start_session"
    assert captured["method"] == "POST"
    assert captured["payload"] == {"task_id": "task-1", "group_size": 1}
    assert captured["bearer"] == "admin"


def test_create_session_negotiates_rl_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    routed = []
    started = []

    class FakeRouterClient:
        def __init__(self, base_url, admin_api_key):
            self.base_url = base_url
            self.admin_api_key = admin_api_key

        def route(self, session_key):
            routed.append(session_key)

    class FakeInferenceClient:
        def __init__(self, base_url, admin_api_key):
            self.base_url = base_url
            self.admin_api_key = admin_api_key

        def start_session(self, *, task_id):
            started.append(task_id)
            return {
                "group_id": "grp",
                "sessions": [{"session_id": "sid-1", "session_api_key": "sess-key"}],
            }

    monkeypatch.setattr(session_ops, "AgentRouterClient", FakeRouterClient)
    monkeypatch.setattr(session_ops, "InferenceClient", FakeInferenceClient)
    state = ServiceState(
        service="svc",
        launch_mode="detached",
        agent="pkg.Agent",
        admin_api_key="agent-key",
        gateway=ProcessState(
            component="gateway",
            pid=11,
            url="http://127.0.0.1:1",
            log_file="gateway.log",
        ),
        router=ProcessState(
            component="router",
            pid=12,
            url="http://127.0.0.1:2",
            log_file="router.log",
        ),
        pairs=[],
        inf_addr="http://inf",
        inf_api_key="admin",
        inf_model="",
    )

    session = create_session(
        state,
        SessionsState(service="svc"),
        session_key="agent-session",
    )

    assert routed == ["agent-session"]
    assert started == ["agent-svc-agent-session"]
    assert session.rl_negotiated
    assert session.rl_session_id == "sid-1"
    assert session.rl_session_api_key == "sess-key"
