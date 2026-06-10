# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from areal.experimental.cli.state import pid_alive


def pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _spawn(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "ab", buffering=0)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    return proc.pid


def spawn_router(
    *,
    host: str,
    admin_api_key: str,
    routing_strategy: str,
    log_level: str,
    log_file: Path,
) -> tuple[int, int]:
    port = pick_free_port()
    cmd = [
        sys.executable, "-m", "areal.experimental.inference_service.router",
        "--host", host,
        "--port", str(port),
        "--admin-api-key", admin_api_key,
        "--routing-strategy", routing_strategy,
        "--log-level", log_level,
    ]
    return _spawn(cmd, log_file), port


def spawn_gateway(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    router_url: str,
    log_level: str,
    log_file: Path,
) -> int:
    cmd = [
        sys.executable, "-m", "areal.experimental.inference_service.gateway",
        "--host", host,
        "--port", str(port),
        "--admin-api-key", admin_api_key,
        "--router-addr", router_url,
        "--log-level", log_level,
    ]
    return _spawn(cmd, log_file)


def spawn_sglang(
    *,
    model_path: str,
    host: str,
    port: int,
    tp: int,
    base_gpu_id: int,
    extra_args: list[str],
    log_file: Path,
) -> int:
    from areal.api.cli_args import SGLangConfig

    cfg = SGLangConfig(model_path=model_path)
    cmd = list(SGLangConfig.build_cmd(
        sglang_config=cfg,
        tp_size=tp,
        base_gpu_id=base_gpu_id,
        host=host,
        port=port,
        n_nodes=1,
        node_rank=0,
        pp_size=1,
    ))
    cmd.extend(extra_args)
    return _spawn(cmd, log_file)


def spawn_vllm(
    *,
    model_path: str,
    host: str,
    port: int,
    tp: int,
    pp: int,
    extra_args: list[str],
    log_file: Path,
) -> int:
    from areal.api.cli_args import vLLMConfig

    cfg = vLLMConfig(model=model_path)
    cmd = list(vLLMConfig.build_cmd(
        vllm_config=cfg,
        tp_size=tp,
        pp_size=pp,
        host=host,
        port=port,
    ))
    cmd.extend(extra_args)
    return _spawn(cmd, log_file)


def spawn_data_proxy(
    *,
    host: str,
    port: int,
    backend_addr: str,
    backend_type: str,
    tokenizer_path: str,
    admin_api_key: str,
    log_level: str,
    extra_args: list[str],
    log_file: Path,
) -> int:
    cmd = [
        sys.executable, "-m", "areal.experimental.inference_service.data_proxy",
        "--host", host,
        "--port", str(port),
        "--backend-addr", backend_addr,
        "--backend-type", backend_type,
        "--tokenizer-path", tokenizer_path,
        "--admin-api-key", admin_api_key,
        "--log-level", log_level,
    ]
    cmd.extend(extra_args)
    return _spawn(cmd, log_file)


def signal_pid(pid: int, sig: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.killpg(os.getpgid(pid), sig)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        pass
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def kill_pids(pids: list[int], grace_s: float) -> None:
    pids = [p for p in pids if p > 0]
    if not pids:
        return
    for p in pids:
        if pid_alive(p):
            signal_pid(p, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not any(pid_alive(p) for p in pids):
            return
        time.sleep(0.2)
    for p in pids:
        if pid_alive(p):
            signal_pid(p, signal.SIGKILL)
