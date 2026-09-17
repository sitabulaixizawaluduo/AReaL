"""Opt-in capture subclass; the original bridge owns timing and game events."""

import base64
import hashlib
import json
import os
import random
import runpy
import sys
import zlib
from pathlib import Path

import numpy as np
from maapacman.env._pygame_worker import _parse_args, _PygameBridge
from PIL import Image


class RecordingBridge(_PygameBridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = Path(os.environ["PACMAN_RECORDING_ROOT"])
        self.root.mkdir(parents=True, exist_ok=False)
        self.index = (self.root / "frames.jsonl").open("x", buffering=1)
        self.previous = None
        self.frame_count = 0

    def _capture(self, globals_dict):
        payload = super()._capture(globals_dict)
        state, image = payload["state"], payload["frame"]
        frame_id = int(state["logic_frame"])
        raw = zlib.decompress(base64.b64decode(image["data"]))
        digest = hashlib.sha256(raw).hexdigest()
        # Only an explicitly paused re-capture may be omitted. Rendered death
        # animations/static movement frames are retained even at the same ID/RGB.
        duplicate_wait = globals_dict.get("agentPaused") and self.previous == (
            frame_id,
            digest,
        )
        if not duplicate_wait:
            if self.previous is not None and frame_id not in (
                self.previous[0],
                self.previous[0] + 1,
            ):
                raise RuntimeError("Recording missed a simulation tick")
            filename = f"frame-{self.frame_count:06d}.png"
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(image["shape"])
            Image.fromarray(frame).save(self.root / filename, compress_level=1)
            self.index.write(
                json.dumps(
                    {
                        "index": self.frame_count,
                        "file": filename,
                        "logic_frame": frame_id,
                        "rgb_sha256": digest,
                        **{
                            k: state[k]
                            for k in ("score", "lives", "mode", "normal_pellets")
                        },
                    }
                )
                + "\n"
            )
            self.frame_count += 1
            self.previous = (frame_id, digest)
        return payload


def main():
    args = _parse_args()
    script = Path(args.script).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    output = sys.stdout
    sys.stdout = sys.stderr
    sys.path[0] = str(script.parent)
    sys.argv = [
        str(script),
        "--start-level",
        "1",
        "--curriculum",
        "2",
        "--ghost-mode",
        args.ghost_mode,
    ]
    random.seed(args.seed)
    os.environ["MAAPACMAN_PAUSE_ON_START"] = "1"
    import pygame

    bridge = RecordingBridge(pygame, output, args.ghost_mode, args.episode_life_mode)
    bridge.start()
    # Retain pygame's original 60 Hz clock. The IPC bridge blocks only while
    # waiting for actions; those wall-clock waits produce no captured frames.
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        bridge.index.close()
        pygame.quit()


if __name__ == "__main__":
    main()
