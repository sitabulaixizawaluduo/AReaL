"""Recording-only adapter to the pinned game's IPC worker, without game edits."""

import hashlib
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from maapacman.env.pygame_environment import PygamePacmanEnv


class RecordingEnv(PygamePacmanEnv):
    """Only substitute the worker entrypoint; retain all original game transitions.

    The upstream version lacks a worker-module constructor argument. This small
    launch adapter uses its pinned private IPC methods; it never replaces globals
    or copies the game. Keep this seam covered when updating the source lock.
    """

    def __init__(self, config, recording_root):
        super().__init__(config)
        self.recording_root = str(Path(recording_root).resolve())

    def _start_worker(self):
        self._prepare_runtime_dir()
        environment = os.environ.copy()
        environment.update(
            PYTHONUNBUFFERED="1",
            PYTHONHASHSEED=str(self._seed),
            PACMAN_RECORDING_ROOT=self.recording_root,
        )
        for option, key in (
            (self.config.video_driver, "SDL_VIDEODRIVER"),
            (self.config.audio_driver, "SDL_AUDIODRIVER"),
        ):
            if option is not None:
                environment[key] = option
            else:
                environment.pop(key, None)
        try:
            process = subprocess.Popen(
                [
                    str(self.config.python_executable),
                    str(Path(__file__).resolve().with_name("recording_worker.py")),
                    "--script",
                    str(self._runtime_script),
                    "--seed",
                    str(self._seed),
                    "--ghost-mode",
                    self.config.ghost_mode,
                    "--episode-life-mode",
                    self.config.episode_life_mode,
                ],
                cwd=self._runtime_dir,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except BaseException:
            self._cleanup_runtime_dir()
            raise
        self._process = process
        self._messages = queue.Queue()
        self._stderr.clear()
        threading.Thread(
            target=self._read_stdout, args=(process.stdout,), daemon=True
        ).start()
        threading.Thread(
            target=self._read_stderr, args=(process.stderr,), daemon=True
        ).start()


class VideoAudit:
    @staticmethod
    def encode(root: Path, episode: dict):
        from PIL import Image

        if not episode.get("zero_death_win") or episode.get("death_count") != 0:
            raise ValueError(
                "This complete attempt is not a zero-death win; frames retained"
            )
        if (episode.get("ghost_mode"), episode.get("episode_life_mode")) != (
            "normal",
            "third_death_ends_episode",
        ):
            raise ValueError(
                "Video must preserve normal ghosts and the third-death episode boundary"
            )
        trajectory = episode.get("trajectory", [])
        events = [
            event for step in trajectory for event in step.get("logic_frame_events", [])
        ]
        if not trajectory or any(
            step.get("death_count", 0) != 0 for step in trajectory
        ):
            raise ValueError("Missing trajectory or an actual death was recorded")
        if any(event["event_type"] == "death" for event in events):
            raise ValueError(
                "Death event cannot be removed from a successful recording"
            )
        if not any(event["event_type"] == "level_cleared" for event in events):
            raise ValueError("Missing native level-cleared event")
        if episode.get("terminal_reason") != "all_normal_pellets":
            raise ValueError("Episode did not terminate by clearing ordinary pellets")
        if episode.get("normal_pellet_clear_rate") != 1.0:
            raise ValueError("Normal pellets have not been cleared")
        frames = [
            json.loads(line)
            for line in (root / "frames.jsonl").read_text().splitlines()
        ]
        if not frames or frames[0]["logic_frame"] != 0 or frames[0]["mode"] != 1:
            raise ValueError("Recording must include the initial reset frame")
        ids = [row["logic_frame"] for row in frames]
        if sorted(set(ids)) != list(range(ids[-1] + 1)) or ids != sorted(ids):
            raise ValueError("Missing or reordered simulation frames")
        if frames[-1]["mode"] not in (6, 9) or frames[-1]["normal_pellets"] != 0:
            raise ValueError("Final win frame is missing")
        if ids[-1] != trajectory[-1]["logic_frame"]:
            raise ValueError("Recording and trajectory end at different logical frames")
        if any(event["logic_frame"] not in ids for event in events):
            raise ValueError("An event frame is missing from the recording")
        if frames[-1]["score"] != episode["game_score"]:
            raise ValueError("Episode and recorded game score differ")
        if any(a["lives"] > b["lives"] for a, b in zip(frames, frames[1:])):
            raise ValueError("Life loss present in recording")
        for i, row in enumerate(frames):
            if row["index"] != i:
                raise ValueError("Frame sequence indices are not contiguous")
            with Image.open(root / row["file"]) as image:
                if (
                    hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                    != row["rgb_sha256"]
                ):
                    raise ValueError("PNG integrity mismatch")
        output = root.parent / "zero-death-game.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-framerate",
                "60",
                "-i",
                str(root / "frame-%06d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_streams",
                    "-of",
                    "json",
                    str(output),
                ]
            )
        )
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        if (
            int(video["nb_read_frames"]) != len(frames)
            or video["avg_frame_rate"] != "60/1"
        ):
            raise ValueError("Encoded video frame count or FPS differs from source")
        acceptance = {
            "episode_id": episode["episode_id"],
            "frames": len(frames),
            "first_logic_frame": 0,
            "last_logic_frame": ids[-1],
            "fps": 60,
            "duration_seconds": len(frames) / 60,
            "mp4": str(output),
            "death_count": 0,
            "game_score": episode["game_score"],
            "editing": "full rendered simulation sequence; no inference-wait frames inserted",
            "ffprobe": probe,
        }
        (root.parent / "video-acceptance.json").write_text(
            json.dumps(acceptance, indent=2) + "\n"
        )
        return acceptance
