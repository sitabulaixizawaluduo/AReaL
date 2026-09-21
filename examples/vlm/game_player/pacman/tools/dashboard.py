# SPDX-License-Identifier: Apache-2.0

"""Read-only local dashboard: one latest PNG and bounded state per planned game."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import unquote, urlsplit


class DashboardState:
    TEXT_LIMIT = 8192

    def __init__(self, keep_open=False):
        self._lock = threading.Lock()
        self._episodes = {}
        self._frames = {}
        self._started = {}
        self.keep_open = bool(keep_open)
        self.status = "preparing"
        self.error_type = None
        self.counts = None

    def plan(self, rows):
        with self._lock:
            self._episodes = {
                row["episode_id"]: {
                    "episode_id": row["episode_id"],
                    "seed": row["seed"],
                    "status": "queued",
                    "frame_version": 0,
                    "elapsed_seconds": 0,
                    "raw_output": "",
                    "reasoning": "",
                    "reward": None,
                }
                for row in rows
            }
            self._frames.clear()
            self._started.clear()
            self.status = "running"
            self.error_type = None
            self.counts = None

    def begin(self, episode_id):
        with self._lock:
            self._started[episode_id] = time.monotonic()
            self._episodes[episode_id].update(status="running", phase="reset")

    def frame(self, episode_id, image, info):
        from PIL import Image

        try:
            with Image.fromarray(image) as picture, BytesIO() as buffer:
                picture.save(buffer, format="PNG")
                png = buffer.getvalue()
        except Exception as error:
            with self._lock:
                self._episodes[episode_id]["display_error"] = type(error).__name__
            return
        with self._lock:
            row = self._episodes[episode_id]
            self._frames[episode_id] = png
            row["frame_version"] += 1
            for name in (
                "score",
                "death_count",
                "normal_pellets_remaining",
                "power_pellets_remaining",
                "step",
            ):
                if info.get(name) is not None:
                    row[name] = int(info[name])
            lives = info.get("lives_after_step", info.get("lives"))
            row["lives"] = int(lives) if lives is not None else None
            if "normal_pellets_remaining" in row:
                row.setdefault(
                    "normal_pellets_initial", row["normal_pellets_remaining"]
                )
                initial = row["normal_pellets_initial"]
                row["normal_pellet_clear_rate"] = (
                    1 - row["normal_pellets_remaining"] / initial if initial else 1.0
                )

    def event(self, episode_id, event):
        with self._lock:
            row = self._episodes[episode_id]
            kind = event["kind"]
            if kind == "request":
                row.update(
                    phase="thinking",
                    decision=int(event["decision"]) + 1,
                )
            elif kind == "decision":
                text = event["text"]
                reasoning = event.get("reasoning", "")
                reasoning = reasoning if isinstance(reasoning, str) else ""
                row.update(
                    phase="received",
                    parsed_action=None,
                    executed_action=None,
                    raw_output=text[: self.TEXT_LIMIT],
                    output_truncated=len(text) > self.TEXT_LIMIT,
                    reasoning=reasoning[: self.TEXT_LIMIT],
                    reasoning_truncated=len(reasoning) > self.TEXT_LIMIT,
                )
            elif kind == "parsed_action":
                row.update(parsed_action=event["action"], action_legal=event["legal"])
            elif kind == "executed_action":
                row.update(phase="running", executed_action=event["action"])

    def finish(self, episode_id, result):
        with self._lock:
            row = self._episodes[episode_id]
            row.update(
                status="error" if result.get("status") == "error" else "terminal",
                phase="finished",
                elapsed_seconds=float(result.get("elapsed_seconds", 0)),
            )
            for key in (
                "win",
                "death_count",
                "game_score",
                "normal_pellet_clear_rate",
                "total_shaped_reward",
            ):
                if key in result:
                    row[key] = result[key]
            row["reward"] = result.get("reward", result.get("total_shaped_reward"))
            row["terminal_reason"] = str(result.get("terminal_reason", "unknown"))[:256]
            # Do not expose SDK exception messages, request headers or credentials.
            if result.get("error_type"):
                row["error_type"] = str(result["error_type"])[:128]
            self._started.pop(episode_id, None)

    def finalizing(self):
        with self._lock:
            if self.status in {"preparing", "running"}:
                self.status = "finalizing"

    def complete(self, summary=None, error_type=None):
        with self._lock:
            self.error_type = error_type
            if error_type:
                for episode_id, row in self._episodes.items():
                    if row["status"] in {"queued", "running"}:
                        row.update(status="error", error_type=error_type)
                        started = self._started.pop(episode_id, None)
                        if started is not None:
                            row["elapsed_seconds"] = time.monotonic() - started
            observed_complete = sum(
                row["status"] == "terminal" for row in self._episodes.values()
            )
            observed_errors = sum(
                row["status"] == "error" for row in self._episodes.values()
            )
            planned = len(self._episodes)
            counts = {
                "planned_episodes": planned,
                "recorded_episodes": observed_complete + observed_errors,
                "completed_episodes": observed_complete,
                "pending_episodes": planned - observed_complete - observed_errors,
            }
            if summary is not None:
                counts.update({key: int(summary[key]) for key in counts})
            self.counts = {**counts, "error_episodes": observed_errors}
            planned = counts["planned_episodes"]
            if (
                not error_type
                and not observed_errors
                and planned > 0
                and counts["recorded_episodes"]
                == counts["completed_episodes"]
                == planned
                and counts["pending_episodes"] == 0
                and observed_complete == len(self._episodes) == planned
            ):
                self.status = "complete"
            elif observed_complete > 0:
                self.status = "partial"
            else:
                self.status = "error"

    def exit_code(self):
        with self._lock:
            if self.status == "complete":
                return 0
            if self.status in {"partial", "error"}:
                return 2
            return 130

    def snapshot(self):
        with self._lock:
            rows = [dict(row) for row in self._episodes.values()]
            for row in rows:
                if row["episode_id"] in self._started:
                    row["elapsed_seconds"] = (
                        time.monotonic() - self._started[row["episode_id"]]
                    )
            return {
                "status": self.status,
                "lifecycle": "keep_open" if self.keep_open else "auto_close",
                "keep_open": self.keep_open,
                "error_type": self.error_type,
                "counts": dict(self.counts) if self.counts is not None else None,
                "episodes": rows,
            }

    def png(self, episode_id):
        with self._lock:
            return self._frames.get(episode_id)


PAGE = rb"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pacman live</title>
<style>
body{margin:0;background:#101522;color:#e8ecf7;font:15px system-ui,sans-serif}header{padding:22px 28px;border-bottom:1px solid #303950}h1{margin:0 0 8px;font-size:24px}#status{color:#a4b6d6}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(340px,100%),1fr));gap:18px;padding:24px}.card{min-width:0;background:#1a2234;border:1px solid #303950;border-radius:12px;overflow:hidden}.title,.metrics,.action{padding:12px 16px}.title{font-weight:650;display:flex;justify-content:space-between;gap:8px}.image{background:#05070c;min-height:160px;display:grid;place-items:center}.image img{display:block;max-width:100%;max-height:480px;image-rendering:pixelated}.metrics{line-height:1.7}.bar{height:6px;background:#303950}.fill{height:100%;background:#70dca2;width:0}.action{border-top:1px solid #303950}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:160px;overflow:auto;font:13px ui-monospace,monospace}.label{color:#a4b6d6;font-size:12px;margin-top:10px}.error{color:#ff9999}.terminal{color:#70dca2}
</style><header><h1>Pacman live</h1><div id="status">Connecting...</div></header><main class="grid" id="grid"></main>
<script>
const cards=new Map();const number=(x,d=0)=>x==null?'\u2014':Number(x).toFixed(d);
function card(id){if(cards.has(id))return cards.get(id);const el=document.createElement('section');el.className='card';el.innerHTML='<div class="title"><span class="name"></span><span class="state"></span></div><div class="image"><img alt="Waiting for game frame"></div><div class="bar"><div class="fill"></div></div><div class="metrics"></div><div class="action"><div class="moves"></div><div class="label">Reasoning</div><pre class="reasoning"></pre><div class="label">Final output</div><pre class="output"></pre></div>';document.getElementById('grid').append(el);cards.set(id,el);return el;}
async function refresh(){try{const response=await fetch('/api/episodes',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();const lifecycle=data.keep_open?'Page stays open until Ctrl+C':'Closes automatically after finalization';document.getElementById('status').textContent=data.status+' | '+data.episodes.length+' games'+(data.counts?' | completed '+data.counts.completed_episodes+'/'+data.counts.planned_episodes+' | errors '+data.counts.error_episodes+' | pending '+data.counts.pending_episodes:'')+' | '+(data.error_type||lifecycle);for(const row of data.episodes){const el=card(row.episode_id);el.querySelector('.name').textContent=row.episode_id+' | seed '+row.seed;const state=el.querySelector('.state');state.textContent=row.status+(row.status==='running'?' / '+row.phase:'');state.className='state '+row.status;if(row.frame_version&&el.dataset.frame!==String(row.frame_version)){el.querySelector('img').src='/frames/'+encodeURIComponent(row.episode_id)+'.png?v='+row.frame_version;el.dataset.frame=row.frame_version;}el.querySelector('.fill').style.width=Math.max(0,Math.min(100,100*(row.normal_pellet_clear_rate||0)))+'%';el.querySelector('.metrics').textContent='Score '+number(row.game_score??row.score)+' | Lives '+number(row.lives)+' | Deaths '+number(row.death_count)+' | Pellets '+number(row.normal_pellet_clear_rate==null?null:100*row.normal_pellet_clear_rate,1)+'% | Step '+number(row.step)+' | Decision '+number(row.decision)+' | Time '+number(row.elapsed_seconds,1)+'s | Reward '+number(row.reward,3)+(row.terminal_reason?' | '+row.terminal_reason:'')+(row.error_type?' | '+row.error_type:'');el.querySelector('.moves').textContent='Parsed: '+(row.parsed_action??'\u2014')+' | Executed: '+(row.executed_action??'\u2014');el.querySelector('.reasoning').textContent=(row.reasoning||'No reasoning returned')+(row.reasoning_truncated?'\n[display truncated]':'');el.querySelector('.output').textContent=(row.raw_output||'Waiting for model output')+(row.output_truncated?'\n[display truncated]':'');}}catch(error){document.getElementById('status').textContent='Connection interrupted: '+error.message;}setTimeout(refresh,500);}refresh();
</script></html>"""


class DashboardServer:
    def __init__(self, state, host="127.0.0.1", port=8765):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = unquote(urlsplit(self.path).path)
                if path == "/":
                    payload, content_type = PAGE, "text/html; charset=utf-8"
                elif path == "/api/episodes":
                    payload = json.dumps(state.snapshot(), allow_nan=False).encode()
                    content_type = "application/json"
                elif path.startswith("/frames/") and path.endswith(".png"):
                    payload = state.png(path[len("/frames/") : -4])
                    content_type = "image/png"
                else:
                    payload, content_type = None, "text/plain"
                if payload is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @staticmethod
    async def run(args):
        from examples.vlm.game_player.pacman.tools.evaluate import PacmanEvaluation

        state = DashboardState(keep_open=args.keep_open)
        dashboard = DashboardServer(state, args.host, args.port)
        dashboard.start()
        host, port = dashboard.server.server_address[:2]
        lifecycle = " (Ctrl+C to close)" if args.keep_open else ""
        print(f"Pacman dashboard: http://{host}:{port}/{lifecycle}")
        try:
            try:
                summary = await PacmanEvaluation.run(args, dashboard=state)
                state.complete(summary)
            except Exception as error:
                state.complete(error_type=type(error).__name__)
            if args.keep_open:
                await asyncio.Event().wait()
            return state.exit_code()
        except asyncio.CancelledError:
            # Ctrl+C closes the retained page without hiding the evaluation result.
            return state.exit_code()
        finally:
            await asyncio.to_thread(dashboard.close)
