"""WebSocket AI backend for ModernTetris web (Phase 2: PvE).

A thin, stateless inference oracle. It wraps the minizero console binary as a
subprocess and, for each request, injects the client's board via `set_state`
then runs `genmove` to get the AI's placement.

Must run where the minizero binary works (i.e. inside the training container:
libtorch + GPU + the model checkpoint). The container uses --network=host so
the frontend can reach this over localhost.

Usage (from the repo root, inside the container):

    pip install -r web/backend/requirements.txt
    python web/backend/server.py \\
        --cfg   <path to model .cfg> \\
        --model <path to model .pt> \\
        [--bin build/moderntetris_placement/minizero_moderntetris_placement] \\
        [--host 127.0.0.1] [--port 8001]

The model cfg and .pt path are REQUIRED and explicit -- nothing is hardcoded.

Protocol (JSON over WebSocket):
    client -> {"type": "request_move", "state": [<codec ints>]}
    server -> {"type": "move", "placement": {"use_hold", "lock_x", "lock_y",
                                             "orientation", "spin_type"}}
              placement is null if the AI resigned / topped out / passed.
    server -> {"type": "error", "message": "..."}  on failure
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import websockets

GENMOVE_RE = re.compile(r"^h([01])_x(-?\d+)_y(-?\d+)_o(\d)_s(\d)$")
REPO_ROOT = Path(__file__).resolve().parents[2]


class MinizeroConsole:
    """Owns the minizero console subprocess. One inference at a time."""

    def __init__(self, bin_path: Path, cfg: Path, model: Path, conf_str: str = ""):
        # minizero's -conf_str is a ':'-separated list of key=value pairs;
        # later pairs override earlier ones. The model path and the
        # web-play garbage setting come first; the caller's --conf_str is
        # appended last so it can override anything (it is an explicit
        # power-user knob). env_modern_tetris_garbage_probability=0 is
        # intentional for web play -- garbage should only come from the
        # opponent board, not random injection.
        conf_parts = [
            f"nn_file_name={model}",
            "env_modern_tetris_garbage_probability=0",
        ]
        if conf_str:
            conf_parts.append(conf_str)
        self._cmd = [
            str(bin_path),
            "-mode", "console",
            "-conf_file", str(cfg),
            "-conf_str", ":".join(conf_parts),
        ]
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        print(f"[backend] launching: {' '.join(self._cmd)}", flush=True)
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            cwd=str(REPO_ROOT),
        )
        # First command triggers network load + warmup; wait for it to settle.
        reply = await self._command("name")
        print(f"[backend] console ready (name -> {reply!r})", flush=True)

    async def _command(self, line: str) -> str:
        """Send one console command, return the reply message (without the '='/'?')."""
        assert self._proc and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write((line + "\n").encode())
        await self._proc.stdin.drain()
        # minizero's reply() writes "<=|?><id> <msg>\n\n"; skip anything until
        # the response line. Our commands all produce single-line messages.
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                raise RuntimeError("minizero console closed unexpectedly")
            text = raw.decode(errors="replace").rstrip("\r\n")
            if text.startswith("=") or text.startswith("?"):
                ok = text.startswith("=")
                msg = text[1:].strip()
                if not ok:
                    raise RuntimeError(f"console command failed: {line!r} -> {text!r}")
                return msg

    async def get_move(self, state: list[int]) -> dict | None:
        """Inject `state`, run genmove, return the parsed placement (or None)."""
        async with self._lock:
            await self._command("set_state " + " ".join(str(int(v)) for v in state))
            reply = await self._command("genmove b")
        m = GENMOVE_RE.match(reply)
        if not m:
            # Resign / PASS / unexpected -> AI has no move (treated as a loss).
            print(f"[backend] genmove -> {reply!r} (no placement)", flush=True)
            return None
        return {
            "use_hold": int(m.group(1)),
            "lock_x": int(m.group(2)),
            "lock_y": int(m.group(3)),
            "orientation": int(m.group(4)),
            "spin_type": int(m.group(5)),
        }


async def handle(ws, console: MinizeroConsole) -> None:
    print(f"[backend] client connected: {ws.remote_address}", flush=True)
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "message": "invalid JSON"}))
                continue
            if msg.get("type") != "request_move":
                await ws.send(json.dumps({"type": "error", "message": "unknown message type"}))
                continue
            state = msg.get("state")
            if not isinstance(state, list):
                await ws.send(json.dumps({"type": "error", "message": "missing state array"}))
                continue
            try:
                placement = await console.get_move(state)
            except Exception as e:  # noqa: BLE001 -- surface any console failure to the client
                await ws.send(json.dumps({"type": "error", "message": str(e)}))
                continue
            await ws.send(json.dumps({"type": "move", "placement": placement}))
    except websockets.ConnectionClosed:
        pass
    finally:
        print("[backend] client disconnected", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description="ModernTetris web AI backend")
    parser.add_argument("--cfg", required=True, help="path to the model .cfg file")
    parser.add_argument("--model", required=True, help="path to the model .pt checkpoint")
    parser.add_argument(
        "--bin",
        default=str(REPO_ROOT / "build/moderntetris_placement/minizero_moderntetris_placement"),
        help="path to the minizero_moderntetris_placement binary",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--conf_str",
        default="",
        help="extra minizero -conf_str overrides (':'-separated key=value pairs), "
        "appended last so they override the defaults",
    )
    args = parser.parse_args()

    cfg = Path(args.cfg).resolve()
    model = Path(args.model).resolve()
    bin_path = Path(args.bin).resolve()
    for label, p in [("cfg", cfg), ("model", model), ("bin", bin_path)]:
        if not p.exists():
            sys.exit(f"error: --{label} not found: {p}")

    console = MinizeroConsole(bin_path, cfg, model, args.conf_str)
    await console.start()

    async with websockets.serve(lambda ws: handle(ws, console), args.host, args.port):
        print(f"[backend] listening on ws://{args.host}:{args.port}", flush=True)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
