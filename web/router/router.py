"""WSS model router for the ModernTetris web frontend.

Sits between nginx and the per-model minizero inference servers. Two jobs:

    GET  /models       -> the registry's public view (id + display_name [+ desc]),
                          so the frontend can list models and label each board
                          with a real name instead of "AI1"/"AI2".
    WSS  /model/<id>    -> a transparent WebSocket proxy to that model's minizero
                          server (host:port from the registry). Frames are
                          forwarded verbatim in both directions, so the existing
                          request_move/move protocol is untouched and the
                          minizero servers need NO changes -- they never learn
                          their own display name; only the router does.

The registry is a JSON file, re-read whenever it changes on disk (mtime), so
models can be added / renamed / repointed by editing the file -- no router
restart ("hot" edit). A registry entry whose backend is down simply fails to
proxy: the client WS is closed and the frontend shows "disconnected", exactly
as it did when pointed straight at a dead backend.

Run (inside the project container, alongside the minizero servers):

    pip install -r web/router/requirements.txt
    python web/router/router.py --registry web/router/models.json \
        [--host 127.0.0.1] [--port 8000]

See web/router/README.md for the registry format and the nginx mapping.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import json
from pathlib import Path

import websockets

MODEL_PREFIX = "/model/"


class Registry:
    """The router's only state: id -> {display_name, host, port, path, desc}.

    Reloaded from disk whenever the file's mtime changes, so the JSON can be
    edited live. A parse error keeps the previous good copy rather than dropping
    every model."""

    def __init__(self, path: Path):
        self._path = path
        self._mtime: float | None = None
        self._entries: dict[str, dict] = {}
        self._order: list[str] = []
        self.reload()

    def reload(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError as e:
            if self._mtime is not None or self._entries:
                print(f"[router] registry gone: {self._path} ({e})", flush=True)
            self._entries, self._order, self._mtime = {}, [], None
            return
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[router] registry parse error, keeping previous copy: {e}", flush=True)
            return
        entries: dict[str, dict] = {}
        order: list[str] = []
        for raw in data:
            try:
                mid = str(raw["id"])
                entry = {
                    "id": mid,
                    "display_name": str(raw.get("display_name", mid)),
                    "host": str(raw["host"]),
                    "port": int(raw["port"]),
                    "path": str(raw.get("path", "")),
                    "desc": str(raw["desc"]) if raw.get("desc") not in (None, "") else None,
                }
            except (KeyError, TypeError, ValueError) as e:
                print(f"[router] skipping bad registry entry {raw!r}: {e}", flush=True)
                continue
            if mid in entries:
                print(f"[router] duplicate model id {mid!r}, keeping the first", flush=True)
                continue
            entries[mid] = entry
            order.append(mid)
        self._entries, self._order, self._mtime = entries, order, mtime
        print(f"[router] loaded {len(order)} model(s): {', '.join(order) or '(none)'}", flush=True)

    def public(self) -> list[dict]:
        """The frontend-facing view: never leaks host/port."""
        self.reload()
        out = []
        for mid in self._order:
            e = self._entries[mid]
            item = {"id": e["id"], "display_name": e["display_name"]}
            if e["desc"]:
                item["desc"] = e["desc"]
            out.append(item)
        return out

    def lookup(self, mid: str) -> dict | None:
        self.reload()
        return self._entries.get(mid)


REGISTRY: Registry | None = None


async def process_request(path: str, request_headers):
    """Intercept plain-HTTP requests before the WS handshake.

    `/models` and health checks are answered here as HTTP; `/model/<id>` returns
    None so websockets proceeds with the WebSocket upgrade (handled by `proxy`).
    """
    raw_path = path.split("?", 1)[0]
    if raw_path.startswith(MODEL_PREFIX):
        return None  # let the WS handshake proceed -> proxy()
    if raw_path.rstrip("/") == "/models":
        body = json.dumps(REGISTRY.public()).encode()
        headers = [
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", "*"),  # dev frontend runs on a different origin
            ("Cache-Control", "no-store"),
        ]
        return http.HTTPStatus.OK, headers, body
    if raw_path.rstrip("/") in ("", "/healthz"):
        return http.HTTPStatus.OK, [("Content-Type", "text/plain")], b"ok\n"
    return http.HTTPStatus.NOT_FOUND, [("Content-Type", "text/plain")], b"not found\n"


async def _pump(src, dst) -> None:
    """Forward every frame src -> dst, then close dst so the peer tears down too."""
    try:
        async for message in src:
            await dst.send(message)
    except Exception:  # noqa: BLE001 -- any src failure just ends this direction
        pass
    finally:
        try:
            await dst.close()
        except Exception:  # noqa: BLE001
            pass


async def proxy(ws) -> None:
    """Bridge a client WS at /model/<id> to that model's minizero server."""
    raw_path = ws.path.split("?", 1)[0]
    mid = raw_path[len(MODEL_PREFIX):].strip("/")
    entry = REGISTRY.lookup(mid)
    if entry is None:
        print(f"[router] unknown model id: {mid!r}", flush=True)
        await ws.close(code=1008, reason="unknown model")
        return

    upstream_url = f"ws://{entry['host']}:{entry['port']}{entry['path']}"
    try:
        upstream = await websockets.connect(upstream_url, max_size=None, open_timeout=5)
    except Exception as e:  # noqa: BLE001 -- backend down / unreachable -> client sees disconnect
        print(f"[router] upstream connect failed for {mid!r} ({upstream_url}): {e}", flush=True)
        await ws.close(code=1011, reason="upstream unavailable")
        return

    print(f"[router] proxy open: {mid!r} <-> {upstream_url}", flush=True)
    try:
        await asyncio.gather(_pump(ws, upstream), _pump(upstream, ws))
    finally:
        await upstream.close()
        print(f"[router] proxy closed: {mid!r}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description="ModernTetris web model router")
    parser.add_argument(
        "--registry",
        default=str(Path(__file__).resolve().parent / "models.json"),
        help="path to the models registry JSON (hot-reloaded on change)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global REGISTRY
    REGISTRY = Registry(Path(args.registry).resolve())

    async with websockets.serve(
        proxy, args.host, args.port, process_request=process_request, max_size=None
    ):
        print(
            f"[router] listening on {args.host}:{args.port}  "
            f"(GET /models, WSS {MODEL_PREFIX}<id>)",
            flush=True,
        )
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
