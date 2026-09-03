# Model router

A thin WebSocket router that sits between nginx and the per-model minizero
inference servers, so the demo can switch models **from the UI** without anyone
opening/closing backends by hand.

```
                              ┌───────────────────────── minizero server (model A)  :62220
browser ── nginx ── router ──┤   ws://host:port          minizero server (model B)  :62221
        (https/wss)   │       └───────────────────────── ...
                      │
              GET /models         → list models (id + display_name) for the UI
              WSS /model/<id>     → transparent proxy to that model's server
```

## Why a router (and not per-server changes)

- Each minizero `server.py` stays exactly as it is: **one process, one model,
  one port**. No console pool, no manifest, no protocol change.
- Model identity (display name) lives in **one place** — this router's registry
  — instead of being baked into every inference server.
- The `request_move` / `move` frames are forwarded **verbatim**, so the frontend
  and the minizero servers speak the same protocol they always did.
- A model whose backend is down just fails to proxy → the frontend shows
  `disconnected`, the same behaviour as pointing straight at a dead backend.

## Registry (`models.json`)

A JSON array, one object per model. **Hot-reloaded**: edit and save, the next
`/models` request or `/model/<id>` connection picks it up — no restart. A parse
error keeps the last good copy.

```json
[
  {
    "id": "gpz-n160",
    "display_name": "GumbelZero 3b×256 · 160 sims",
    "desc": "placement gpz, 160 simulations",
    "host": "127.0.0.1",
    "port": 62220
  }
]
```

| field          | required | meaning                                             |
| -------------- | -------- | --------------------------------------------------- |
| `id`           | yes      | URL slug: the frontend connects to `/model/<id>`    |
| `display_name` | no       | shown on the board label (defaults to `id`)         |
| `desc`         | no       | optional extra line for the UI                      |
| `host`         | yes      | the minizero server's host                          |
| `port`         | yes      | the minizero server's port                          |
| `path`         | no       | upstream WS path (default `""`; servers serve root) |

`host`/`port` are **never** sent to the browser — `/models` returns only
`id` / `display_name` / `desc`.

## Endpoints

- `GET /models` → `[{"id", "display_name", "desc"?}, ...]` (CORS-open, no-store).
- `GET /healthz` → `ok`.
- `WSS /model/<id>` → proxy to `ws://<host>:<port><path>` for that `id`.
  Unknown `id` → close 1008; upstream down → close 1011.

## Running

Inside the project container (same place the minizero servers run):

```bash
pip install -r web/router/requirements.txt

# one minizero server per model, each on its own port (unchanged):
python web/backend/server.py --cfg <A.cfg> --model <A.pt> --port 62220 &
python web/backend/server.py --cfg <B.cfg> --model <B.pt> --port 62221 &

# the router in front of them:
python web/router/router.py --registry web/router/models.json --port 8000
```

Options: `--registry` (default `web/router/models.json`), `--host`
(default `127.0.0.1`), `--port` (default `8000`).

## nginx

Point one location at the router for both the model list (HTTP) and the proxy
(WS upgrade). Example:

```nginx
# model list (plain HTTP/HTTPS)
location = /models {
    proxy_pass http://127.0.0.1:8000/models;
}

# per-model websocket proxy
location /model/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade    $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
}
```

The frontend is then configured with a single base URL
(`wss://tetris.cgi.lab.nycu.edu.tw`); it fetches `/models` for the dropdown and
connects to `/model/<id>` for play. The old `/service1` `/service2` locations
can stay as direct (router-less) connections or be retired.
