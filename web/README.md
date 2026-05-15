# ModernTetris — Web

Browser frontend for `moderntetris` / `moderntetris_placement`. The game engine
(`minizero/environment/stochastic/moderntetris/engine`) is compiled to WASM and
runs entirely client-side (both the player's and the AI's boards). The AI
opponent's *decisions* come from a backend that wraps the minizero network.

## Phase status

- **Phase 1 (done):** Vite + React + TS frontend, WASM engine, keyboard
  (DAS/ARR/gravity) local single-player.
- **Phase 2 (done):** minizero `set_state` console command, AI inference
  backend (`backend/server.py`), PvE mode with garbage exchange.
- **Phase 3:** EvE.

## Prerequisites

- Node (tested with v24).
- [emscripten SDK](https://emscripten.org/) for building the WASM engine,
  installed locally at `~/emsdk`:

  ```sh
  git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
  cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest
  ```

  emscripten needs Python >= 3.10. If the system Python is older, install a
  modern interpreter (e.g. Miniconda at `~/miniconda3`) — `engine/build.sh`
  auto-detects it, or set `EMSDK_PYTHON` explicitly.
- For the AI backend: the minizero `moderntetris_placement` binary, which is
  built inside the training container (libtorch + GPU). See below.

## Build & run — frontend

The WASM engine is a separate, manual build step (run once, and again whenever
the engine C++ changes):

```sh
npm install
npm run build:engine     # compiles engine -> src/engine/engine-wasm.{js,wasm}
npm run test:engine      # optional: Node smoke test of the WASM wrapper
npm run test:codec       # optional: C++ round-trip test of the shared state codec
npm run dev              # Vite dev server
```

`npm run build` produces a production bundle in `dist/`.

## Build & run — AI backend (PvE)

The backend must run where the minizero binary works — i.e. **inside the
training container** (libtorch + GPU + the model checkpoint). The container
uses `--network=host`, so the frontend reaches it over localhost.

1. Build the minizero binary (inside the container, from the repo root):

   ```sh
   ./scripts/build.sh moderntetris_placement release
   ```

2. Start the backend (inside the container, from the repo root). The model
   `.cfg` and `.pt` paths are explicit and required:

   ```sh
   pip install -r web/backend/requirements.txt
   python web/backend/server.py \
       --cfg   <model dir>/<model>.cfg \
       --model <model dir>/model/weight_iter_<N>.pt
   # optional: --bin <path> --host 127.0.0.1 --port 8001
   ```

3. In the web UI, switch to **PvE** mode. Set the backend URL in the PvE panel
   if it isn't `ws://localhost:8001`.

## Layout

```
engine/
  engine_wasm.cpp         extern "C" wrapper around the in-tree engine
  build.sh                em++ build -> src/engine/engine-wasm.{js,wasm}
  smoke-test.mjs          Node smoke test for the wrapper
  codec-roundtrip-test.cpp  C++ test for the shared state codec
backend/
  server.py               WebSocket AI oracle (wraps the minizero console)
  requirements.txt
src/
  engine/    WASM glue, serialized-view layout (view.ts), Engine class
  data/      tetromino shapes / colors
  input/     source-agnostic input controller (keyboard + gamepad) with
             DAS/ARR timing; gamepad reader / remappable button mapping
  render/    canvas rendering
  game/      useGame hook — engines + input + loop + AI wiring
  ai/        WebSocket client for the AI backend
  components/ React UI
```

Two serialization formats are shared across the C++/TS boundary and must be
kept in sync:

- the **render view** — `engine/engine_wasm.cpp` ↔ `src/engine/view.ts`
- the **full state codec** — `minizero/.../engine/state_codec.hpp` (used by both
  the WASM `et_serialize_full` and the minizero console `set_state`)
