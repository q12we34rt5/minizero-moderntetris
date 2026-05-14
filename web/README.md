# ModernTetris — Web

Browser frontend for `moderntetris` / `moderntetris_placement`. The game engine
(`minizero/environment/stochastic/moderntetris/engine`) is compiled to WASM and
runs entirely client-side; the AI opponent (Phase 2+) runs on a backend.

This directory is intentionally **not** tracked in git.

## Phase status

- **Phase 1 (current):** Vite + React + TS frontend, WASM engine, keyboard
  (DAS/ARR/gravity) local single-player. No AI, no garbage.
- **Phase 2:** minizero C++ state-setter + console mode, AI inference backend, PvE.
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

## Build & run

The WASM engine is a separate, manual build step (run once, and again whenever
the engine C++ changes):

```sh
npm install
npm run build:engine     # compiles engine -> src/engine/engine-wasm.{js,wasm}
npm run test:engine      # optional: Node smoke test of the WASM wrapper
npm run dev              # Vite dev server
```

`npm run build` produces a production bundle in `dist/`.

## Layout

```
engine/
  engine_wasm.cpp    extern "C" wrapper around the in-tree engine
  build.sh           em++ build -> src/engine/engine-wasm.{js,wasm}
  smoke-test.mjs     Node smoke test for the wrapper
src/
  engine/    WASM glue + serialized-view layout (view.ts)
  data/      tetromino shapes / colors
  input/     keyboard DAS/ARR controller (ported from TetRL-Arena)
  render/    canvas rendering
  game/      useGame hook — engine + input + loop wiring
  components/ React UI
```

The serialized view layout in `engine/engine_wasm.cpp` and `src/engine/view.ts`
must be kept in sync.
