// Types for the emscripten-generated engine-wasm.js (built by web/engine/build.sh).
export interface EngineWasmModule {
  _et_view_size(): number;
  _et_create(): number;
  _et_free(ctx: number): void;
  _et_set_config(ctx: number, pieceLife: number, autoDrop: number): void;
  _et_reset(ctx: number, seed: number): void;
  _et_step(ctx: number, action: number): number;
  _et_add_garbage(ctx: number, lines: number, delay: number): number;
  _et_serialize(ctx: number, outPtr: number): void;
  _malloc(size: number): number;
  _free(ptr: number): void;
  HEAP32: Int32Array;
}

declare const createEngineModule: (opts?: Record<string, unknown>) => Promise<EngineWasmModule>;
export default createEngineModule;
