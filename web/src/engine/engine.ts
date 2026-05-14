import createEngineModule, { type EngineWasmModule } from './engine-wasm.js';
import { VIEW_SIZE, parseView, type GameView } from './view.ts';

export interface StepResult {
  success: boolean;
  forcedHardDrop: boolean;
}

/**
 * Thin wrapper around the WASM moderntetris engine. Owns one engine context
 * (one board / one player) plus a scratch buffer for serialized views.
 */
export class Engine {
  private readonly mod: EngineWasmModule;
  private readonly ctx: number;
  private readonly viewPtr: number;

  private constructor(mod: EngineWasmModule, ctx: number, viewPtr: number) {
    this.mod = mod;
    this.ctx = ctx;
    this.viewPtr = viewPtr;
  }

  static async create(): Promise<Engine> {
    const mod = await createEngineModule();
    const ctx = mod._et_create();
    const viewPtr = mod._malloc(VIEW_SIZE * 4);
    return new Engine(mod, ctx, viewPtr);
  }

  /** piece_life <= 0 disables the forced hard drop. */
  setConfig(pieceLife: number, autoDrop: boolean): void {
    this.mod._et_set_config(this.ctx, pieceLife, autoDrop ? 1 : 0);
  }

  reset(seed: number): void {
    this.mod._et_reset(this.ctx, seed >>> 0);
  }

  step(action: number): StepResult {
    const r = this.mod._et_step(this.ctx, action);
    return { success: (r & 1) !== 0, forcedHardDrop: (r & 2) !== 0 };
  }

  addGarbage(lines: number, delay: number): boolean {
    return this.mod._et_add_garbage(this.ctx, lines, delay) !== 0;
  }

  /** Serialize the current state into a fresh GameView. */
  read(): GameView {
    this.mod._et_serialize(this.ctx, this.viewPtr);
    const base = this.viewPtr >> 2;
    return parseView(this.mod.HEAP32.subarray(base, base + VIEW_SIZE));
  }

  dispose(): void {
    this.mod._free(this.viewPtr);
    this.mod._et_free(this.ctx);
  }
}
