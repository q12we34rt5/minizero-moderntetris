import createEngineModule, { type EngineWasmModule } from './engine-wasm.js';
import { VIEW_SIZE, parseView, type GameView } from './view.ts';

export interface StepResult {
  success: boolean;
  forcedHardDrop: boolean;
}

/** A placement-level move. lockX/lockY are engine board coordinates. */
export interface Placement {
  useHold: boolean;
  lockX: number;
  lockY: number;
  orientation: number;
  spinType: number;
}

const MAX_PLACEMENTS = 512;
const MAX_PATH = 256;

/**
 * Thin wrapper around the WASM moderntetris engine. Owns one engine context
 * (one board / one player) plus scratch buffers for serialized output.
 */
export class Engine {
  private readonly mod: EngineWasmModule;
  private readonly ctx: number;
  private readonly viewPtr: number;
  private readonly fullPtr: number;
  private readonly placementsPtr: number;
  private readonly pathPtr: number;
  readonly codecSize: number;

  private constructor(mod: EngineWasmModule, ctx: number) {
    this.mod = mod;
    this.ctx = ctx;
    this.codecSize = mod._et_codec_size();
    this.viewPtr = mod._malloc(VIEW_SIZE * 4);
    this.fullPtr = mod._malloc(this.codecSize * 4);
    this.placementsPtr = mod._malloc(MAX_PLACEMENTS * 4 * 4);
    this.pathPtr = mod._malloc(MAX_PATH * 4);
  }

  static async create(): Promise<Engine> {
    const mod = await createEngineModule();
    return new Engine(mod, mod._et_create());
  }

  /** piece_life <= 0 disables the forced hard drop. allSpin enables the all-spin
   *  ruleset (carried in the serialized state, so it must match the AI backend). */
  setConfig(pieceLife: number, autoDrop: boolean, allSpin: boolean): void {
    this.mod._et_set_config(this.ctx, pieceLife, autoDrop ? 1 : 0, allSpin ? 1 : 0);
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

  /** Serialize the current state into a fresh GameView (for rendering). */
  read(): GameView {
    this.mod._et_serialize(this.ctx, this.viewPtr);
    const base = this.viewPtr >> 2;
    return parseView(this.mod.HEAP32.subarray(base, base + VIEW_SIZE));
  }

  /** Serialize the full engine context (for the AI backend). Returns a copy. */
  serializeFull(): Int32Array {
    this.mod._et_serialize_full(this.ctx, this.fullPtr);
    const base = this.fullPtr >> 2;
    return this.mod.HEAP32.slice(base, base + this.codecSize);
  }

  /** Enumerate legal placements for the current piece (no-hold branch only). */
  findPlacements(): Placement[] {
    const n = this.mod._et_find_placements(this.ctx, this.placementsPtr, MAX_PLACEMENTS);
    const base = this.placementsPtr >> 2;
    const raw = this.mod.HEAP32.subarray(base, base + n * 4);
    const out: Placement[] = [];
    for (let i = 0; i < n; i++) {
      out.push({
        useHold: false,
        lockX: raw[i * 4 + 0],
        lockY: raw[i * 4 + 1],
        orientation: raw[i * 4 + 2],
        spinType: raw[i * 4 + 3],
      });
    }
    return out;
  }

  /**
   * Resolve a placement to its step-action sequence ([HOLD?] + path + HARD_DROP)
   * without mutating the engine. Replay it through step() to animate the move.
   * Returns an empty array if no matching placement exists.
   */
  placementPath(p: Placement): number[] {
    const n = this.mod._et_placement_path(
      this.ctx,
      p.useHold ? 1 : 0,
      p.lockX,
      p.lockY,
      p.orientation,
      p.spinType,
      this.pathPtr,
      MAX_PATH,
    );
    const base = this.pathPtr >> 2;
    return Array.from(this.mod.HEAP32.subarray(base, base + n));
  }

  /** Apply a placement-level move. Returns false if no matching placement exists. */
  applyPlacement(p: Placement): boolean {
    return (
      this.mod._et_apply_placement(
        this.ctx,
        p.useHold ? 1 : 0,
        p.lockX,
        p.lockY,
        p.orientation,
        p.spinType,
      ) !== 0
    );
  }

  dispose(): void {
    this.mod._free(this.pathPtr);
    this.mod._free(this.placementsPtr);
    this.mod._free(this.fullPtr);
    this.mod._free(this.viewPtr);
    this.mod._et_free(this.ctx);
  }
}
