// Keyboard input with DAS/ARR/gravity, ported faithfully from
// TetRL-Arena/web/static/index.html. Produces step-action ids that the game
// loop applies directly to the (local, synchronous) WASM engine.

export const Action = {
  MOVE_LEFT: 0,
  MOVE_RIGHT: 1,
  SOFT_DROP: 2,
  HARD_DROP: 3,
  ROTATE_CW: 4,
  ROTATE_CCW: 5,
  ROTATE_180: 6,
  HOLD: 7,
  MOVE_LEFT_TO_WALL: 8,
  MOVE_RIGHT_TO_WALL: 9,
  SOFT_DROP_TO_FLOOR: 10,
  NOOP: 11,
} as const;

export const ACTION_NAMES = [
  'MOVE_LEFT', 'MOVE_RIGHT', 'SOFT_DROP', 'HARD_DROP',
  'ROTATE_CW', 'ROTATE_CCW', 'ROTATE_180', 'HOLD',
  'MOVE_LEFT_TO_WALL', 'MOVE_RIGHT_TO_WALL', 'SOFT_DROP_TO_FLOOR', 'NOOP',
] as const;

export interface InputSettings {
  das: number;
  arr: number;
  dropDas: number;
  dropArr: number;
  dropRate: number;
}

export const DEFAULT_SETTINGS: InputSettings = {
  das: 100,
  arr: 0,
  dropDas: 50,
  dropArr: 0,
  dropRate: 500,
};

type Category = 'horizontal' | 'vertical' | 'instant';
interface Binding {
  action: number;
  wallAction?: number;
  category: Category;
}

const KEY_MAP: Record<string, Binding> = {
  ArrowLeft: { action: Action.MOVE_LEFT, wallAction: Action.MOVE_LEFT_TO_WALL, category: 'horizontal' },
  ArrowRight: { action: Action.MOVE_RIGHT, wallAction: Action.MOVE_RIGHT_TO_WALL, category: 'horizontal' },
  ArrowDown: { action: Action.SOFT_DROP, wallAction: Action.SOFT_DROP_TO_FLOOR, category: 'vertical' },
  ' ': { action: Action.HARD_DROP, category: 'instant' },
  z: { action: Action.ROTATE_CCW, category: 'instant' },
  ArrowUp: { action: Action.ROTATE_CW, category: 'instant' },
  a: { action: Action.ROTATE_180, category: 'instant' },
  c: { action: Action.HOLD, category: 'instant' },
};

function normalizeKey(key: string): string {
  return key.length === 1 ? key.toLowerCase() : key;
}

interface KeyState {
  pressed: boolean;
  pressTime: number;
  dasTriggered: boolean;
  lastRepeatTime: number;
  pressOrder: number;
}

function newKeyState(): KeyState {
  return { pressed: false, pressTime: 0, dasTriggered: false, lastRepeatTime: 0, pressOrder: 0 };
}

export interface InputControllerOptions {
  getSettings: () => InputSettings;
  onReset?: () => void;
}

export class InputController {
  private readonly getSettings: () => InputSettings;
  private readonly onReset?: () => void;
  private readonly keys: Record<string, KeyState> = {};
  private pending: number[] = [];
  private pressCounter = 0;
  private prevHKey: KeyState | null = null;
  private lastGravityTime = 0;
  private enabled = false;

  constructor(opts: InputControllerOptions) {
    this.getSettings = opts.getSettings;
    this.onReset = opts.onReset;
    for (const k of Object.keys(KEY_MAP)) this.keys[k] = newKeyState();
  }

  attach() {
    window.addEventListener('keydown', this.onKeyDown, { passive: false });
    window.addEventListener('keyup', this.onKeyUp, { passive: false });
  }

  detach() {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
  }

  /** Enable/disable game-action capture. Reset key still works when disabled. */
  setEnabled(enabled: boolean) {
    this.enabled = enabled;
  }

  /** Clear all transient input state. Call on game (re)start. */
  reset(now: number) {
    for (const k of Object.keys(this.keys)) this.keys[k] = newKeyState();
    this.pending.length = 0;
    this.prevHKey = null;
    this.pressCounter = 0;
    this.lastGravityTime = now;
  }

  private onKeyDown = (e: KeyboardEvent) => {
    if (e.repeat) return;
    if (e.target instanceof HTMLElement && e.target.tagName === 'INPUT') return;

    if (e.key === 'r' || e.key === 'R') {
      e.preventDefault();
      this.onReset?.();
      return;
    }

    const binding = KEY_MAP[normalizeKey(e.key)];
    if (!binding) return;
    e.preventDefault();
    if (!this.enabled) return;

    if (binding.category === 'instant') {
      this.pending.push(binding.action);
      return;
    }

    const ks = this.keys[normalizeKey(e.key)];
    if (ks.pressed) return;
    ks.pressed = true;
    ks.pressTime = performance.now();
    ks.dasTriggered = false;
    ks.lastRepeatTime = 0;
    ks.pressOrder = ++this.pressCounter;

    // fire the initial press immediately
    const s = this.getSettings();
    if (binding.category === 'horizontal') {
      this.pending.push(s.das === 0 ? binding.wallAction! : binding.action);
    } else {
      this.pending.push(s.dropDas === 0 ? binding.wallAction! : binding.action);
    }
  };

  private onKeyUp = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLElement && e.target.tagName === 'INPUT') return;
    const binding = KEY_MAP[normalizeKey(e.key)];
    if (!binding) return;
    e.preventDefault();
    const ks = this.keys[normalizeKey(e.key)];
    if (ks) ks.pressed = false;
  };

  /** Collect all actions for this frame: queued instants + DAS/ARR repeats + gravity. */
  collect(now: number): number[] {
    const actions: number[] = [];
    const s = this.getSettings();

    // 1. queued instant + initial-press actions
    while (this.pending.length > 0) actions.push(this.pending.shift()!);

    // 2. horizontal movement (DAS/ARR)
    const leftKey = this.keys.ArrowLeft;
    const rightKey = this.keys.ArrowRight;
    let hKey: KeyState | null = null;
    let hBinding: Binding | null = null;
    if (leftKey.pressed && rightKey.pressed) {
      if (leftKey.pressOrder > rightKey.pressOrder) {
        hKey = leftKey;
        hBinding = KEY_MAP.ArrowLeft;
      } else {
        hKey = rightKey;
        hBinding = KEY_MAP.ArrowRight;
      }
    } else if (leftKey.pressed) {
      hKey = leftKey;
      hBinding = KEY_MAP.ArrowLeft;
    } else if (rightKey.pressed) {
      hKey = rightKey;
      hBinding = KEY_MAP.ArrowRight;
    }

    if (hKey && hBinding) {
      // priority key changed to one that is already charged
      if (hKey !== this.prevHKey && hKey.dasTriggered) {
        hKey.lastRepeatTime = now;
        actions.push(s.arr === 0 ? hBinding.wallAction! : hBinding.action);
      }

      const elapsed = now - hKey.pressTime;
      if (!hKey.dasTriggered && elapsed >= s.das) {
        hKey.dasTriggered = true;
        hKey.lastRepeatTime = now;
        if (s.das > 0 || elapsed > 0) {
          actions.push(s.arr === 0 ? hBinding.wallAction! : hBinding.action);
        }
      }
      if (hKey.dasTriggered && s.arr > 0) {
        while (now - hKey.lastRepeatTime >= s.arr) {
          hKey.lastRepeatTime += s.arr;
          actions.push(hBinding.action);
        }
      }
    }
    this.prevHKey = hKey;

    // 3. vertical soft drop (Drop DAS/ARR)
    const downKey = this.keys.ArrowDown;
    const downBinding = KEY_MAP.ArrowDown;
    if (downKey.pressed) {
      const elapsed = now - downKey.pressTime;
      if (!downKey.dasTriggered && elapsed >= s.dropDas) {
        downKey.dasTriggered = true;
        downKey.lastRepeatTime = downKey.pressTime + s.dropDas;
        if (s.dropDas > 0 || elapsed > 0) {
          actions.push(s.dropArr === 0 ? downBinding.wallAction! : downBinding.action);
        }
      }
      if (downKey.dasTriggered && s.dropArr > 0) {
        while (now - downKey.lastRepeatTime >= s.dropArr) {
          downKey.lastRepeatTime += s.dropArr;
          actions.push(downBinding.action);
        }
      }
    }

    // 4. gravity (automatic soft drop at the configured drop rate)
    if (s.dropRate > 0 && this.lastGravityTime > 0) {
      while (now - this.lastGravityTime >= s.dropRate) {
        this.lastGravityTime += s.dropRate;
        actions.push(Action.SOFT_DROP);
      }
    }

    // 5. re-affirm ARR=0 wall actions if anything else moved this frame
    const anyManual = actions.length > 0;
    if (hKey && hBinding && hKey.dasTriggered && s.arr === 0 && anyManual) {
      if (!actions.includes(hBinding.wallAction!)) actions.push(hBinding.wallAction!);
    }
    if (downKey.pressed && downKey.dasTriggered && s.dropArr === 0 && anyManual) {
      if (!actions.includes(downBinding.wallAction!)) actions.push(downBinding.wallAction!);
    }

    return actions;
  }

  /**
   * When a new piece spawns, ARR=0 wall actions (one-shots) must be re-applied
   * so a held direction keeps the new piece pinned to the wall.
   */
  onNewPiece(now: number): number[] {
    const actions: number[] = [];
    const s = this.getSettings();

    if (s.arr === 0) {
      const leftKey = this.keys.ArrowLeft;
      const rightKey = this.keys.ArrowRight;
      let hKey: KeyState | null = null;
      let hBinding: Binding | null = null;
      if (leftKey.pressed && rightKey.pressed) {
        if (leftKey.pressOrder > rightKey.pressOrder) {
          hKey = leftKey;
          hBinding = KEY_MAP.ArrowLeft;
        } else {
          hKey = rightKey;
          hBinding = KEY_MAP.ArrowRight;
        }
      } else if (leftKey.pressed) {
        hKey = leftKey;
        hBinding = KEY_MAP.ArrowLeft;
      } else if (rightKey.pressed) {
        hKey = rightKey;
        hBinding = KEY_MAP.ArrowRight;
      }
      if (hKey && hBinding && hKey.dasTriggered) {
        hKey.lastRepeatTime = now;
        actions.push(hBinding.wallAction!);
      }
    }

    if (s.dropArr === 0) {
      const downKey = this.keys.ArrowDown;
      if (downKey.pressed && downKey.dasTriggered) {
        downKey.lastRepeatTime = now;
        actions.push(KEY_MAP.ArrowDown.wallAction!);
      }
    }

    return actions;
  }
}
