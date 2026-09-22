// Source-agnostic input controller. Keyboard and gamepad both feed the same
// set of logical inputs (left / right / softDrop held, plus instant actions);
// the DAS/ARR/gravity timing — carefully tuned for game feel and ported from
// TetRL-Arena — runs once over that merged state.

import {
  GamepadReader,
  HELD_INPUTS,
  type GamepadMapping,
  type GamepadOptions,
  type GamepadStatus,
  type HeldInput,
  type InstantInput,
  type LogicalInput,
} from './gamepad.ts';
import { inputForKey, normalizeKey, RESET_KEY, type KeyboardMapping } from './keyboard.ts';

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

export type { GamepadStatus };

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

interface HeldMeta {
  action: number;
  wallAction: number;
  kind: 'horizontal' | 'vertical';
}

const HELD_META: Record<HeldInput, HeldMeta> = {
  left: { action: Action.MOVE_LEFT, wallAction: Action.MOVE_LEFT_TO_WALL, kind: 'horizontal' },
  right: { action: Action.MOVE_RIGHT, wallAction: Action.MOVE_RIGHT_TO_WALL, kind: 'horizontal' },
  softDrop: { action: Action.SOFT_DROP, wallAction: Action.SOFT_DROP_TO_FLOOR, kind: 'vertical' },
};

const INSTANT_ACTION: Record<InstantInput, number> = {
  hardDrop: Action.HARD_DROP,
  rotateCW: Action.ROTATE_CW,
  rotateCCW: Action.ROTATE_CCW,
  rotate180: Action.ROTATE_180,
  hold: Action.HOLD,
};

function isHeld(input: LogicalInput): input is HeldInput {
  return input === 'left' || input === 'right' || input === 'softDrop';
}

/** Press state for one held logical input, with merged keyboard + gamepad sources. */
interface HeldState {
  kbPressed: boolean;
  padPressed: boolean;
  pressed: boolean; // kbPressed || padPressed
  pressTime: number;
  dasTriggered: boolean;
  lastRepeatTime: number;
  pressOrder: number;
}

function newHeldState(): HeldState {
  return {
    kbPressed: false,
    padPressed: false,
    pressed: false,
    pressTime: 0,
    dasTriggered: false,
    lastRepeatTime: 0,
    pressOrder: 0,
  };
}

export interface InputControllerOptions {
  getSettings: () => InputSettings;
  getGamepadMapping: () => GamepadMapping;
  getGamepadOptions: () => GamepadOptions;
  getKeyboardMapping: () => KeyboardMapping;
  onReset?: () => void;
  onGamepadStatus?: (status: GamepadStatus) => void;
}

export class InputController {
  private readonly getSettings: () => InputSettings;
  private readonly getGamepadMapping: () => GamepadMapping;
  private readonly getGamepadOptions: () => GamepadOptions;
  private readonly getKeyboardMapping: () => KeyboardMapping;
  private readonly onReset?: () => void;
  private readonly onGamepadStatus?: (status: GamepadStatus) => void;

  private readonly held: Record<HeldInput, HeldState> = {
    left: newHeldState(),
    right: newHeldState(),
    softDrop: newHeldState(),
  };
  private pending: number[] = [];
  private pressCounter = 0;
  private prevHeld: HeldState | null = null;
  private lastGravityTime = 0;
  private enabled = false;
  private readonly gamepad = new GamepadReader();

  constructor(opts: InputControllerOptions) {
    this.getSettings = opts.getSettings;
    this.getGamepadMapping = opts.getGamepadMapping;
    this.getGamepadOptions = opts.getGamepadOptions;
    this.getKeyboardMapping = opts.getKeyboardMapping;
    this.onReset = opts.onReset;
    this.onGamepadStatus = opts.onGamepadStatus;
  }

  attach() {
    window.addEventListener('keydown', this.onKeyDown, { passive: false });
    window.addEventListener('keyup', this.onKeyUp, { passive: false });
    window.addEventListener('gamepadconnected', this.onGamepadChange);
    window.addEventListener('gamepaddisconnected', this.onGamepadChange);
    this.refreshGamepadStatus(); // report a gamepad that is already connected
  }

  detach() {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('gamepadconnected', this.onGamepadChange);
    window.removeEventListener('gamepaddisconnected', this.onGamepadChange);
  }

  /** Enable/disable game-action capture. Reset key still works when disabled. */
  setEnabled(enabled: boolean) {
    this.enabled = enabled;
  }

  /** Clear all transient input state. Call on game (re)start. */
  reset(now: number) {
    for (const input of HELD_INPUTS) this.held[input] = newHeldState();
    this.pending.length = 0;
    this.prevHeld = null;
    this.pressCounter = 0;
    this.lastGravityTime = now;
    this.gamepad.reset();
  }

  private onGamepadChange = () => {
    this.refreshGamepadStatus();
  };

  private refreshGamepadStatus() {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    let id: string | null = null;
    for (const gp of pads) {
      if (gp) {
        id = gp.id;
        break;
      }
    }
    this.onGamepadStatus?.({ connected: id !== null, id });
  }

  private onKeyDown = (e: KeyboardEvent) => {
    if (e.repeat) return;
    if (e.target instanceof HTMLElement && e.target.tagName === 'INPUT') return;

    const key = normalizeKey(e.key);
    if (key === RESET_KEY) {
      e.preventDefault();
      this.onReset?.();
      return;
    }

    const input = inputForKey(this.getKeyboardMapping(), key);
    if (!input) return;
    e.preventDefault();
    if (!this.enabled) return;

    if (isHeld(input)) {
      this.setHeld(input, 'kb', true, performance.now());
    } else {
      this.pending.push(INSTANT_ACTION[input]);
    }
  };

  private onKeyUp = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLElement && e.target.tagName === 'INPUT') return;
    const input = inputForKey(this.getKeyboardMapping(), normalizeKey(e.key));
    if (!input) return;
    e.preventDefault();
    if (isHeld(input)) this.setHeld(input, 'kb', false, performance.now());
  };

  /** Update one source of a held input; fires the initial press on a rising edge. */
  private setHeld(input: HeldInput, source: 'kb' | 'pad', down: boolean, now: number) {
    const s = this.held[input];
    if (source === 'kb') s.kbPressed = down;
    else s.padPressed = down;

    const nowPressed = s.kbPressed || s.padPressed;
    if (nowPressed && !s.pressed) {
      s.pressed = true;
      s.pressTime = now;
      s.dasTriggered = false;
      s.lastRepeatTime = 0;
      s.pressOrder = ++this.pressCounter;
      // fire the initial press immediately
      const settings = this.getSettings();
      const meta = HELD_META[input];
      const das = meta.kind === 'horizontal' ? settings.das : settings.dropDas;
      this.pending.push(das === 0 ? meta.wallAction : meta.action);
    } else if (!nowPressed && s.pressed) {
      s.pressed = false;
    }
  }

  /** Poll the gamepad onto the logical inputs. Edges are always consumed so a
   *  press during a disabled period does not leak in when play resumes. */
  private pollGamepad(now: number) {
    const raw = this.gamepad.poll(this.getGamepadMapping());
    if (!this.enabled) return;
    for (const input of HELD_INPUTS) this.setHeld(input, 'pad', raw.held[input], now);

    // Anti-misfire: optionally drop a hard drop that fires while any other
    // gamepad input (a held direction or another button this poll) is active,
    // so you don't accidentally slam the piece down mid-movement.
    const blockHardDrop =
      this.getGamepadOptions().blockHardDropWhileInput &&
      (raw.held.left ||
        raw.held.right ||
        raw.held.softDrop ||
        raw.instants.some((i) => i !== 'hardDrop'));

    for (const inst of raw.instants) {
      if (inst === 'hardDrop' && blockHardDrop) continue;
      this.pending.push(INSTANT_ACTION[inst]);
    }
  }

  /** Collect all actions for this frame: queued instants + DAS/ARR repeats + gravity. */
  collect(now: number): number[] {
    this.pollGamepad(now);

    const actions: number[] = [];
    const s = this.getSettings();

    // 1. queued instant + initial-press actions
    while (this.pending.length > 0) actions.push(this.pending.shift()!);

    // 2. horizontal movement (DAS/ARR)
    const leftKey = this.held.left;
    const rightKey = this.held.right;
    let hKey: HeldState | null = null;
    let hMeta: HeldMeta | null = null;
    if (leftKey.pressed && rightKey.pressed) {
      if (leftKey.pressOrder > rightKey.pressOrder) {
        hKey = leftKey;
        hMeta = HELD_META.left;
      } else {
        hKey = rightKey;
        hMeta = HELD_META.right;
      }
    } else if (leftKey.pressed) {
      hKey = leftKey;
      hMeta = HELD_META.left;
    } else if (rightKey.pressed) {
      hKey = rightKey;
      hMeta = HELD_META.right;
    }

    if (hKey && hMeta) {
      // priority key changed to one that is already charged
      if (hKey !== this.prevHeld && hKey.dasTriggered) {
        hKey.lastRepeatTime = now;
        actions.push(s.arr === 0 ? hMeta.wallAction : hMeta.action);
      }

      const elapsed = now - hKey.pressTime;
      if (!hKey.dasTriggered && elapsed >= s.das) {
        hKey.dasTriggered = true;
        hKey.lastRepeatTime = now;
        if (s.das > 0 || elapsed > 0) {
          actions.push(s.arr === 0 ? hMeta.wallAction : hMeta.action);
        }
      }
      if (hKey.dasTriggered && s.arr > 0) {
        while (now - hKey.lastRepeatTime >= s.arr) {
          hKey.lastRepeatTime += s.arr;
          actions.push(hMeta.action);
        }
      }
    }
    this.prevHeld = hKey;

    // 3. vertical soft drop (Drop DAS/ARR)
    const downKey = this.held.softDrop;
    const downMeta = HELD_META.softDrop;
    if (downKey.pressed) {
      const elapsed = now - downKey.pressTime;
      if (!downKey.dasTriggered && elapsed >= s.dropDas) {
        downKey.dasTriggered = true;
        downKey.lastRepeatTime = downKey.pressTime + s.dropDas;
        if (s.dropDas > 0 || elapsed > 0) {
          actions.push(s.dropArr === 0 ? downMeta.wallAction : downMeta.action);
        }
      }
      if (downKey.dasTriggered && s.dropArr > 0) {
        while (now - downKey.lastRepeatTime >= s.dropArr) {
          downKey.lastRepeatTime += s.dropArr;
          actions.push(downMeta.action);
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
    if (hKey && hMeta && hKey.dasTriggered && s.arr === 0 && anyManual) {
      if (!actions.includes(hMeta.wallAction)) actions.push(hMeta.wallAction);
    }
    if (downKey.pressed && downKey.dasTriggered && s.dropArr === 0 && anyManual) {
      if (!actions.includes(downMeta.wallAction)) actions.push(downMeta.wallAction);
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
      const leftKey = this.held.left;
      const rightKey = this.held.right;
      let hKey: HeldState | null = null;
      let hMeta: HeldMeta | null = null;
      if (leftKey.pressed && rightKey.pressed) {
        if (leftKey.pressOrder > rightKey.pressOrder) {
          hKey = leftKey;
          hMeta = HELD_META.left;
        } else {
          hKey = rightKey;
          hMeta = HELD_META.right;
        }
      } else if (leftKey.pressed) {
        hKey = leftKey;
        hMeta = HELD_META.left;
      } else if (rightKey.pressed) {
        hKey = rightKey;
        hMeta = HELD_META.right;
      }
      if (hKey && hMeta && hKey.dasTriggered) {
        hKey.lastRepeatTime = now;
        actions.push(hMeta.wallAction);
      }
    }

    if (s.dropArr === 0) {
      const downKey = this.held.softDrop;
      if (downKey.pressed && downKey.dasTriggered) {
        downKey.lastRepeatTime = now;
        actions.push(HELD_META.softDrop.wallAction);
      }
    }

    return actions;
  }
}
