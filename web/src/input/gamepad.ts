// Gamepad input: the browser Gamepad API is poll-based, so this module reads
// button/stick state each frame and maps it onto the same logical inputs the
// keyboard uses. Standard mapping (gamepad.mapping === 'standard') is assumed,
// which is consistent across Xbox / PlayStation controllers.

/** Inputs with press-duration semantics (DAS/ARR applies). */
export type HeldInput = 'left' | 'right' | 'softDrop';
/** Fire-once-on-press inputs. */
export type InstantInput = 'hardDrop' | 'rotateCW' | 'rotateCCW' | 'rotate180' | 'hold';
export type LogicalInput = HeldInput | InstantInput;

export const HELD_INPUTS: HeldInput[] = ['left', 'right', 'softDrop'];
export const INSTANT_INPUTS: InstantInput[] = ['hardDrop', 'rotateCW', 'rotateCCW', 'rotate180', 'hold'];
export const LOGICAL_INPUTS: LogicalInput[] = [...HELD_INPUTS, ...INSTANT_INPUTS];

/** Maps each logical input to a standard-layout gamepad button index. */
export type GamepadMapping = Record<LogicalInput, number>;

// Standard layout: 12-15 = D-pad up/down/left/right, 0-3 = face buttons.
export const DEFAULT_GAMEPAD_MAPPING: GamepadMapping = {
  left: 14, // D-pad left
  right: 15, // D-pad right
  softDrop: 13, // D-pad down
  hardDrop: 12, // D-pad up
  rotateCW: 0, // A / Cross
  rotateCCW: 1, // B / Circle
  rotate180: 2, // X / Square
  hold: 3, // Y / Triangle
};

export const GAMEPAD_INPUT_LABELS: Record<LogicalInput, string> = {
  left: 'Move Left',
  right: 'Move Right',
  softDrop: 'Soft Drop',
  hardDrop: 'Hard Drop',
  rotateCW: 'Rotate CW',
  rotateCCW: 'Rotate CCW',
  rotate180: 'Rotate 180',
  hold: 'Hold',
};

// Left stick is also accepted for movement; past this magnitude it counts as
// a digital press (so the stick behaves like the D-pad).
export const STICK_DEADZONE = 0.5;

export interface GamepadStatus {
  connected: boolean;
  id: string | null;
}

const STANDARD_BUTTON_LABELS: Record<number, string> = {
  0: 'A / ✕',
  1: 'B / ○',
  2: 'X / □',
  3: 'Y / △',
  4: 'LB / L1',
  5: 'RB / R1',
  6: 'LT / L2',
  7: 'RT / R2',
  8: 'Back / Share',
  9: 'Start / Options',
  10: 'L3',
  11: 'R3',
  12: 'D-Pad ↑',
  13: 'D-Pad ↓',
  14: 'D-Pad ←',
  15: 'D-Pad →',
  16: 'Home',
};

export function buttonLabel(index: number): string {
  return STANDARD_BUTTON_LABELS[index] ?? `Button ${index}`;
}

/** First connected gamepad, or null. */
function activeGamepad(): Gamepad | null {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  for (const gp of pads) {
    if (gp) return gp;
  }
  return null;
}

export interface GamepadPollResult {
  held: Record<HeldInput, boolean>;
  /** Instant inputs whose button had a rising edge this poll. */
  instants: InstantInput[];
}

const EMPTY_HELD: Record<HeldInput, boolean> = { left: false, right: false, softDrop: false };

/**
 * Polls the active gamepad and resolves its raw state onto logical inputs.
 * Holds edge-detection state for the instant inputs, so it must be polled
 * every frame. reset() re-seeds that state so a button held across a game
 * reset is not seen as a fresh press.
 */
export class GamepadReader {
  private prevInstant: Record<InstantInput, boolean> = {
    hardDrop: false,
    rotateCW: false,
    rotateCCW: false,
    rotate180: false,
    hold: false,
  };
  // After a reset (or reconnect) the next poll seeds prevInstant without
  // emitting, so held buttons do not produce a spurious instant action.
  private freshStart = true;

  reset(): void {
    this.freshStart = true;
  }

  poll(mapping: GamepadMapping): GamepadPollResult {
    const gp = activeGamepad();
    if (!gp) {
      this.freshStart = true;
      return { held: { ...EMPTY_HELD }, instants: [] };
    }

    const btn = (i: number): boolean => (i >= 0 && i < gp.buttons.length ? gp.buttons[i].pressed : false);
    const axX = gp.axes[0] ?? 0;
    const axY = gp.axes[1] ?? 0;

    const held: Record<HeldInput, boolean> = {
      left: btn(mapping.left) || axX < -STICK_DEADZONE,
      right: btn(mapping.right) || axX > STICK_DEADZONE,
      softDrop: btn(mapping.softDrop) || axY > STICK_DEADZONE,
    };

    const instants: InstantInput[] = [];
    for (const input of INSTANT_INPUTS) {
      const down = btn(mapping[input]);
      if (!this.freshStart && down && !this.prevInstant[input]) instants.push(input);
      this.prevInstant[input] = down;
    }
    this.freshStart = false;

    return { held, instants };
  }
}

/**
 * Watches every connected gamepad for the next button press and reports its
 * index — used by the remap UI. Buttons already held when capture starts are
 * ignored (the first frame only seeds state). Returns a cancel function.
 */
export function captureNextButton(onCapture: (buttonIndex: number) => void): () => void {
  let raf = 0;
  let cancelled = false;
  let prev: boolean[] = [];

  const snapshot = (): boolean[] => {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const gp of pads) {
      if (gp) return gp.buttons.map((b) => b.pressed);
    }
    return [];
  };

  const tick = () => {
    if (cancelled) return;
    const cur = snapshot();
    for (let i = 0; i < cur.length; i++) {
      if (cur[i] && !prev[i]) {
        cancelled = true;
        onCapture(i);
        return;
      }
    }
    prev = cur;
    raf = requestAnimationFrame(tick);
  };

  // Seed prev on the first frame so an already-held button is not captured.
  raf = requestAnimationFrame(() => {
    prev = snapshot();
    raf = requestAnimationFrame(tick);
  });

  return () => {
    cancelled = true;
    cancelAnimationFrame(raf);
  };
}
