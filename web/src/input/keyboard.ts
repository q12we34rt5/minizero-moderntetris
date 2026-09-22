// Keyboard bindings: logical input -> physical key (KeyboardEvent.key, with
// single characters lowercased). Mirrors the gamepad mapping so keys are
// remappable from the UI, and both feed the same logical inputs in the
// controller (keyboard + gamepad play simultaneously).

import { LOGICAL_INPUTS, type LogicalInput } from './gamepad.ts';

export type KeyboardMapping = Record<LogicalInput, string>;

export const DEFAULT_KEYBOARD_MAPPING: KeyboardMapping = {
  left: 'ArrowLeft',
  right: 'ArrowRight',
  softDrop: 'ArrowDown',
  hardDrop: ' ',
  rotateCW: 'ArrowUp',
  rotateCCW: 'z',
  rotate180: 'a',
  hold: 'c',
};

/** Reset is bound to R and is not remappable (kept clear of game actions). */
export const RESET_KEY = 'r';

/** Canonical form of a physical key: single chars are case-insensitive. */
export function normalizeKey(key: string): string {
  return key.length === 1 ? key.toLowerCase() : key;
}

const KEY_LABELS: Record<string, string> = {
  ' ': 'Space',
  ArrowLeft: '←',
  ArrowRight: '→',
  ArrowUp: '↑',
  ArrowDown: '↓',
  Escape: 'Esc',
  Enter: 'Enter',
  Tab: 'Tab',
  Backspace: 'Bksp',
};

/** Human-readable label for a physical key. */
export function keyLabel(key: string): string {
  if (KEY_LABELS[key]) return KEY_LABELS[key];
  return key.length === 1 ? key.toUpperCase() : key;
}

/** Reverse lookup: physical key -> logical input, or null. */
export function inputForKey(mapping: KeyboardMapping, key: string): LogicalInput | null {
  for (const input of LOGICAL_INPUTS) {
    if (mapping[input] === key) return input;
  }
  return null;
}

/**
 * Capture the next key press for the remap UI. Ignores bare modifier keys.
 * Consumes the event (preventDefault + stopPropagation) so the captured press
 * never leaks into gameplay. Returns a cancel function.
 */
export function captureNextKey(onCapture: (key: string) => void): () => void {
  let cancelled = false;
  const MODIFIERS = new Set(['Shift', 'Control', 'Alt', 'Meta']);

  const handler = (e: KeyboardEvent) => {
    if (MODIFIERS.has(e.key)) return; // wait for a real key
    e.preventDefault();
    e.stopPropagation();
    cleanup();
    onCapture(normalizeKey(e.key));
  };

  const cleanup = () => {
    if (cancelled) return;
    cancelled = true;
    window.removeEventListener('keydown', handler, true);
  };

  // Capture phase so we intercept before the controller's window keydown.
  window.addEventListener('keydown', handler, true);
  return cleanup;
}
