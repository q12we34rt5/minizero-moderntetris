// Serialized engine view layout. Must stay in sync with web/engine/engine_wasm.cpp.

export const BOARD_W = 10;
export const BOARD_H = 20;
export const BOARD_CELLS = BOARD_W * BOARD_H;
export const NEXT_COUNT = 5;

const I_BOARD = 0;
const I_CURRENT = BOARD_CELLS;
const I_ORIENTATION = I_CURRENT + 1;
const I_X = I_CURRENT + 2;
const I_Y = I_CURRENT + 3;
const I_GHOST_Y = I_CURRENT + 4;
const I_HOLD = I_CURRENT + 5;
const I_HAS_HELD = I_CURRENT + 6;
const I_NEXT = I_CURRENT + 7;
const I_IS_ALIVE = I_NEXT + NEXT_COUNT;
const I_PIECE_COUNT = I_IS_ALIVE + 1;
const I_COMBO_COUNT = I_IS_ALIVE + 2;
const I_B2B_COUNT = I_IS_ALIVE + 3;
const I_LINES_CLEARED = I_IS_ALIVE + 4;
const I_ATTACK = I_IS_ALIVE + 5;
const I_LINES_SENT = I_IS_ALIVE + 6;
const I_TOTAL_LINES_CLEARED = I_IS_ALIVE + 7;
const I_TOTAL_ATTACK = I_IS_ALIVE + 8;
const I_TOTAL_LINES_SENT = I_IS_ALIVE + 9;
const I_SPIN_TYPE = I_IS_ALIVE + 10;
const I_SRS_INDEX = I_IS_ALIVE + 11;
const I_PERFECT_CLEAR = I_IS_ALIVE + 12;
const I_PENDING_GARBAGE = I_IS_ALIVE + 13;
const I_LIFETIME = I_IS_ALIVE + 14;

/** Number of int32 slots a serialized view occupies. */
export const VIEW_SIZE = I_LIFETIME + 1;

/** Cell value in the board grid: 0 empty, 1 block, 3 garbage. */
export type CellValue = 0 | 1 | 3;

export interface GameView {
  /** Row-major, BOARD_H rows of BOARD_W cells. Does NOT include the active piece. */
  board: Int32Array;
  current: number; // piece type 0..6, -1 if none
  orientation: number; // 0..3
  x: number; // active piece x, relative to visible board
  y: number; // active piece y, relative to visible board
  ghostY: number; // y of the hard-drop landing position
  hold: number; // piece type 0..6, -1 if none
  hasHeld: boolean;
  next: number[]; // upcoming piece types
  isAlive: boolean;
  pieceCount: number;
  comboCount: number;
  b2bCount: number;
  linesCleared: number; // from the last placement
  attack: number; // from the last placement
  linesSent: number; // from the last placement
  totalLinesCleared: number;
  totalAttack: number;
  totalLinesSent: number;
  spinType: number; // 0 none, 1 spin, 2 mini
  srsIndex: number;
  perfectClear: boolean;
  pendingGarbage: number;
  lifetime: number;
}

/** Parse a serialized view (an Int32Array of length VIEW_SIZE) into a GameView. */
export function parseView(raw: Int32Array): GameView {
  const next: number[] = [];
  for (let i = 0; i < NEXT_COUNT; i++) next.push(raw[I_NEXT + i]);
  return {
    board: raw.slice(I_BOARD, I_BOARD + BOARD_CELLS),
    current: raw[I_CURRENT],
    orientation: raw[I_ORIENTATION],
    x: raw[I_X],
    y: raw[I_Y],
    ghostY: raw[I_GHOST_Y],
    hold: raw[I_HOLD],
    hasHeld: raw[I_HAS_HELD] !== 0,
    next,
    isAlive: raw[I_IS_ALIVE] !== 0,
    pieceCount: raw[I_PIECE_COUNT],
    comboCount: raw[I_COMBO_COUNT],
    b2bCount: raw[I_B2B_COUNT],
    linesCleared: raw[I_LINES_CLEARED],
    attack: raw[I_ATTACK],
    linesSent: raw[I_LINES_SENT],
    totalLinesCleared: raw[I_TOTAL_LINES_CLEARED],
    totalAttack: raw[I_TOTAL_ATTACK],
    totalLinesSent: raw[I_TOTAL_LINES_SENT],
    spinType: raw[I_SPIN_TYPE],
    srsIndex: raw[I_SRS_INDEX],
    perfectClear: raw[I_PERFECT_CLEAR] !== 0,
    pendingGarbage: raw[I_PENDING_GARBAGE],
    lifetime: raw[I_LIFETIME],
  };
}
