// Per-cell colors for locked blocks.
//
// The engine board stores only EMPTY/BLOCK/GARBAGE (2 bits per cell, see
// engine/tetris.hpp), so a serialized view cannot say which tetromino a locked
// cell came from. This mirrors the engine's board mutations on the JS side to
// keep a parallel color plane:
//
//   - a piece locks only in hardDrop() (tetris.cpp), and the web always issues
//     that step itself (piece_life and auto_drop are disabled), so the landing
//     position is known from the view read right before the step;
//   - processPiecePlacement() then clears full rows, which is replayed here;
//   - garbage is pushed in afterwards -- the shift amount is not exposed, so it
//     is recovered by matching the predicted occupancy against the real board.
//
// Every sync ends by checking the prediction against the engine's own board and
// falling back to a rebuild when it disagrees, so the colors can never drift
// into showing blocks where the engine has none (worst case: flat colors).

import { BOARD_W, BOARD_H, type GameView } from '../engine/view.ts';
import { PIECES } from '../data/pieces.ts';

// Rows above the visible board are tracked too: pieces spawn at engine row
// BOARD_TOP - 1, and line clears / garbage shifts move those cells into view.
const HIDDEN_ROWS = 9; // engine BOARD_TOP
const ROWS = HIDDEN_ROWS + BOARD_H; // engine rows 0..BOARD_BOTTOM
const CELLS = ROWS * BOARD_W;
const VISIBLE = HIDDEN_ROWS * BOARD_W; // index of the first visible cell

/** Cell color id: -1 empty, 0..6 piece type, 7 garbage, 8 unknown block. */
export const COLOR_EMPTY = -1;
export const COLOR_GARBAGE = 7;
export const COLOR_UNKNOWN = 8;

interface PendingLock {
  type: number;
  orientation: number;
  x: number; // visible-board coordinates, as in GameView
  y: number;
}

export class ColorTracker {
  private readonly grid = new Int8Array(CELLS).fill(COLOR_EMPTY);
  private readonly scratch = new Int8Array(CELLS);
  private pending: PendingLock | null = null;

  reset(): void {
    this.grid.fill(COLOR_EMPTY);
    this.pending = null;
  }

  /** Visible-board colors, row-major BOARD_H x BOARD_W, aligned with view.board. */
  visible(): Int8Array {
    return this.grid.subarray(VISIBLE);
  }

  /** Call right before issuing a HARD_DROP step: records where the piece lands. */
  noteLock(view: GameView): void {
    this.pending =
      view.current >= 0 && view.isAlive
        ? { type: view.current, orientation: view.orientation, x: view.x, y: view.ghostY }
        : null;
  }

  /** Call right after the HARD_DROP step, with the resulting view. */
  sync(view: GameView): void {
    const lock = this.pending;
    this.pending = null;
    if (lock) this.paint(lock);
    this.clearFullRows();
    const board = view.board;
    if (this.occupancyMatches(this.grid, board)) {
      this.relabel(board);
      return;
    }
    for (let k = 1; k <= BOARD_H; k++) {
      if (this.tryGarbageShift(k, board)) {
        this.relabel(board);
        return;
      }
    }
    this.rebuild(board);
  }

  /** Paint the locked piece's cells with its piece type. */
  private paint(lock: PendingLock): void {
    const shape = PIECES[lock.type][lock.orientation];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        if (!shape[r][c]) continue;
        const x = lock.x + c;
        const y = lock.y + r + HIDDEN_ROWS;
        if (x < 0 || x >= BOARD_W || y < 0 || y >= ROWS) continue;
        this.grid[y * BOARD_W + x] = lock.type;
      }
    }
  }

  /** Drop out full rows, letting everything above fall (mirrors clearLines()). */
  private clearFullRows(): void {
    const g = this.grid;
    let write = ROWS - 1;
    for (let r = ROWS - 1; r >= 0; r--) {
      let full = true;
      for (let c = 0; c < BOARD_W; c++) {
        if (g[r * BOARD_W + c] === COLOR_EMPTY) {
          full = false;
          break;
        }
      }
      if (full) continue;
      if (write !== r) g.copyWithin(write * BOARD_W, r * BOARD_W, (r + 1) * BOARD_W);
      write--;
    }
    for (; write >= 0; write--) g.fill(COLOR_EMPTY, write * BOARD_W, (write + 1) * BOARD_W);
  }

  /**
   * Try the transform applyGarbage() performs for `lines` garbage rows: shift
   * everything up, then refill the bottom rows from the engine's own board (so
   * the hole columns come out exact). Commits only if the result matches.
   */
  private tryGarbageShift(lines: number, board: Int32Array): boolean {
    const s = this.scratch;
    s.set(this.grid.subarray(lines * BOARD_W));
    s.fill(COLOR_EMPTY, CELLS - lines * BOARD_W);
    for (let r = ROWS - lines; r < ROWS; r++) {
      for (let c = 0; c < BOARD_W; c++) {
        const i = (r - HIDDEN_ROWS) * BOARD_W + c;
        s[r * BOARD_W + c] = board[i] !== 0 ? COLOR_GARBAGE : COLOR_EMPTY;
      }
    }
    if (!this.occupancyMatches(s, board)) return false;
    this.grid.set(s);
    return true;
  }

  /** Whether a color plane's visible rows agree with the engine board's occupancy. */
  private occupancyMatches(grid: Int8Array, board: Int32Array): boolean {
    for (let i = 0; i < BOARD_H * BOARD_W; i++) {
      if ((grid[VISIBLE + i] !== COLOR_EMPTY) !== (board[i] !== 0)) return false;
    }
    return true;
  }

  /** Keep the garbage/block distinction in step with the engine's own flags. */
  private relabel(board: Int32Array): void {
    const g = this.grid;
    for (let i = 0; i < BOARD_H * BOARD_W; i++) {
      const cell = board[i];
      if (cell === 3) {
        g[VISIBLE + i] = COLOR_GARBAGE;
      } else if (cell !== 0 && g[VISIBLE + i] === COLOR_GARBAGE) {
        g[VISIBLE + i] = COLOR_UNKNOWN;
      }
    }
  }

  /** Last resort: take occupancy from the engine and give up on piece colors. */
  private rebuild(board: Int32Array): void {
    this.grid.fill(COLOR_EMPTY);
    for (let i = 0; i < BOARD_H * BOARD_W; i++) {
      const cell = board[i];
      if (cell === 0) continue;
      this.grid[VISIBLE + i] = cell === 3 ? COLOR_GARBAGE : COLOR_UNKNOWN;
    }
  }
}
