// Canvas rendering for the playfield and piece previews.
// Ported from TetRL-Arena/web/static/index.html.

import { BOARD_W, BOARD_H, type GameView } from '../engine/view.ts';
import {
  PIECES,
  PIECE_COLORS,
  PIECE_COLORS_DIM,
  GARBAGE_COLOR,
  BLOCK_COLOR,
} from '../data/pieces.ts';

export const CELL = 28;
export const GAP = 1;
export const BOARD_PX_W = BOARD_W * CELL + (BOARD_W + 1) * GAP;
export const BOARD_PX_H = BOARD_H * CELL + (BOARD_H + 1) * GAP;

const cellX = (c: number) => c * (CELL + GAP) + GAP;
const cellY = (r: number) => r * (CELL + GAP) + GAP;

function fillCell(ctx: CanvasRenderingContext2D, c: number, r: number, color: string) {
  const x = cellX(c);
  const y = cellY(r);
  ctx.fillStyle = color;
  ctx.fillRect(x, y, CELL, CELL);
  // top/left sheen
  ctx.fillStyle = 'rgba(255,255,255,0.12)';
  ctx.fillRect(x, y, CELL, 2);
  ctx.fillRect(x, y, 2, CELL);
  // bottom/right shade
  ctx.fillStyle = 'rgba(0,0,0,0.15)';
  ctx.fillRect(x + CELL - 2, y, 2, CELL);
  ctx.fillRect(x, y + CELL - 2, CELL, 2);
}

/**
 * Draw the playfield: board cells, ghost piece, then the active piece.
 * `colors` (see game/color-tracker.ts) gives the piece type each locked cell
 * came from; occupancy always comes from the view, so a null/stale color plane
 * only costs the per-piece tint.
 */
export function drawBoard(
  ctx: CanvasRenderingContext2D,
  view: GameView,
  colors: Int8Array | null = null,
) {
  ctx.fillStyle = '#0c0c18';
  ctx.fillRect(0, 0, BOARD_PX_W, BOARD_PX_H);

  // grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.03)';
  ctx.lineWidth = 1;
  for (let r = 0; r <= BOARD_H; r++) {
    const y = r * (CELL + GAP) + GAP;
    ctx.beginPath();
    ctx.moveTo(0, y - 0.5);
    ctx.lineTo(BOARD_PX_W, y - 0.5);
    ctx.stroke();
  }
  for (let c = 0; c <= BOARD_W; c++) {
    const x = c * (CELL + GAP) + GAP;
    ctx.beginPath();
    ctx.moveTo(x - 0.5, 0);
    ctx.lineTo(x - 0.5, BOARD_PX_H);
    ctx.stroke();
  }

  // placed cells
  for (let r = 0; r < BOARD_H; r++) {
    for (let c = 0; c < BOARD_W; c++) {
      const i = r * BOARD_W + c;
      const cell = view.board[i];
      if (cell === 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.015)';
        ctx.fillRect(cellX(c), cellY(r), CELL, CELL);
      } else if (cell === 3) {
        fillCell(ctx, c, r, GARBAGE_COLOR);
      } else {
        const type = colors ? colors[i] : -1;
        fillCell(ctx, c, r, type >= 0 && type < PIECE_COLORS.length ? PIECE_COLORS[type] : BLOCK_COLOR);
      }
    }
  }

  // active + ghost piece
  if (view.current >= 0 && view.isAlive) {
    const shape = PIECES[view.current][view.orientation];

    // ghost
    if (view.ghostY !== view.y) {
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          if (!shape[r][c]) continue;
          const bx = view.x + c;
          const by = view.ghostY + r;
          if (bx < 0 || bx >= BOARD_W || by < 0 || by >= BOARD_H) continue;
          const x = cellX(bx);
          const y = cellY(by);
          ctx.fillStyle = PIECE_COLORS_DIM[view.current];
          ctx.fillRect(x, y, CELL, CELL);
          ctx.strokeStyle = PIECE_COLORS[view.current];
          ctx.globalAlpha = 0.3;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
          ctx.globalAlpha = 1;
        }
      }
    }

    // active
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        if (!shape[r][c]) continue;
        const bx = view.x + c;
        const by = view.y + r;
        if (bx < 0 || bx >= BOARD_W || by < 0 || by >= BOARD_H) continue;
        fillCell(ctx, bx, by, PIECE_COLORS[view.current]);
      }
    }
  }
}

/** Draw a single piece centered on a small preview canvas (hold / next). */
export function drawPiecePreview(ctx: CanvasRenderingContext2D, pieceType: number) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (pieceType < 0 || pieceType > 6) return;

  const shape = PIECES[pieceType][0];
  const color = PIECE_COLORS[pieceType];
  const size = 16;
  const gap = 1;

  let minR = 4;
  let maxR = 0;
  let minC = 4;
  let maxC = 0;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      if (!shape[r][c]) continue;
      minR = Math.min(minR, r);
      maxR = Math.max(maxR, r);
      minC = Math.min(minC, c);
      maxC = Math.max(maxC, c);
    }
  }
  const pw = (maxC - minC + 1) * (size + gap) + gap;
  const ph = (maxR - minR + 1) * (size + gap) + gap;
  const ox = (w - pw) / 2;
  const oy = (h - ph) / 2;

  for (let r = minR; r <= maxR; r++) {
    for (let c = minC; c <= maxC; c++) {
      if (!shape[r][c]) continue;
      const x = ox + (c - minC) * (size + gap) + gap;
      const y = oy + (r - minR) * (size + gap) + gap;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, size, size);
      ctx.fillStyle = 'rgba(255,255,255,0.15)';
      ctx.fillRect(x, y, size, 2);
      ctx.fillRect(x, y, 2, size);
    }
  }
}
