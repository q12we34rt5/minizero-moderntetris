import { useLayoutEffect, useRef } from 'react';
import { CELL, GAP, BOARD_PX_H } from '../render/board.ts';
import { BOARD_H } from '../engine/view.ts';

const BAR_W = 14;

// Red, dimmed by each queued garbage entry's delay (mirrors the delay symbols
// in engine::toString): delay 0-1 full strength, 2-3 at 0.7 alpha, 4+ at 0.4.
function garbageColor(delay: number): string {
  const alpha = delay < 2 ? 1 : delay < 4 ? 0.7 : 0.4;
  return `rgba(239, 68, 68, ${alpha})`;
}

interface Props {
  /** Per-entry queued garbage lines, front of the queue first. */
  queue: number[];
  /** Per-entry garbage delay, aligned with queue. */
  delay: number[];
}

/**
 * Vertical pending-garbage meter shown flush against the board's left edge.
 * One red cell per queued garbage line, stacked from the bottom and aligned
 * row-for-row with the board. Each queue entry is tinted by its delay (cf. the
 * garbage column in engine::toString).
 */
export function GarbageBar({ queue, delay }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0c0c18';
    ctx.fillRect(0, 0, BAR_W, BOARD_PX_H);

    let row = BOARD_H - 1; // bottom row, fill upward
    for (let i = 0; i < queue.length && row >= 0; i++) {
      const lines = queue[i];
      if (lines <= 0) break; // queue is compacted: first empty entry ends it
      const color = garbageColor(delay[i] ?? 0);
      for (let k = 0; k < lines && row >= 0; k++, row--) {
        const y = row * (CELL + GAP) + GAP;
        ctx.fillStyle = color;
        ctx.fillRect(1, y, BAR_W - 2, CELL);
        ctx.fillStyle = 'rgba(255,255,255,0.15)';
        ctx.fillRect(1, y, BAR_W - 2, 2);
        ctx.fillStyle = 'rgba(0,0,0,0.2)';
        ctx.fillRect(1, y + CELL - 2, BAR_W - 2, 2);
      }
    }
  }, [queue, delay]);

  return <canvas ref={ref} width={BAR_W} height={BOARD_PX_H} className="garbage-bar" />;
}
