import { useLayoutEffect, useRef } from 'react';
import type { GameView } from '../engine/view.ts';
import { drawBoard, BOARD_PX_W, BOARD_PX_H } from '../render/board.ts';

interface Props {
  view: GameView | null;
  /** Per-cell piece colors for locked blocks; null falls back to a flat color. */
  colors?: Int8Array | null;
}

export function BoardCanvas({ view, colors = null }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !view) return;
    const ctx = canvas.getContext('2d');
    if (ctx) drawBoard(ctx, view, colors);
  }, [view, colors]);

  return <canvas ref={canvasRef} width={BOARD_PX_W} height={BOARD_PX_H} />;
}
