import { useLayoutEffect, useRef } from 'react';
import { drawPiecePreview } from '../render/board.ts';

interface Props {
  pieceType: number; // 0..6, -1 for empty
  width?: number;
  height?: number;
}

export function PiecePreview({ pieceType, width = 80, height = 50 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (ctx) drawPiecePreview(ctx, pieceType);
  }, [pieceType, width, height]);

  return <canvas ref={canvasRef} width={width} height={height} />;
}
