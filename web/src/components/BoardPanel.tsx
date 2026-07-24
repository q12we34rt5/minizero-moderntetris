import { AiInfoPanel } from './AiInfoPanel.tsx';
import { BoardCanvas } from './BoardCanvas.tsx';
import { GarbageBar } from './GarbageBar.tsx';
import { PiecePreview } from './PiecePreview.tsx';
import type { GameHud } from '../game/useGame.ts';
import { NEXT_COUNT } from '../engine/view.ts';
import { SPIN_NAMES } from '../data/pieces.ts';

export interface BoardOverlay {
  title: string;
  tone: 'win' | 'lose' | 'neutral';
  sub?: string;
}

interface Props {
  label: string;
  hud: GameHud | null;
  overlay?: BoardOverlay | null;
}

function MiniStat({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="stat-item">
      <span className="stat-label">{label}</span>
      <span className={accent ? 'stat-value accent' : 'stat-value'}>{value}</span>
    </div>
  );
}

export function BoardPanel({ label, hud, overlay }: Props) {
  const view = hud?.view ?? null;

  return (
    <div className="board-panel">
      <div className="board-panel-label">{label}</div>
      <div className="board-panel-body">
        <div className="side-col">
          <div className="panel hold-box">
            <div className="panel-title">Hold</div>
            <PiecePreview pieceType={view?.hold ?? -1} width={80} height={50} />
          </div>
          <div className="panel">
            <div className="panel-title">Stats</div>
            <div className="stat-grid" style={{ gridTemplateColumns: '1fr' }}>
              <MiniStat label="Lines" value={view?.totalLinesCleared ?? 0} />
              <MiniStat label="Attack" value={view?.totalAttack ?? 0} />
              <MiniStat label="Combo" value={view?.comboCount ?? 0} />
              <MiniStat label="B2B" value={view?.b2bCount ?? 0} />
              <MiniStat label="PPS" value={(hud?.pps ?? 0).toFixed(2)} />
              <MiniStat label="APM" value={(hud?.apm ?? 0).toFixed(1)} />
              <MiniStat
                label="Spin"
                value={SPIN_NAMES[view?.spinType ?? 0] ?? 'NONE'}
                accent={(view?.spinType ?? 0) > 0}
              />
            </div>
          </div>
        </div>

        <div className="board-stack">
          <GarbageBar queue={view?.garbageQueue ?? []} delay={view?.garbageDelay ?? []} />
          <div className="board-wrapper">
            <BoardCanvas view={view} />
            {overlay && (
              <div className={`overlay ${overlay.tone}`}>
                <h2>{overlay.title}</h2>
                {overlay.sub && <p>{overlay.sub}</p>}
              </div>
            )}
          </div>
        </div>

        <div className="side-col">
          <div className="panel">
            <div className="panel-title">Next</div>
            <div className="next-queue">
              {Array.from({ length: NEXT_COUNT }, (_, i) => (
                <PiecePreview key={i} pieceType={view?.next[i] ?? -1} width={80} height={44} />
              ))}
            </div>
          </div>
          <AiInfoPanel info={hud?.aiInfo ?? {}} />
        </div>
      </div>
    </div>
  );
}
