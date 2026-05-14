import { useGame } from './game/useGame.ts';
import { BoardCanvas } from './components/BoardCanvas.tsx';
import { PiecePreview } from './components/PiecePreview.tsx';
import { SettingsPanel } from './components/SettingsPanel.tsx';
import { NEXT_COUNT } from './engine/view.ts';
import { SPIN_NAMES } from './data/pieces.ts';

const KEYBINDS: [string, string][] = [
  ['←', 'Move Left'],
  ['→', 'Move Right'],
  ['↓', 'Soft Drop'],
  ['Space', 'Hard Drop'],
  ['Z', 'Rotate CCW'],
  ['↑', 'Rotate CW'],
  ['A', 'Rotate 180'],
  ['C', 'Hold'],
  ['R', 'Reset'],
];

function Stat({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="stat-item">
      <span className="stat-label">{label}</span>
      <span className={accent ? 'stat-value accent' : 'stat-value'}>{value}</span>
    </div>
  );
}

export default function App() {
  const { status, hud, settings, setSettings, seed, setSeed, reset } = useGame();
  const view = hud?.view ?? null;

  const statusLabel =
    status === 'loading' ? 'Loading…' : status === 'playing' ? 'Playing' : 'Game Over';

  return (
    <>
      <header>
        <h1>ModernTetris — Web</h1>
        <span className={`status-chip ${status}`}>{statusLabel}</span>
      </header>

      <main>
        <div className="col-left">
          <div className="panel hold-box">
            <div className="panel-title">Hold</div>
            <PiecePreview pieceType={view?.hold ?? -1} width={88} height={56} />
          </div>

          <div className="panel">
            <div className="panel-title">Run Stats</div>
            <div className="stat-grid" style={{ gridTemplateColumns: '1fr' }}>
              <Stat label="PPS" value={(hud?.pps ?? 0).toFixed(2)} />
              <Stat label="APM" value={(hud?.apm ?? 0).toFixed(2)} />
              <Stat label="Lines" value={view?.totalLinesCleared ?? 0} />
              <Stat label="Attack" value={view?.totalAttack ?? 0} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">Controls</div>
            <div className="keybinds">
              {KEYBINDS.map(([k, desc]) => (
                <Keybind key={desc} k={k} desc={desc} />
              ))}
            </div>
          </div>
        </div>

        <div className="center-column">
          <div className="board-wrapper">
            <BoardCanvas view={view} />
            {status === 'loading' && (
              <div className="overlay loading">
                <h2>Loading</h2>
                <p>Compiling engine…</p>
              </div>
            )}
            {status === 'gameover' && view && (
              <div className="overlay">
                <h2>GAME OVER</h2>
                <p>
                  Lines: {view.totalLinesCleared} · Attack: {view.totalAttack} · Pieces:{' '}
                  {view.pieceCount}
                </p>
                <button className="btn btn-primary" onClick={reset}>
                  Play Again (R)
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="col-right">
          <div className="panel">
            <div className="panel-title">Next</div>
            <div className="next-queue">
              {Array.from({ length: NEXT_COUNT }, (_, i) => (
                <PiecePreview key={i} pieceType={view?.next[i] ?? -1} />
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">Statistics</div>
            <div className="stat-grid">
              <Stat label="Lines" value={view?.totalLinesCleared ?? 0} />
              <Stat label="Attack" value={view?.totalAttack ?? 0} />
              <Stat label="Combo" value={view?.comboCount ?? 0} />
              <Stat label="B2B" value={view?.b2bCount ?? 0} />
              <Stat
                label="Spin"
                value={SPIN_NAMES[view?.spinType ?? 0] ?? 'NONE'}
                accent={(view?.spinType ?? 0) > 0}
              />
              <Stat label="SRS Idx" value={view?.srsIndex ?? 0} />
              <Stat label="Pieces" value={view?.pieceCount ?? 0} />
              <Stat
                label="Garbage"
                value={view?.pendingGarbage ?? 0}
                accent={(view?.pendingGarbage ?? 0) > 0}
              />
            </div>
          </div>
        </div>
      </main>

      <div className="bottom-bar">
        <SettingsPanel settings={settings} onChange={setSettings} />
        <div className="panel" style={{ flex: 1, minWidth: 260 }}>
          <div className="panel-title">Game</div>
          <div className="controls-row">
            <label htmlFor="seed-input">Seed</label>
            <input
              id="seed-input"
              type="text"
              className="seed-input"
              placeholder="random"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
            <button className="btn btn-primary" onClick={reset}>
              Reset
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

function Keybind({ k, desc }: { k: string; desc: string }) {
  return (
    <>
      <kbd>{k}</kbd>
      <span>{desc}</span>
    </>
  );
}
