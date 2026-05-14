import { useGame } from './game/useGame.ts';
import { BoardPanel, type BoardOverlay } from './components/BoardPanel.tsx';
import { SettingsPanel } from './components/SettingsPanel.tsx';
import { PveSettingsPanel } from './components/PveSettingsPanel.tsx';

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

export default function App() {
  const game = useGame();
  const { mode, status, winner, hud, aiHud } = game;

  const statusLabel =
    status === 'loading' ? 'Loading…' : status === 'playing' ? 'Playing' : 'Game Over';

  // Per-board overlays.
  let playerOverlay: BoardOverlay | null = null;
  let aiOverlay: BoardOverlay | null = null;
  if (status === 'loading') {
    playerOverlay = { title: 'Loading', tone: 'neutral', sub: 'Compiling engine…' };
    aiOverlay = playerOverlay;
  } else if (status === 'gameover') {
    if (mode === 'single') {
      const v = hud?.view;
      playerOverlay = {
        title: 'GAME OVER',
        tone: 'lose',
        sub: v ? `Lines ${v.totalLinesCleared} · Attack ${v.totalAttack} · Pieces ${v.pieceCount}` : undefined,
      };
    } else {
      const playerWon = winner === 'player';
      playerOverlay = { title: playerWon ? 'YOU WIN' : 'YOU LOSE', tone: playerWon ? 'win' : 'lose' };
      aiOverlay = { title: playerWon ? 'AI LOSES' : 'AI WINS', tone: playerWon ? 'lose' : 'win' };
    }
  }

  return (
    <>
      <header>
        <h1>ModernTetris — Web</h1>
        <div className="header-controls">
          <div className="mode-toggle">
            <button
              className={`btn ${mode === 'single' ? 'btn-primary' : ''}`}
              onClick={() => game.setMode('single')}
            >
              Single
            </button>
            <button
              className={`btn ${mode === 'pve' ? 'btn-primary' : ''}`}
              onClick={() => game.setMode('pve')}
            >
              PvE
            </button>
          </div>
          <span className={`status-chip ${status}`}>{statusLabel}</span>
        </div>
      </header>

      <main>
        <BoardPanel label={mode === 'pve' ? 'You' : 'Player'} hud={hud} overlay={playerOverlay} />
        {mode === 'pve' && <BoardPanel label="AI" hud={aiHud} overlay={aiOverlay} />}

        {mode === 'single' && (
          <div className="col-right">
            <div className="panel">
              <div className="panel-title">Controls</div>
              <div className="keybinds">
                {KEYBINDS.map(([k, desc]) => (
                  <Keybind key={desc} k={k} desc={desc} />
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      <div className="bottom-bar">
        <SettingsPanel settings={game.settings} onChange={game.setSettings} />
        {mode === 'pve' && (
          <PveSettingsPanel
            settings={game.pveSettings}
            onChange={game.setPveSettings}
            aiStatus={game.aiStatus}
            onReconnect={game.reconnectAi}
          />
        )}
        <div className="panel" style={{ flex: 1, minWidth: 220 }}>
          <div className="panel-title">Game</div>
          <div className="controls-row">
            <label htmlFor="seed-input">Seed</label>
            <input
              id="seed-input"
              type="text"
              className="seed-input"
              placeholder="random"
              value={game.seed}
              onChange={(e) => game.setSeed(e.target.value)}
            />
            <button className="btn btn-primary" onClick={game.reset}>
              Reset
            </button>
          </div>
          {mode === 'pve' && (
            <p className="hint">
              Both boards share the same piece sequence. Clear lines to send garbage to the AI.
            </p>
          )}
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
