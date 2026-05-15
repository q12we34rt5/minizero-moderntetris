import { useGame, type GameMode } from './game/useGame.ts';
import { BoardPanel, type BoardOverlay } from './components/BoardPanel.tsx';
import { SettingsPanel } from './components/SettingsPanel.tsx';
import { PveSettingsPanel } from './components/PveSettingsPanel.tsx';
import { GamepadPanel } from './components/GamepadPanel.tsx';

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

const MODES: { id: GameMode; label: string }[] = [
  { id: 'single', label: 'Single' },
  { id: 'pve', label: 'PvE' },
  { id: 'eve', label: 'EvE' },
];

function boardLabels(mode: GameMode): [string, string] {
  if (mode === 'pve') return ['You', 'AI'];
  if (mode === 'eve') return ['AI A', 'AI B'];
  return ['Player', ''];
}

export default function App() {
  const game = useGame();
  const { mode, status, winner, hud, aiHud } = game;
  const [labelA, labelB] = boardLabels(mode);

  const statusLabel =
    status === 'loading'
      ? 'Loading…'
      : status === 'playing'
        ? 'Playing'
        : status === 'paused'
          ? 'Paused'
          : 'Game Over';

  // Per-board overlays.
  let overlayA: BoardOverlay | null = null;
  let overlayB: BoardOverlay | null = null;
  if (status === 'loading') {
    overlayA = { title: 'Loading', tone: 'neutral', sub: 'Compiling engine…' };
    overlayB = overlayA;
  } else if (status === 'paused') {
    overlayA = { title: 'PAUSED', tone: 'neutral' };
    overlayB = mode === 'single' ? null : overlayA;
  } else if (status === 'gameover') {
    if (mode === 'single') {
      const v = hud?.view;
      overlayA = {
        title: 'GAME OVER',
        tone: 'lose',
        sub: v ? `Lines ${v.totalLinesCleared} · Attack ${v.totalAttack} · Pieces ${v.pieceCount}` : undefined,
      };
    } else {
      overlayA = winnerOverlay(winner === 'A', mode);
      overlayB = winnerOverlay(winner === 'B', mode);
    }
  }

  const showAiPanel = mode === 'pve' || mode === 'eve';
  const canPause = status === 'playing' || status === 'paused';

  return (
    <>
      <header>
        <h1>ModernTetris — Web</h1>
        <div className="header-controls">
          <div className="mode-toggle">
            {MODES.map((m) => (
              <button
                key={m.id}
                className={`btn ${mode === m.id ? 'btn-primary' : ''}`}
                onClick={() => game.setMode(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
          {canPause && (
            <button className="btn" onClick={game.togglePause}>
              {status === 'paused' ? 'Resume' : 'Pause'}
            </button>
          )}
          <span className={`status-chip ${status}`}>{statusLabel}</span>
        </div>
      </header>

      <main>
        <BoardPanel label={labelA} hud={hud} overlay={overlayA} />
        {showAiPanel && <BoardPanel label={labelB} hud={aiHud} overlay={overlayB} />}

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
        {mode !== 'eve' && <SettingsPanel settings={game.settings} onChange={game.setSettings} />}
        {mode !== 'eve' && (
          <GamepadPanel
            status={game.gamepadStatus}
            mapping={game.gamepadMapping}
            onChange={game.setGamepadMapping}
          />
        )}
        {showAiPanel && (
          <PveSettingsPanel
            mode={mode}
            settings={game.pveSettings}
            onChange={game.setPveSettings}
            aiStatusA={game.aiStatusA}
            aiStatusB={game.aiStatusB}
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
          {mode === 'eve' && (
            <p className="hint">
              Two AIs play with different piece sequences. Point the two backend URLs at
              different models for a model-vs-model match.
            </p>
          )}
        </div>
      </div>
    </>
  );
}

function winnerOverlay(won: boolean, mode: GameMode): BoardOverlay {
  if (mode === 'eve') {
    return won ? { title: 'WINNER', tone: 'win' } : { title: 'LOSER', tone: 'lose' };
  }
  // pve, from board A's (the human's) perspective the labels read naturally;
  // board B just gets the opposite.
  return won ? { title: 'WIN', tone: 'win' } : { title: 'LOSE', tone: 'lose' };
}

function Keybind({ k, desc }: { k: string; desc: string }) {
  return (
    <>
      <kbd>{k}</kbd>
      <span>{desc}</span>
    </>
  );
}
