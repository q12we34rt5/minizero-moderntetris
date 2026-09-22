import { useState } from 'react';
import { useGame, type GameMode, type ModelInfo, type PveSettings } from './game/useGame.ts';
import { BoardPanel, type BoardOverlay } from './components/BoardPanel.tsx';
import { SettingsPanel } from './components/SettingsPanel.tsx';
import { PveSettingsPanel } from './components/PveSettingsPanel.tsx';
import { InputBindingsPanel } from './components/InputBindingsPanel.tsx';
import { GAMEPAD_INPUT_LABELS, LOGICAL_INPUTS } from './input/gamepad.ts';
import { keyLabel, type KeyboardMapping } from './input/keyboard.ts';

/** Quick reference of the current keyboard bindings (+ the fixed Reset key). */
function keybinds(mapping: KeyboardMapping): [string, string][] {
  const rows: [string, string][] = LOGICAL_INPUTS.map((input) => [
    keyLabel(mapping[input]),
    GAMEPAD_INPUT_LABELS[input],
  ]);
  rows.push(['R', 'Reset']);
  return rows;
}

const MODES: { id: GameMode; label: string }[] = [
  { id: 'single', label: 'Single' },
  { id: 'pve', label: 'PvE' },
  { id: 'eve', label: 'EvE' },
];

/** Display name for a board's selected model (empty selection = first available). */
function modelLabel(models: ModelInfo[], selectedId: string, fallback: string): string {
  const eff = selectedId || models[0]?.id || '';
  return models.find((m) => m.id === eff)?.displayName ?? fallback;
}

function boardLabels(mode: GameMode, models: ModelInfo[], pve: PveSettings): [string, string] {
  if (mode === 'pve') return ['You', modelLabel(models, pve.pveModel, 'AI')];
  if (mode === 'eve') return [modelLabel(models, pve.eveModelA, 'AI A'), modelLabel(models, pve.eveModelB, 'AI B')];
  return ['Player', ''];
}

export default function App() {
  const game = useGame();
  const { mode, status, winner, hud, aiHud } = game;
  const [labelA, labelB] = boardLabels(mode, game.models, game.pveSettings);
  const [dump, setDump] = useState('');

  const onDump = () => {
    const text = game.dumpState();
    setDump(text);
    void navigator.clipboard?.writeText(text).catch(() => {});
  };

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
                {keybinds(game.keyboardMapping).map(([k, desc]) => (
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
          <InputBindingsPanel
            gamepadStatus={game.gamepadStatus}
            gamepadMapping={game.gamepadMapping}
            onGamepadChange={game.setGamepadMapping}
            gamepadOptions={game.gamepadOptions}
            onGamepadOptionsChange={game.setGamepadOptions}
            keyboardMapping={game.keyboardMapping}
            onKeyboardChange={game.setKeyboardMapping}
          />
        )}
        {showAiPanel && (
          <PveSettingsPanel
            mode={mode}
            settings={game.pveSettings}
            onChange={game.setPveSettings}
            models={game.models}
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
          <div className="controls-row">
            <label htmlFor="set-all-spin">All-Spin ruleset</label>
            <input
              id="set-all-spin"
              type="checkbox"
              checked={game.rules.allSpin}
              onChange={(e) => game.setRules({ ...game.rules, allSpin: e.target.checked })}
            />
          </div>
          <p className="hint">
            All-Spin scores any immobile spin (not just 3-corner T-spins). Must match the
            AI backend's env_modern_tetris_all_spin. Takes effect on Reset.
          </p>
          {mode === 'pve' && (
            <p className="hint">
              Both boards share the same piece sequence. Clear lines to send garbage to the AI.
            </p>
          )}
          {mode === 'eve' && (
            <p className="hint">
              Two AIs play with different piece sequences. Pick a different model for
              each board for a model-vs-model match.
            </p>
          )}
          <div className="controls-row">
            <button className="btn" onClick={onDump}>
              Dump State
            </button>
            <span className="hint">Copies a backend <code>set_state</code> command for the current position.</span>
          </div>
          {dump && (
            <textarea
              className="seed-input"
              readOnly
              rows={4}
              style={{ width: '100%', fontFamily: 'monospace', fontSize: 11, marginTop: 6, resize: 'vertical' }}
              value={dump}
              onFocus={(e) => e.currentTarget.select()}
            />
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
