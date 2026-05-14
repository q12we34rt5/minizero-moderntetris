import type { GameMode, PveSettings } from '../game/useGame.ts';
import type { AiConnectionStatus } from '../ai/client.ts';

interface Props {
  mode: GameMode; // 'pve' or 'eve'
  settings: PveSettings;
  onChange: (s: PveSettings) => void;
  aiStatusA: AiConnectionStatus;
  aiStatusB: AiConnectionStatus;
  onReconnect: () => void;
}

export function PveSettingsPanel({ mode, settings, onChange, aiStatusA, aiStatusB, onReconnect }: Props) {
  const updateNum = (key: 'aiIntervalMs' | 'garbageDelay', value: string) => {
    const n = parseInt(value, 10);
    onChange({ ...settings, [key]: Number.isFinite(n) ? n : 0 });
  };
  const isEve = mode === 'eve';

  return (
    <div className="panel" style={{ flex: 1, minWidth: 300 }}>
      <div className="panel-title">{isEve ? 'EvE / AI Backends' : 'PvE / AI Backend'}</div>
      <div className="settings-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="setting-item">
          <label htmlFor="set-ai-interval">AI Interval (ms)</label>
          <input
            id="set-ai-interval"
            type="number"
            min={0}
            max={5000}
            value={settings.aiIntervalMs}
            onChange={(e) => updateNum('aiIntervalMs', e.target.value)}
          />
        </div>
        <div className="setting-item">
          <label htmlFor="set-garbage-delay">Garbage Delay</label>
          <input
            id="set-garbage-delay"
            type="number"
            min={0}
            max={20}
            value={settings.garbageDelay}
            onChange={(e) => updateNum('garbageDelay', e.target.value)}
          />
        </div>
      </div>

      {isEve && (
        <div className="setting-item" style={{ marginTop: 8 }}>
          <label htmlFor="set-backend-a">Board A Backend</label>
          <input
            id="set-backend-a"
            type="text"
            value={settings.backendUrlA}
            onChange={(e) => onChange({ ...settings, backendUrlA: e.target.value })}
          />
        </div>
      )}
      <div className="setting-item" style={{ marginTop: 8 }}>
        <label htmlFor="set-backend-b">{isEve ? 'Board B Backend' : 'AI Backend URL'}</label>
        <input
          id="set-backend-b"
          type="text"
          value={settings.backendUrlB}
          onChange={(e) => onChange({ ...settings, backendUrlB: e.target.value })}
        />
      </div>

      <div className="controls-row">
        {isEve && <span className={`status-chip ${aiStatusA}`}>A: {aiStatusA}</span>}
        <span className={`status-chip ${aiStatusB}`}>{isEve ? 'B' : 'AI'}: {aiStatusB}</span>
        <button className="btn" onClick={onReconnect}>
          Reconnect
        </button>
      </div>
    </div>
  );
}
