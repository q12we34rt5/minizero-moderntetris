import type { PveSettings } from '../game/useGame.ts';
import type { AiConnectionStatus } from '../ai/client.ts';

interface Props {
  settings: PveSettings;
  onChange: (s: PveSettings) => void;
  aiStatus: AiConnectionStatus;
  onReconnect: () => void;
}

export function PveSettingsPanel({ settings, onChange, aiStatus, onReconnect }: Props) {
  const updateNum = (key: 'aiIntervalMs' | 'garbageDelay', value: string) => {
    const n = parseInt(value, 10);
    onChange({ ...settings, [key]: Number.isFinite(n) ? n : 0 });
  };

  return (
    <div className="panel" style={{ flex: 1, minWidth: 300 }}>
      <div className="panel-title">PvE / AI Backend</div>
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
      <div className="setting-item" style={{ marginTop: 8 }}>
        <label htmlFor="set-backend-url">Backend URL</label>
        <input
          id="set-backend-url"
          type="text"
          value={settings.backendUrl}
          onChange={(e) => onChange({ ...settings, backendUrl: e.target.value })}
        />
      </div>
      <div className="controls-row">
        <span className={`status-chip ${aiStatus}`}>AI: {aiStatus}</span>
        <button className="btn" onClick={onReconnect}>
          Reconnect
        </button>
      </div>
    </div>
  );
}
