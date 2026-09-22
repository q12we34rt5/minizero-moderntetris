import type { GameMode, PveSettings, ModelInfo } from '../game/useGame.ts';
import type { AiConnectionStatus } from '../ai/client.ts';

interface Props {
  mode: GameMode; // 'pve' or 'eve'
  settings: PveSettings;
  onChange: (s: PveSettings) => void;
  models: ModelInfo[];
  aiStatusA: AiConnectionStatus;
  aiStatusB: AiConnectionStatus;
  onReconnect: () => void; // also re-fetches the model list
}

function ModelSelect({
  id,
  label,
  value,
  models,
  onPick,
}: {
  id: string;
  label: string;
  value: string;
  models: ModelInfo[];
  onPick: (modelId: string) => void;
}) {
  // Empty selection resolves to the first advertised model; reflect that in the
  // <select> so it never shows a blank while still storing '' as "auto". An id
  // the router no longer advertises resolves the same way -- matching what the
  // browser renders anyway (no matching <option> => it selects the first one),
  // and what useGame connects to, so the label can't lie about the backend.
  const effective = models.some((m) => m.id === value) ? value : models[0]?.id || '';
  return (
    <div className="setting-item" style={{ marginTop: 8 }}>
      <label htmlFor={id}>{label}</label>
      <select id={id} value={effective} onChange={(e) => onPick(e.target.value)}>
        {models.length === 0 && <option value="">(no models — check router)</option>}
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.displayName}
          </option>
        ))}
      </select>
    </div>
  );
}

export function PveSettingsPanel({ mode, settings, onChange, models, aiStatusA, aiStatusB, onReconnect }: Props) {
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

      <div className="setting-item" style={{ marginTop: 8 }}>
        <label htmlFor="set-router-url">Router URL</label>
        <input
          id="set-router-url"
          type="text"
          value={settings.routerUrl}
          placeholder="ws://localhost:8000"
          onChange={(e) => onChange({ ...settings, routerUrl: e.target.value })}
        />
      </div>

      {isEve && (
        <ModelSelect
          id="set-model-a"
          label="Board A Model"
          value={settings.modelA}
          models={models}
          onPick={(modelId) => onChange({ ...settings, modelA: modelId })}
        />
      )}
      <ModelSelect
        id="set-model-b"
        label={isEve ? 'Board B Model' : 'AI Model'}
        value={settings.modelB}
        models={models}
        onPick={(modelId) => onChange({ ...settings, modelB: modelId })}
      />

      <div className="controls-row">
        {isEve && <span className={`status-chip ${aiStatusA}`}>A: {aiStatusA}</span>}
        <span className={`status-chip ${aiStatusB}`}>{isEve ? 'B' : 'AI'}: {aiStatusB}</span>
        <button className="btn" onClick={onReconnect}>
          Reconnect
        </button>
      </div>
      <p className="hint">Model changes reconnect immediately; Reconnect also re-reads the model list.</p>
    </div>
  );
}
