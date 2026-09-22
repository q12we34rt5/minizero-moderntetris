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

type NumKey =
  | 'pveAiIntervalMs'
  | 'pveGarbageDelay'
  | 'eveAiIntervalMsA'
  | 'eveGarbageDelayA'
  | 'eveAiIntervalMsB'
  | 'eveGarbageDelayB';

/** One AI's timing pair (interval + garbage delay). */
function TimingPair({
  idBase,
  heading,
  interval,
  garbage,
  onInterval,
  onGarbage,
}: {
  idBase: string;
  heading?: string;
  interval: number;
  garbage: number;
  onInterval: (v: string) => void;
  onGarbage: (v: string) => void;
}) {
  return (
    <>
      {heading && (
        <div className="setting-item" style={{ gridColumn: '1 / -1', marginTop: 4 }}>
          <label>{heading}</label>
        </div>
      )}
      <div className="setting-item">
        <label htmlFor={`${idBase}-interval`}>AI Interval (ms)</label>
        <input
          id={`${idBase}-interval`}
          type="number"
          min={0}
          max={5000}
          value={interval}
          onChange={(e) => onInterval(e.target.value)}
        />
      </div>
      <div className="setting-item">
        <label htmlFor={`${idBase}-garbage`}>Garbage Delay</label>
        <input
          id={`${idBase}-garbage`}
          type="number"
          min={0}
          max={20}
          value={garbage}
          onChange={(e) => onGarbage(e.target.value)}
        />
      </div>
    </>
  );
}

export function PveSettingsPanel({ mode, settings, onChange, models, aiStatusA, aiStatusB, onReconnect }: Props) {
  const updateNum = (key: NumKey, value: string) => {
    const n = parseInt(value, 10);
    onChange({ ...settings, [key]: Number.isFinite(n) ? n : 0 });
  };
  const isEve = mode === 'eve';

  return (
    <div className="panel" style={{ flex: 1, minWidth: 300 }}>
      <div className="panel-title">{isEve ? 'EvE / AI Backends' : 'PvE / AI Backend'}</div>
      <div className="settings-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        {isEve ? (
          <>
            <TimingPair
              idBase="set-eve-a"
              heading="AI A (Board A)"
              interval={settings.eveAiIntervalMsA}
              garbage={settings.eveGarbageDelayA}
              onInterval={(v) => updateNum('eveAiIntervalMsA', v)}
              onGarbage={(v) => updateNum('eveGarbageDelayA', v)}
            />
            <TimingPair
              idBase="set-eve-b"
              heading="AI B (Board B)"
              interval={settings.eveAiIntervalMsB}
              garbage={settings.eveGarbageDelayB}
              onInterval={(v) => updateNum('eveAiIntervalMsB', v)}
              onGarbage={(v) => updateNum('eveGarbageDelayB', v)}
            />
          </>
        ) : (
          <TimingPair
            idBase="set-pve"
            interval={settings.pveAiIntervalMs}
            garbage={settings.pveGarbageDelay}
            onInterval={(v) => updateNum('pveAiIntervalMs', v)}
            onGarbage={(v) => updateNum('pveGarbageDelay', v)}
          />
        )}
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

      {isEve ? (
        <>
          <ModelSelect
            id="set-model-a"
            label="Board A Model"
            value={settings.eveModelA}
            models={models}
            onPick={(modelId) => onChange({ ...settings, eveModelA: modelId })}
          />
          <ModelSelect
            id="set-model-b"
            label="Board B Model"
            value={settings.eveModelB}
            models={models}
            onPick={(modelId) => onChange({ ...settings, eveModelB: modelId })}
          />
        </>
      ) : (
        <ModelSelect
          id="set-model-b"
          label="AI Model"
          value={settings.pveModel}
          models={models}
          onPick={(modelId) => onChange({ ...settings, pveModel: modelId })}
        />
      )}

      {isEve && (
        <div className="controls-row" style={{ marginTop: 8 }}>
          <label htmlFor="set-eve-seed-sync">Sync seed (same start)</label>
          <input
            id="set-eve-seed-sync"
            type="checkbox"
            checked={settings.eveSeedSync}
            onChange={(e) => onChange({ ...settings, eveSeedSync: e.target.checked })}
          />
        </div>
      )}

      <div className="controls-row">
        {isEve && <span className={`status-chip ${aiStatusA}`}>A: {aiStatusA}</span>}
        <span className={`status-chip ${aiStatusB}`}>{isEve ? 'B' : 'AI'}: {aiStatusB}</span>
        <button className="btn" onClick={onReconnect}>
          Reconnect
        </button>
      </div>
      <p className="hint">
        Switching model reconnects that board immediately. Reconnect also re-fetches
        the model list (use it after editing the router registry).
        {isEve && ' Seed sync takes effect on Reset: on = both boards share the piece sequence, off = independent bags.'}
      </p>
    </div>
  );
}
