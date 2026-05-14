import type { InputSettings } from '../input/keyboard.ts';

interface Props {
  settings: InputSettings;
  onChange: (s: InputSettings) => void;
}

const FIELDS: { key: keyof InputSettings; label: string; min: number; max: number }[] = [
  { key: 'das', label: 'DAS', min: 0, max: 500 },
  { key: 'arr', label: 'ARR', min: 0, max: 200 },
  { key: 'dropDas', label: 'Drop DAS', min: 0, max: 500 },
  { key: 'dropArr', label: 'Drop ARR', min: 0, max: 200 },
  { key: 'dropRate', label: 'Drop Rate', min: 0, max: 5000 },
];

export function SettingsPanel({ settings, onChange }: Props) {
  const update = (key: keyof InputSettings, value: string) => {
    const n = parseInt(value, 10);
    onChange({ ...settings, [key]: Number.isFinite(n) ? n : 0 });
  };

  return (
    <div className="panel" style={{ flex: 1, minWidth: 280 }}>
      <div className="panel-title">Timing Settings (ms)</div>
      <div className="settings-grid">
        {FIELDS.map((f) => (
          <div className="setting-item" key={f.key}>
            <label htmlFor={`set-${f.key}`}>{f.label}</label>
            <input
              id={`set-${f.key}`}
              type="number"
              min={f.min}
              max={f.max}
              value={settings[f.key]}
              onChange={(e) => update(f.key, e.target.value)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
