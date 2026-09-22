/**
 * Generic AI-info box: renders whatever key->value pairs the backend attached
 * to the last AI move (e.g. value, winloss). Purely presentational — new keys
 * from the backend show up automatically, no frontend change needed.
 */
interface Props {
  info: Record<string, string>;
}

// Pretty labels for known keys; unknown keys fall back to the raw key.
const LABELS: Record<string, string> = {
  value: 'Value',
  winloss: 'Win/Loss',
};

function formatValue(raw: string): string {
  const n = Number(raw);
  if (!Number.isNaN(n) && raw.trim() !== '') {
    // Compact fixed-point for numbers; keeps small win/loss deltas readable.
    return Math.abs(n) >= 100 ? n.toFixed(1) : n.toFixed(3);
  }
  return raw;
}

export function AiInfoPanel({ info }: Props) {
  const keys = Object.keys(info);
  if (keys.length === 0) return null;
  return (
    <div className="panel ai-info-box">
      <div className="panel-title">AI</div>
      <div className="stat-grid" style={{ gridTemplateColumns: '1fr' }}>
        {keys.map((key) => (
          <div className="stat-item" key={key}>
            <span className="stat-label">{LABELS[key] ?? key}</span>
            <span className="stat-value accent">{formatValue(info[key])}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
