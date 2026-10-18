const COLORS = { red: '#d93025', yellow: '#f9ab00', green: '#188038', none: '#9aa0a6' }

export function signalQuality(confidence) {
  if (typeof confidence !== 'number' || !Number.isFinite(confidence)) return 'none'
  if (confidence < 0.4) return 'red'
  if (confidence <= 0.7) return 'yellow'
  return 'green'
}

const styles = {
  root: {
    position: 'fixed', top: 12, right: 12, zIndex: 10, background: '#fff', border: '1px solid #ddd',
    borderRadius: 8, padding: '8px 12px', fontFamily: 'system-ui, sans-serif', fontSize: 13,
    boxShadow: '0 1px 4px rgba(0,0,0,0.12)', minWidth: 150,
  },
  row: { display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 },
  dot: { width: 10, height: 10, borderRadius: '50%', display: 'inline-block' },
}

export default function BrainWidget({ connected, confidence, fused }) {
  if (!connected) {
    return (
      <div style={styles.root} data-testid="brain-widget">
        <div style={styles.row}>
          <span style={{ ...styles.dot, background: COLORS.none }} />
          <span>Backend not connected</span>
        </div>
      </div>
    )
  }

  const quality = signalQuality(confidence)
  const percent = quality === 'none' ? '—' : `${Math.round(confidence * 100)}%`
  const score = typeof fused === 'number' ? fused.toFixed(2) : '—'

  return (
    <div style={styles.root} data-testid="brain-widget">
      <div style={styles.row}>
        <span style={{ ...styles.dot, background: '#188038' }} />
        <span>Connected</span>
      </div>
      <div style={styles.row}>
        <span
          style={{ ...styles.dot, background: COLORS[quality] }}
          data-testid="signal-quality"
          data-quality={quality}
        />
        <span>Signal: {percent}</span>
      </div>
      <div style={styles.row}>
        <span>Engagement: {score}</span>
      </div>
    </div>
  )
}
