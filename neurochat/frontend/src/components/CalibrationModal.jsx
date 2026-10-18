export function formatCountdown(totalSeconds) {
  const s = Math.max(0, Math.ceil(totalSeconds))
  const minutes = Math.floor(s / 60)
  const seconds = String(s % 60).padStart(2, '0')
  return `${minutes}:${seconds}`
}

const styles = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 100, background: 'rgba(255,255,255,0.97)', display: 'flex',
    alignItems: 'center', justifyContent: 'center', fontFamily: 'system-ui, sans-serif', textAlign: 'center',
  },
  card: { maxWidth: 520, padding: 24 },
  timer: { fontSize: 64, fontVariantNumeric: 'tabular-nums', margin: '16px 0' },
  button: { fontSize: 18, padding: '10px 28px', cursor: 'pointer' },
}

export default function CalibrationModal({ state, secondsLeft, onStart }) {
  if (state !== 'idle' && state !== 'phase1' && state !== 'phase2') return null

  return (
    <div style={styles.overlay} role="dialog" aria-modal="true" aria-label="Calibration">
      <div style={styles.card}>
        {state === 'idle' && (
          <>
            <h1>Calibration</h1>
            <p>Before you start, two short tasks (2 minutes each) set your personal baseline:</p>
            <ol style={{ textAlign: 'left' }}>
              <li><strong>Relaxation.</strong> Sit still and clear your mind.</li>
              <li><strong>Word association.</strong> Chain words together, each starting with the last letter of the previous one.</li>
            </ol>
            <button style={styles.button} onClick={onStart}>Start</button>
          </>
        )}
        {state === 'phase1' && (
          <>
            <h1>Phase 1 of 2</h1>
            <p>Relax. Sit still and clear your mind.</p>
            <div style={styles.timer} aria-label="Time remaining">{formatCountdown(secondsLeft)}</div>
          </>
        )}
        {state === 'phase2' && (
          <>
            <h1>Phase 2 of 2</h1>
            <p>Word association. Think of a word, then a new word starting with its last letter.</p>
            <p>For example: apple → elephant → table</p>
            <div style={styles.timer} aria-label="Time remaining">{formatCountdown(secondsLeft)}</div>
          </>
        )}
      </div>
    </div>
  )
}
