import { useCallback, useEffect, useRef, useState } from 'react'

export const PHASE_SECONDS = 120
const TICK_MS = 250

// Backend states: idle | phase1 | phase2 | complete. When phase 1 ends the backend goes back to
// "idle" with e_min set; this hook then moves on to phase 2 by itself.
export function useCalibration({ calibration, connected, send }) {
  const backendState = connected ? (calibration?.state ?? 'idle') : 'idle'
  const phase1Over = connected && backendState === 'idle' && calibration?.e_min != null

  const [session, setSession] = useState(false)
  const [advancing, setAdvancing] = useState(false)
  const [secondsLeft, setSecondsLeft] = useState(PHASE_SECONDS)
  const sawPhase1 = useRef(false)
  const sawPhase2 = useRef(false)
  const advanced = useRef(false)
  const stopped = useRef(false)
  const sendRef = useRef(send)
  useEffect(() => {
    sendRef.current = send
  })

  let state
  if (backendState === 'complete') state = 'complete'
  else if (backendState === 'phase1' || backendState === 'phase2') state = backendState
  else if (session && (advancing || (phase1Over && sawPhase1.current))) state = 'phase2'
  else state = 'idle'

  const resetSession = useCallback(() => {
    setSession(false)
    setAdvancing(false)
    sawPhase1.current = sawPhase2.current = advanced.current = stopped.current = false
  }, [])

  const advance = useCallback(() => {
    if (advanced.current) return
    advanced.current = true
    setAdvancing(true)
    sendRef.current({ cmd: 'start_calibration', phase: 2 })
  }, [])

  const start = useCallback(() => {
    sawPhase1.current = sawPhase2.current = advanced.current = stopped.current = false
    setAdvancing(false)
    setSession(true)
    sendRef.current({ cmd: 'start_calibration', phase: 1 })
  }, [])

  // Track which backend phases this session has actually seen.
  useEffect(() => {
    if (!session) return
    if (backendState === 'phase1') sawPhase1.current = true
    if (backendState === 'phase2') sawPhase2.current = true
  }, [session, backendState])

  // Backend finalized phase 1 (stop or its own timeout): move on to phase 2.
  useEffect(() => {
    if (session && sawPhase1.current && phase1Over && !advanced.current) advance()
  }, [session, phase1Over, advance])

  // Calibration ended, or the connection dropped: leave the session.
  useEffect(() => {
    if (!session) return
    if (backendState === 'complete' || !connected) resetSession()
    else if (sawPhase2.current && backendState === 'idle') resetSession() // phase 2 ended without bounds
  }, [session, backendState, connected, resetSession])

  // Per-phase countdown; at zero, phase 1 advances and phase 2 is stopped (each exactly once).
  useEffect(() => {
    if (state !== 'phase1' && state !== 'phase2') return
    const startedAt = Date.now()
    setSecondsLeft(PHASE_SECONDS)
    const timer = setInterval(() => {
      const left = Math.max(0, Math.ceil(PHASE_SECONDS - (Date.now() - startedAt) / 1000))
      setSecondsLeft(left)
      if (left > 0) return
      if (state === 'phase1') advance()
      else if (!stopped.current) {
        stopped.current = true
        sendRef.current({ cmd: 'stop_calibration' })
      }
    }, TICK_MS)
    return () => clearInterval(timer)
  }, [state, advance])

  return { state, secondsLeft, start }
}
