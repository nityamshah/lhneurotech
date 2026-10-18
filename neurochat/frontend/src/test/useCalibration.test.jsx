import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useCalibration } from '../hooks/useCalibration.js'

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

const cal = (state, e_min = null, e_max = null) => ({ state, e_min, e_max })

function setup(initial = cal('idle')) {
  const send = vi.fn()
  const hook = renderHook((props) => useCalibration(props), {
    initialProps: { calibration: initial, connected: true, send },
  })
  const backend = (calibration, connected = true) => hook.rerender({ calibration, connected, send })
  const commands = () => send.mock.calls.map(([c]) => c)
  const count = (cmd, phase) =>
    commands().filter((c) => c.cmd === cmd && (phase === undefined || c.phase === phase)).length
  return { ...hook, backend, send, commands, count }
}

describe('useCalibration', () => {
  it.each(['idle', 'phase1', 'phase2', 'complete'])('maps backend state %s', (state) => {
    const { result } = setup(cal(state))
    expect(result.current.state).toBe(state)
  })

  it('treats a missing payload or a disconnected backend as idle', () => {
    const { result, backend } = setup(null)
    expect(result.current.state).toBe('idle')
    backend(cal('phase1'), false)
    expect(result.current.state).toBe('idle')
  })

  it('start() sends phase 1', () => {
    const { result, commands } = setup()
    act(() => result.current.start())
    expect(commands()).toEqual([{ cmd: 'start_calibration', phase: 1 }])
  })

  it('counts down from 120 seconds while a phase runs', () => {
    const { result, backend } = setup()
    act(() => result.current.start())
    backend(cal('phase1'))
    expect(result.current.secondsLeft).toBe(120)
    act(() => vi.advanceTimersByTime(30_000))
    expect(result.current.secondsLeft).toBe(90)
  })

  it('moves to phase 2 when the backend reports phase 1 finished', () => {
    const { result, backend, count } = setup()
    act(() => result.current.start())
    backend(cal('phase1', 0.5))
    backend(cal('idle', 0.5)) // backend finalized phase 1
    expect(count('start_calibration', 2)).toBe(1)
    expect(result.current.state).toBe('phase2')
    backend(cal('phase2', 0.5))
    expect(result.current.state).toBe('phase2')
  })

  it('does not skip phase 1 because of a stale idle payload with an old e_min', () => {
    const { result, backend, count } = setup(cal('idle', 0.5))
    act(() => result.current.start())
    backend(cal('idle', 0.5)) // payload sent before the backend saw start_calibration
    expect(count('start_calibration', 2)).toBe(0)
    expect(result.current.state).toBe('idle')
  })

  it('advances on its own countdown, exactly once, even if the backend never reports phase 1 over', () => {
    const { result, backend, count } = setup()
    act(() => result.current.start())
    backend(cal('phase1', 0.5))
    act(() => vi.advanceTimersByTime(119_000))
    expect(count('start_calibration', 2)).toBe(0)
    act(() => vi.advanceTimersByTime(2_000))
    expect(count('start_calibration', 2)).toBe(1)
    act(() => vi.advanceTimersByTime(5_000))
    expect(count('start_calibration', 2)).toBe(1)
  })

  it('does not send phase 2 twice when the backend finalizes before the countdown ends', () => {
    const { result, backend, count } = setup()
    act(() => result.current.start())
    backend(cal('phase1', 0.5))
    backend(cal('idle', 0.5))
    backend(cal('phase2', 0.5))
    act(() => vi.advanceTimersByTime(130_000))
    expect(count('start_calibration', 2)).toBe(1)
  })

  it('sends stop_calibration exactly once at the end of phase 2', () => {
    const { result, backend, count } = setup()
    act(() => result.current.start())
    backend(cal('phase1', 0.5))
    backend(cal('idle', 0.5))
    backend(cal('phase2', 0.5))
    act(() => vi.advanceTimersByTime(119_000))
    expect(count('stop_calibration')).toBe(0)
    act(() => vi.advanceTimersByTime(2_000))
    expect(count('stop_calibration')).toBe(1)
    act(() => vi.advanceTimersByTime(10_000))
    expect(count('stop_calibration')).toBe(1)
  })

  it('reports complete and stops sending once the backend is complete', () => {
    const { result, backend, commands } = setup()
    act(() => result.current.start())
    backend(cal('phase1', 0.5))
    backend(cal('idle', 0.5))
    backend(cal('phase2', 0.5))
    backend(cal('complete', 0.5, 0.9))
    const before = commands().length
    act(() => vi.advanceTimersByTime(300_000))
    expect(result.current.state).toBe('complete')
    expect(commands()).toHaveLength(before)
  })

  it('returns to idle (Start available again) if phase 2 ends without completing', () => {
    const { result, backend, count } = setup()
    act(() => result.current.start())
    backend(cal('phase1', 0.5))
    backend(cal('idle', 0.5))
    backend(cal('phase2', 0.5))
    backend(cal('idle', 0.5)) // backend finalized phase 2 without an e_max
    expect(result.current.state).toBe('idle')
    expect(count('start_calibration', 2)).toBe(1) // no restart loop
    act(() => result.current.start())
    expect(count('start_calibration', 1)).toBe(2)
  })

  it('abandons the session when the connection drops', () => {
    const { result, backend, count } = setup()
    act(() => result.current.start())
    backend(cal('phase1', 0.5))
    backend(cal('phase1', 0.5), false)
    expect(result.current.state).toBe('idle')
    backend(cal('idle', 0.5), true) // reconnected; backend looks like "phase 1 over"
    expect(count('start_calibration', 2)).toBe(0)
    expect(result.current.state).toBe('idle')
  })
})
