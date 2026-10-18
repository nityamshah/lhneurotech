import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useEngagementStream } from '../hooks/useEngagementStream.js'
import { MockWebSocket, makePayload } from './helpers.js'

beforeEach(() => {
  MockWebSocket.reset()
  vi.stubGlobal('WebSocket', MockWebSocket)
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

const URL = 'ws://test:1234'

function setup() {
  const hook = renderHook(() => useEngagementStream(URL))
  const ws = () => MockWebSocket.last
  return { ...hook, ws }
}

describe('useEngagementStream', () => {
  it('connects to the given URL and is connected only after open', () => {
    const { result, ws } = setup()
    expect(ws().url).toBe(URL)
    expect(result.current.connected).toBe(false)
    act(() => ws().open())
    expect(result.current.connected).toBe(true)
  })

  it('exposes the payload fields', () => {
    const { result, ws } = setup()
    act(() => ws().open())
    act(() => ws().message(makePayload({ fused: 0.42, confidence: 0.8, state: 'complete', e_min: 1, e_max: 2 })))
    expect(result.current.fused).toBe(0.42)
    expect(result.current.confidence).toBe(0.8)
    expect(result.current.sources.eeg.confidence).toBe(0.8)
    expect(result.current.calibration).toEqual({ state: 'complete', e_min: 1, e_max: 2 })
  })

  it('reports null values before any payload', () => {
    const { result } = setup()
    expect(result.current.fused).toBeNull()
    expect(result.current.confidence).toBeNull()
    expect(result.current.calibration).toBeNull()
  })

  it('reconnects after the connection drops and clears the score meanwhile', () => {
    const { result, ws } = setup()
    act(() => ws().open())
    act(() => ws().message(makePayload({ fused: 0.5, state: 'complete', e_min: 1, e_max: 2 })))
    act(() => ws().drop())
    expect(result.current.connected).toBe(false)
    expect(result.current.fused).toBeNull()
    expect(MockWebSocket.instances).toHaveLength(1)

    act(() => vi.advanceTimersByTime(2000))
    expect(MockWebSocket.instances).toHaveLength(2)
    act(() => ws().open())
    expect(result.current.connected).toBe(true)
  })

  it('freeze() holds the captured score while new payloads arrive; unfreeze() releases it', () => {
    const { result, ws } = setup()
    act(() => ws().open())
    act(() => ws().message(makePayload({ fused: 0.4, state: 'complete', e_min: 1, e_max: 2 })))
    act(() => result.current.freeze())
    act(() => ws().message(makePayload({ fused: 0.9, state: 'complete', e_min: 1, e_max: 2 })))
    expect(result.current.fused).toBe(0.4)
    act(() => result.current.unfreeze())
    expect(result.current.fused).toBe(0.9)
  })

  it('freeze() again re-captures the live score', () => {
    const { result, ws } = setup()
    act(() => ws().open())
    act(() => ws().message(makePayload({ fused: 0.4, state: 'complete', e_min: 1, e_max: 2 })))
    act(() => result.current.freeze())
    act(() => ws().message(makePayload({ fused: 0.9, state: 'complete', e_min: 1, e_max: 2 })))
    act(() => result.current.freeze())
    expect(result.current.fused).toBe(0.9)
  })

  it('freezing while the score is null keeps it null', () => {
    const { result, ws } = setup()
    act(() => ws().open())
    act(() => ws().message(makePayload({ fused: null })))
    act(() => result.current.freeze())
    act(() => ws().message(makePayload({ fused: 0.7, state: 'complete', e_min: 1, e_max: 2 })))
    expect(result.current.fused).toBeNull()
  })

  it('returns a null score while disconnected, even if frozen', () => {
    const { result, ws } = setup()
    act(() => ws().open())
    act(() => ws().message(makePayload({ fused: 0.4, state: 'complete', e_min: 1, e_max: 2 })))
    act(() => result.current.freeze())
    act(() => ws().drop())
    expect(result.current.fused).toBeNull()
  })

  it('ignores malformed messages', () => {
    const { result, ws } = setup()
    act(() => ws().open())
    act(() => ws().message(makePayload({ fused: 0.3, state: 'complete', e_min: 1, e_max: 2 })))
    act(() => ws().message('not json'))
    act(() => ws().message('42'))
    expect(result.current.fused).toBe(0.3)
  })

  it('send() writes JSON only while the socket is open', () => {
    const { result, ws } = setup()
    expect(result.current.send({ cmd: 'x' })).toBe(false)
    act(() => ws().open())
    expect(result.current.send({ cmd: 'stop_calibration' })).toBe(true)
    expect(ws().sent).toEqual([{ cmd: 'stop_calibration' }])
  })

  it('closes the socket on unmount and does not reconnect', () => {
    const { unmount, ws } = setup()
    act(() => ws().open())
    const socket = ws()
    unmount()
    expect(socket.readyState).toBe(MockWebSocket.CLOSED)
    act(() => vi.advanceTimersByTime(10000))
    expect(MockWebSocket.instances).toHaveLength(1)
  })

  it('survives a StrictMode remount: the first socket is closed and the second one wins', () => {
    const { result } = renderHook(() => useEngagementStream(URL), { wrapper: StrictMode })
    expect(MockWebSocket.instances).toHaveLength(2)
    expect(MockWebSocket.instances[0].readyState).toBe(MockWebSocket.CLOSED)
    act(() => MockWebSocket.last.open())
    expect(result.current.connected).toBe(true)
    act(() => vi.advanceTimersByTime(10000))
    expect(MockWebSocket.instances).toHaveLength(2)
  })
})
