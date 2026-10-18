import { useCallback, useEffect, useRef, useState } from 'react'

const DEFAULT_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8765'
const RECONNECT_MS = 2000

export function useEngagementStream(url = DEFAULT_URL) {
  const [payload, setPayload] = useState(null)
  const [connected, setConnected] = useState(false)
  const [frozen, setFrozen] = useState(null) // { value } while frozen, else null
  const wsRef = useRef(null)
  const liveFused = useRef(null)

  useEffect(() => {
    let closedByUs = false
    let retryTimer

    function connect() {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (wsRef.current === ws) setConnected(true)
      }
      ws.onmessage = (event) => {
        if (wsRef.current !== ws) return
        let data
        try {
          data = JSON.parse(event.data)
        } catch {
          return
        }
        if (!data || typeof data !== 'object') return
        liveFused.current = data.fused ?? null
        setPayload(data)
      }
      ws.onclose = () => {
        if (wsRef.current !== ws) return
        wsRef.current = null
        liveFused.current = null
        setConnected(false)
        setPayload(null)
        if (!closedByUs) retryTimer = setTimeout(connect, RECONNECT_MS)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      closedByUs = true
      clearTimeout(retryTimer)
      const ws = wsRef.current
      wsRef.current = null
      if (ws) {
        ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null
        ws.close()
      }
    }
  }, [url])

  const freeze = useCallback(() => setFrozen({ value: liveFused.current }), [])
  const unfreeze = useCallback(() => setFrozen(null), [])

  const send = useCallback((message) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message))
      return true
    }
    return false
  }, [])

  const fused = connected ? (frozen ? frozen.value : (payload?.fused ?? null)) : null

  return {
    fused,
    confidence: payload?.confidence ?? null,
    sources: payload?.sources ?? {},
    calibration: payload?.calibration ?? null,
    connected,
    freeze,
    unfreeze,
    send,
  }
}
