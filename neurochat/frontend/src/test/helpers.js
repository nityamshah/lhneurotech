// Build a fetch Response whose body streams the given string chunks as separate reads.
export function streamResponse(chunks, init = {}) {
  const encoder = new TextEncoder()
  const body = new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c))
      controller.close()
    },
  })
  return new Response(body, { status: 200, ...init })
}

// Minimal stand-in for the browser WebSocket. Tests drive it with open() / message() / drop().
export class MockWebSocket {
  static OPEN = 1
  static CLOSED = 3
  static instances = []

  static reset() {
    MockWebSocket.instances = []
  }

  static get last() {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1]
  }

  constructor(url) {
    this.url = url
    this.readyState = 0
    this.sent = []
    MockWebSocket.instances.push(this)
  }

  send(data) {
    this.sent.push(JSON.parse(data))
  }

  close() {
    if (this.readyState === MockWebSocket.CLOSED) return
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.({})
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.({})
  }

  message(data) {
    this.onmessage?.({ data: typeof data === 'string' ? data : JSON.stringify(data) })
  }

  drop() {
    this.close()
  }
}

// A backend payload with sensible defaults.
export function makePayload({ fused = null, confidence = 0.9, state = 'idle', e_min = null, e_max = null } = {}) {
  return {
    fused,
    confidence,
    sources: { eeg: { score: fused, confidence } },
    calibration: { state, e_min, e_max },
    timestamp: 1,
  }
}

// One NDJSON line in Ollama's /api/chat format.
export function ndjson(content, extra = {}) {
  return JSON.stringify({ message: { role: 'assistant', content }, done: false, ...extra }) + '\n'
}
