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

// One NDJSON line in Ollama's /api/chat format.
export function ndjson(content, extra = {}) {
  return JSON.stringify({ message: { role: 'assistant', content }, done: false, ...extra }) + '\n'
}
