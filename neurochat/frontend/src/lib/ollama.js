const OLLAMA_URL = 'http://localhost:11434/api/chat'
const MODEL = 'qwen3.5:9b'

function handleLine(line, onChunk) {
  if (!line.trim()) return
  let parsed
  try {
    parsed = JSON.parse(line)
  } catch {
    return
  }
  const text = parsed?.message?.content
  if (text) onChunk(text)
}

export async function streamChat({ messages, onChunk, onDone }) {
  const res = await fetch(OLLAMA_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL, stream: true, think: false, messages }),
  })
  if (!res.ok) {
    throw new Error(`Ollama request failed: ${res.status} ${res.statusText}`.trim())
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()
    for (const line of lines) handleLine(line, onChunk)
  }
  buffer += decoder.decode()
  handleLine(buffer, onChunk)

  onDone?.()
}
