import { afterEach, describe, expect, it, vi } from 'vitest'
import { streamChat } from '../lib/ollama.js'
import { ndjson, streamResponse } from './helpers.js'

afterEach(() => vi.unstubAllGlobals())

function run(chunks, init) {
  const fetchMock = vi.fn().mockResolvedValue(streamResponse(chunks, init))
  vi.stubGlobal('fetch', fetchMock)
  const received = []
  const onDone = vi.fn()
  const promise = streamChat({ messages: [{ role: 'user', content: 'hi' }], onChunk: (t) => received.push(t), onDone })
  return { fetchMock, received, onDone, promise }
}

describe('streamChat', () => {
  it('calls onChunk for each content delta and onDone at the end', async () => {
    const { received, onDone, promise } = run([ndjson('Hel') + ndjson('lo'), ndjson('', { done: true })])
    await promise
    expect(received).toEqual(['Hel', 'lo'])
    expect(onDone).toHaveBeenCalledTimes(1)
  })

  it('buffers a JSON line split across reads', async () => {
    const line = ndjson('split')
    const { received, promise } = run([line.slice(0, 10), line.slice(10)])
    await promise
    expect(received).toEqual(['split'])
  })

  it('throws on a non-OK response without calling onDone', async () => {
    const { onDone, promise } = run(['nope'], { status: 500, statusText: 'Server Error' })
    await expect(promise).rejects.toThrow(/500/)
    expect(onDone).not.toHaveBeenCalled()
  })

  it('posts the expected request to the native Ollama chat endpoint', async () => {
    const { fetchMock, promise } = run([ndjson('x')])
    await promise
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:11434/api/chat')
    expect(opts.method).toBe('POST')
    expect(JSON.parse(opts.body)).toEqual({
      model: 'qwen3.5:9b',
      stream: true,
      think: false,
      messages: [{ role: 'user', content: 'hi' }],
    })
  })

  it('skips malformed JSON lines', async () => {
    const { received, promise } = run([ndjson('a') + 'not json\n' + ndjson('b')])
    await promise
    expect(received).toEqual(['a', 'b'])
  })

  it('ignores message.thinking', async () => {
    const thinking = JSON.stringify({ message: { role: 'assistant', content: '', thinking: 'hmm' }, done: false }) + '\n'
    const { received, promise } = run([thinking + ndjson('answer')])
    await promise
    expect(received).toEqual(['answer'])
  })

  it('handles a final line with no trailing newline', async () => {
    const { received, promise } = run([ndjson('a') + JSON.stringify({ message: { content: 'b' } })])
    await promise
    expect(received).toEqual(['a', 'b'])
  })
})
