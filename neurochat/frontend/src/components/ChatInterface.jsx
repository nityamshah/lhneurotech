import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import { streamChat } from '../lib/ollama.js'
import { SYSTEM_PROMPT } from '../lib/systemPrompt.js'

function formatTag(score) {
  return typeof score === 'number' && Number.isFinite(score)
    ? `[Engagement: ${score.toFixed(2)}]`
    : '[Engagement: unavailable]'
}

const styles = {
  root: { display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: 800, margin: '0 auto', fontFamily: 'system-ui, sans-serif' },
  log: { flex: 1, overflowY: 'auto', padding: '96px 16px 16px' }, // top room for the fixed BrainWidget
  user: { background: '#e8f0fe', borderRadius: 8, padding: '8px 12px', margin: '8px 0 8px auto', maxWidth: '80%', width: 'fit-content', whiteSpace: 'pre-wrap' },
  assistant: { background: '#f4f4f5', borderRadius: 8, padding: '8px 12px', margin: '8px 0', maxWidth: '80%', width: 'fit-content' },
  error: { background: '#fdecea', color: '#b3261e', padding: '8px 16px' },
  form: { display: 'flex', gap: 8, padding: 16, borderTop: '1px solid #ddd' },
  input: { flex: 1, padding: 8, fontSize: 16 },
}

export default function ChatInterface({
  score,
  connected = false,
  calibrated = true,
  onInputFocus,
  onMessageSent,
}) {
  const locked = connected && !calibrated
  // What the user sees (untagged) vs. what the API sees (tagged).
  const [messages, setMessages] = useState([])
  const apiMessages = useRef([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState(null)
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [messages])

  async function handleSubmit(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || streaming || locked) return

    const taggedUser = { role: 'user', content: `${formatTag(score)}\n${text}` }
    const history = apiMessages.current
    let reply = ''

    setError(null)
    setInput('')
    setStreaming(true)
    setMessages((m) => [...m, { role: 'user', content: text }, { role: 'assistant', content: '' }])

    try {
      await streamChat({
        messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...history, taggedUser],
        onChunk: (chunk) => {
          reply += chunk
          setMessages((m) => [...m.slice(0, -1), { role: 'assistant', content: reply }])
        },
        onDone: () => {},
      })
      apiMessages.current = [...history, taggedUser, { role: 'assistant', content: reply }]
    } catch (err) {
      setError(`Could not get a response from Ollama: ${err.message}`)
      if (!reply) setMessages((m) => m.slice(0, -1))
    } finally {
      setStreaming(false)
      onMessageSent?.()
    }
  }

  return (
    <div style={styles.root}>
      <div style={styles.log}>
        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} style={styles.user}>{m.content}</div>
          ) : (
            <div key={i} style={styles.assistant}><Markdown>{m.content}</Markdown></div>
          ),
        )}
        <div ref={endRef} />
      </div>
      {error && <div role="alert" style={styles.error}>{error}</div>}
      <form style={styles.form} onSubmit={handleSubmit}>
        <input
          style={styles.input}
          aria-label="Message"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onFocus={() => onInputFocus?.()}
          disabled={streaming || locked}
          placeholder={locked ? 'Complete calibration to begin.' : 'Ask anything...'}
        />
        <button type="submit" disabled={streaming || locked || !input.trim()}>Send</button>
      </form>
    </div>
  )
}
