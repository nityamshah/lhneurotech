import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatInterface from '../components/ChatInterface.jsx'
import { SYSTEM_PROMPT } from '../lib/systemPrompt.js'
import { ndjson, streamResponse } from './helpers.js'

let fetchMock

beforeEach(() => {
  fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const reply = (...parts) => () => Promise.resolve(streamResponse([parts.map((p) => ndjson(p)).join('')]))
const sentMessages = (call = 0) => JSON.parse(fetchMock.mock.calls[call][1].body).messages

async function send(user, text) {
  await user.type(screen.getByLabelText('Message'), text)
  await user.click(screen.getByRole('button', { name: 'Send' }))
}

describe('ChatInterface', () => {
  it('sends the system prompt first and tags the user message with 2 decimals', async () => {
    fetchMock.mockImplementation(reply('ok'))
    const user = userEvent.setup()
    render(<ChatInterface score={0.72} />)
    await send(user, 'What is a neuron?')
    await screen.findByText('ok')

    const msgs = sentMessages()
    expect(msgs[0]).toEqual({ role: 'system', content: SYSTEM_PROMPT })
    expect(msgs[1]).toEqual({ role: 'user', content: '[Engagement: 0.72]\nWhat is a neuron?' })
  })

  it('rounds the score to 2 decimals', async () => {
    fetchMock.mockImplementation(reply('ok'))
    const user = userEvent.setup()
    render(<ChatInterface score={0.5} />)
    await send(user, 'hi')
    await screen.findByText('ok')
    expect(sentMessages()[1].content).toBe('[Engagement: 0.50]\nhi')
  })

  it('never shows the engagement tag in the UI', async () => {
    fetchMock.mockImplementation(reply('ok'))
    const user = userEvent.setup()
    render(<ChatInterface score={0.72} />)
    await send(user, 'What is a neuron?')
    await screen.findByText('ok')

    expect(screen.getByText('What is a neuron?')).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(/Engagement/)
    expect(document.body.textContent).not.toContain('0.72')
  })

  it('uses [Engagement: unavailable] when the score is null', async () => {
    fetchMock.mockImplementation(reply('ok'))
    const user = userEvent.setup()
    render(<ChatInterface score={null} />)
    await send(user, 'hi')
    await screen.findByText('ok')
    expect(sentMessages()[1].content).toBe('[Engagement: unavailable]\nhi')
  })

  it('accumulates streamed chunks into one reply', async () => {
    fetchMock.mockImplementation(reply('Neurons ', 'fire ', 'signals.'))
    const user = userEvent.setup()
    render(<ChatInterface score={0.4} />)
    await send(user, 'hi')
    expect(await screen.findByText('Neurons fire signals.')).toBeInTheDocument()
  })

  it('keeps earlier tags in the API history but not in the UI across turns', async () => {
    fetchMock.mockImplementation(reply('answer'))
    const user = userEvent.setup()
    const { rerender } = render(<ChatInterface score={0.2} />)
    await send(user, 'first')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled())
    await waitFor(() => expect(screen.getByLabelText('Message')).not.toBeDisabled())

    rerender(<ChatInterface score={0.9} />)
    await send(user, 'second')
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

    expect(sentMessages(1).map((m) => m.content)).toEqual([
      SYSTEM_PROMPT,
      '[Engagement: 0.20]\nfirst',
      'answer',
      '[Engagement: 0.90]\nsecond',
    ])
    expect(document.body.textContent).not.toMatch(/Engagement/)
  })

  it('shows an error for a non-OK response and keeps working', async () => {
    fetchMock.mockResolvedValueOnce(streamResponse(['boom'], { status: 500, statusText: 'Server Error' }))
    const user = userEvent.setup()
    render(<ChatInterface score={0.5} />)
    await send(user, 'hi')
    expect(await screen.findByRole('alert')).toHaveTextContent(/500/)
    expect(screen.getByLabelText('Message')).not.toBeDisabled()
  })

  it('shows an error when Ollama is unreachable', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    const user = userEvent.setup()
    render(<ChatInterface score={0.5} />)
    await send(user, 'hi')
    expect(await screen.findByRole('alert')).toHaveTextContent(/Failed to fetch/)
  })

  it('does not commit a failed turn to the API history', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    fetchMock.mockImplementationOnce(reply('ok'))
    const user = userEvent.setup()
    render(<ChatInterface score={0.5} />)
    await send(user, 'one')
    await screen.findByRole('alert')
    await send(user, 'two')
    await screen.findByText('ok')
    expect(sentMessages(1).map((m) => m.content)).toEqual([SYSTEM_PROMPT, '[Engagement: 0.50]\ntwo'])
  })
})
