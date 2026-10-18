import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App.jsx'
import { MockWebSocket, makePayload, ndjson, streamResponse } from './helpers.js'

let fetchMock

beforeEach(() => {
  MockWebSocket.reset()
  vi.stubGlobal('WebSocket', MockWebSocket)
  fetchMock = vi.fn(() => Promise.resolve(streamResponse([ndjson('ok')])))
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const ws = () => MockWebSocket.last
const calibratedPayload = (fused) => makePayload({ fused, state: 'complete', e_min: 1, e_max: 3 })
const sentMessages = () => JSON.parse(fetchMock.mock.calls[0][1].body).messages

describe('App', () => {
  it('shows the calibration modal and locks the chat until calibration completes', async () => {
    const user = userEvent.setup()
    render(<App />)
    act(() => ws().open())
    act(() => ws().message(makePayload({ state: 'idle' })))

    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Complete calibration to begin.')).toBeDisabled()

    await user.click(screen.getByRole('button', { name: 'Start' }))
    expect(ws().sent).toEqual([{ cmd: 'start_calibration', phase: 1 }])

    act(() => ws().message(makePayload({ state: 'phase1', e_min: 1 })))
    expect(screen.getByText('Relax. Sit still and clear your mind.')).toBeInTheDocument()

    act(() => ws().message(makePayload({ fused: 0.5, state: 'complete', e_min: 1, e_max: 3 })))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByPlaceholderText('Ask anything...')).not.toBeDisabled()
  })

  it('freezes the score on focus, sends the frozen value, and unfreezes after the reply', async () => {
    const user = userEvent.setup()
    render(<App />)
    act(() => ws().open())
    act(() => ws().message(calibratedPayload(0.8)))
    expect(screen.getByText('Engagement: 0.80')).toBeInTheDocument()

    await user.click(screen.getByLabelText('Message')) // focus -> freeze at 0.80
    act(() => ws().message(calibratedPayload(0.2))) // live score moves while the user types
    expect(screen.getByText('Engagement: 0.80')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Message'), 'What is a neuron?')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    await screen.findByText('ok')

    expect(sentMessages()[1].content).toBe('[Engagement: 0.80]\nWhat is a neuron?')
    expect(document.body.textContent).not.toMatch(/\[Engagement/)
    await waitFor(() => expect(screen.getByText('Engagement: 0.20')).toBeInTheDocument())
  })

  it('works offline: no modal, "Backend not connected", and an unavailable tag', async () => {
    const user = userEvent.setup()
    render(<App />) // socket never opens
    expect(screen.getByText('Backend not connected')).toBeInTheDocument()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

    await user.type(screen.getByLabelText('Message'), 'hi')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    await screen.findByText('ok')
    expect(sentMessages()[1].content).toBe('[Engagement: unavailable]\nhi')
  })

  it('falls back to an unavailable tag if the backend drops after calibration', async () => {
    const user = userEvent.setup()
    render(<App />)
    act(() => ws().open())
    act(() => ws().message(calibratedPayload(0.8)))
    act(() => ws().drop())

    expect(screen.getByText('Backend not connected')).toBeInTheDocument()
    await user.type(screen.getByLabelText('Message'), 'hi')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    await screen.findByText('ok')
    expect(sentMessages()[1].content).toBe('[Engagement: unavailable]\nhi')
  })
})
