import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import CalibrationModal, { formatCountdown } from '../components/CalibrationModal.jsx'

afterEach(cleanup)

describe('formatCountdown', () => {
  it.each([
    [120, '2:00'],
    [119.2, '2:00'],
    [75, '1:15'],
    [59, '0:59'],
    [7, '0:07'],
    [0, '0:00'],
    [-3, '0:00'],
  ])('%s s -> %s', (seconds, expected) => {
    expect(formatCountdown(seconds)).toBe(expected)
  })
})

describe('CalibrationModal', () => {
  it('idle: explains both tasks and starts on click', async () => {
    const onStart = vi.fn()
    render(<CalibrationModal state="idle" secondsLeft={120} onStart={onStart} />)
    expect(screen.getByText(/Relaxation/)).toBeInTheDocument()
    expect(screen.getByText(/Word association/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Start' }))
    expect(onStart).toHaveBeenCalledTimes(1)
  })

  it('phase 1: relaxation instruction and a 2:00 countdown, no Start button', () => {
    render(<CalibrationModal state="phase1" secondsLeft={120} onStart={() => {}} />)
    expect(screen.getByText('Relax. Sit still and clear your mind.')).toBeInTheDocument()
    expect(screen.getByLabelText('Time remaining')).toHaveTextContent('2:00')
    expect(screen.queryByRole('button', { name: 'Start' })).not.toBeInTheDocument()
  })

  it('phase 2: word association instruction, the example and a countdown', () => {
    render(<CalibrationModal state="phase2" secondsLeft={65} onStart={() => {}} />)
    expect(
      screen.getByText('Word association. Think of a word, then a new word starting with its last letter.'),
    ).toBeInTheDocument()
    expect(screen.getByText(/apple → elephant → table/)).toBeInTheDocument()
    expect(screen.getByLabelText('Time remaining')).toHaveTextContent('1:05')
  })

  it('complete: renders nothing', () => {
    const { container } = render(<CalibrationModal state="complete" secondsLeft={0} onStart={() => {}} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('is a full-screen dialog', () => {
    render(<CalibrationModal state="idle" secondsLeft={120} onStart={() => {}} />)
    expect(screen.getByRole('dialog')).toBeInTheDocument()
  })
})
