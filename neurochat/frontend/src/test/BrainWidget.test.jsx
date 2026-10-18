import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import BrainWidget from '../components/BrainWidget.jsx'

afterEach(cleanup)

const quality = () => screen.getByTestId('signal-quality').dataset.quality

describe('BrainWidget', () => {
  it('shows "Backend not connected" when disconnected', () => {
    render(<BrainWidget connected={false} confidence={null} fused={null} />)
    expect(screen.getByText('Backend not connected')).toBeInTheDocument()
    expect(screen.queryByText(/Signal:/)).not.toBeInTheDocument()
  })

  it('shows connected status when connected', () => {
    render(<BrainWidget connected confidence={0.9} fused={0.5} />)
    expect(screen.getByText('Connected')).toBeInTheDocument()
    expect(screen.queryByText('Backend not connected')).not.toBeInTheDocument()
  })

  it('renders confidence as a percentage', () => {
    render(<BrainWidget connected confidence={0.913} fused={0.5} />)
    expect(screen.getByText('Signal: 91%')).toBeInTheDocument()
  })

  it('renders the fused score with two decimals', () => {
    render(<BrainWidget connected confidence={0.9} fused={0.7} />)
    expect(screen.getByText('Engagement: 0.70')).toBeInTheDocument()
  })

  it.each([
    [0.39, 'red'],
    [0.4, 'yellow'],
    [0.7, 'yellow'],
    [0.71, 'green'],
    [0, 'red'],
    [1, 'green'],
  ])('confidence %s is %s', (confidence, expected) => {
    render(<BrainWidget connected confidence={confidence} fused={0.5} />)
    expect(quality()).toBe(expected)
  })

  it('handles null confidence and null score without crashing', () => {
    render(<BrainWidget connected confidence={null} fused={null} />)
    expect(quality()).toBe('none')
    expect(screen.getByText('Signal: —')).toBeInTheDocument()
    expect(screen.getByText('Engagement: —')).toBeInTheDocument()
  })
})
