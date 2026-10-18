import { describe, expect, it } from 'vitest'
import { SYSTEM_PROMPT } from '../lib/systemPrompt.js'

describe('SYSTEM_PROMPT', () => {
  it('establishes the NeuroChat identity', () => {
    expect(SYSTEM_PROMPT).toContain('You are NeuroChat')
  })

  it('describes the engagement tag format', () => {
    expect(SYSTEM_PROMPT).toContain('[Engagement: 0.72]')
  })

  it('defines the low tier below 0.3', () => {
    expect(SYSTEM_PROMPT).toMatch(/below 0\.3 \(low engagement\)/)
  })

  it('defines the moderate tier from 0.3 to 0.6', () => {
    expect(SYSTEM_PROMPT).toMatch(/0\.3 to 0\.6 \(moderate engagement\)/)
  })

  it('defines the high tier above 0.6', () => {
    expect(SYSTEM_PROMPT).toMatch(/above 0\.6 \(high engagement\)/)
  })

  it('falls back to a balanced style when the score is unavailable', () => {
    expect(SYSTEM_PROMPT).toContain('[Engagement: unavailable]')
    expect(SYSTEM_PROMPT).toMatch(/unavailable\],? use a balanced style/)
  })

  it('uses storytelling and Socratic questions for low engagement', () => {
    expect(SYSTEM_PROMPT).toMatch(/storytelling/)
    expect(SYSTEM_PROMPT).toMatch(/Socratic/)
  })

  it('uses technical depth and bullet points for high engagement', () => {
    expect(SYSTEM_PROMPT).toMatch(/technical depth/)
    expect(SYSTEM_PROMPT).toMatch(/bullet points/)
  })

  it('forbids mentioning the score, EEG or brain signals', () => {
    expect(SYSTEM_PROMPT).toMatch(/NEVER mention the engagement score, the tag, EEG, brain signals/)
  })

  it('asks for Markdown', () => {
    expect(SYSTEM_PROMPT).toMatch(/Respond in Markdown\.$/)
  })
})
