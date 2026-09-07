import { describe, expect, it } from 'vitest'

import { type InputContext, exampleQuestions } from '@/lib/examples'

const EMPTY: InputContext = {
  hasSingle: false,
  hasOptical: false,
  hasSar: false,
  hasTemporalPair: false,
  hasOpticalSarPair: false,
}

describe('exampleQuestions', () => {
  it('offers generic metadata questions when nothing is uploaded', () => {
    const examples = exampleQuestions(EMPTY)
    expect(examples).toContain('What CRS and resolution does this image use?')
    expect(examples.some((q) => /changed between/i.test(q))).toBe(false)
  })

  it('offers optical scene questions for a single optical image', () => {
    const examples = exampleQuestions({ ...EMPTY, hasOptical: true })
    expect(examples).toContain('What land cover dominates this image?')
    expect(examples.some((q) => q.toLowerCase().includes('sar'))).toBe(false)
  })

  it('offers SAR-specific questions for a lone SAR image', () => {
    const examples = exampleQuestions({ ...EMPTY, hasSar: true })
    expect(examples.some((q) => q.includes('SAR'))).toBe(true)
    expect(examples).not.toContain('What changed between these two dates?')
  })

  it('offers temporal questions for a T1/T2 pair', () => {
    const examples = exampleQuestions({ ...EMPTY, hasTemporalPair: true })
    expect(examples).toContain('What changed between these two dates?')
    expect(examples.some((q) => /complementary/i.test(q))).toBe(false)
  })

  it('offers cross-modal questions for an optical+SAR pair', () => {
    const examples = exampleQuestions({
      ...EMPTY,
      hasOptical: true,
      hasSar: true,
      hasOpticalSarPair: true,
    })
    expect(examples.some((q) => /complementary/i.test(q))).toBe(true)
    expect(examples.some((q) => /which modality/i.test(q))).toBe(true)
  })

  it('prefers cross-modal over temporal when both could apply', () => {
    const examples = exampleQuestions({
      ...EMPTY,
      hasOpticalSarPair: true,
      hasTemporalPair: true,
    })
    expect(examples.some((q) => /complementary/i.test(q))).toBe(true)
  })

  it('never proposes more than the limit', () => {
    expect(exampleQuestions({ ...EMPTY, hasOpticalSarPair: true }, 3)).toHaveLength(3)
  })

  it('only proposes questions the current inputs can satisfy (no temporal chip for single image)', () => {
    const examples = exampleQuestions({ ...EMPTY, hasOptical: true })
    expect(examples.every((q) => !/between these|complementary/i.test(q))).toBe(true)
  })
})
