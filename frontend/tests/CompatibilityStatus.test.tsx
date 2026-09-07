import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { CompatibilityStatus } from '@/components/upload/CompatibilityStatus'
import type { CompatibilityResponse } from '@/lib/api'

const compatible: CompatibilityResponse = {
  compatible: true,
  purpose: 'temporal',
  overlap_ratio: 1,
  same_crs: true,
  resampling_required: false,
  alignment_required: false,
  issues: [],
  notes: ['Images are co-registered and directly comparable.'],
}

const incompatible: CompatibilityResponse = {
  compatible: false,
  purpose: 'temporal',
  overlap_ratio: 0,
  same_crs: true,
  resampling_required: false,
  alignment_required: false,
  issues: [
    {
      code: 'no_overlap',
      message:
        'The images do not overlap geographically, so they cannot describe the same area.',
      severity: 'error',
    },
  ],
  notes: [],
}

describe('CompatibilityStatus', () => {
  it('renders nothing before a check has run', () => {
    const { container } = render(<CompatibilityStatus result={null} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('reports a compatible temporal pair with the overlap measurement', () => {
    render(<CompatibilityStatus result={compatible} />)

    expect(screen.getByText('Compatible')).toBeInTheDocument()
    expect(screen.getByText('Temporal pair (T1 / T2)')).toBeInTheDocument()
    expect(screen.getByText('100%')).toBeInTheDocument()
    expect(screen.getByText(/co-registered and directly comparable/)).toBeInTheDocument()
  })

  it('reports an incompatible pair and explains why', () => {
    render(<CompatibilityStatus result={incompatible} />)

    expect(screen.getByText('Not compatible')).toBeInTheDocument()
    expect(screen.getByText(/do not overlap geographically/)).toBeInTheDocument()
    expect(screen.getByText(/Error:/)).toBeInTheDocument()
  })

  it('shows an accessible error when the check itself fails', () => {
    render(<CompatibilityStatus result={null} error="Compatibility check failed." />)
    expect(screen.getByRole('alert')).toHaveTextContent('Compatibility check failed.')
  })

  it('states clearly when overlap cannot be verified', () => {
    render(
      <CompatibilityStatus
        result={{ ...compatible, overlap_ratio: null, same_crs: null, alignment_required: true }}
      />,
    )
    expect(screen.getByText('Cannot be verified')).toBeInTheDocument()
    expect(screen.getByText('Unknown')).toBeInTheDocument()
  })
})
