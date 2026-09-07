import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { ConfidenceBadge } from '@/components/results/ConfidenceBadge'
import { ExecutionTrace } from '@/components/results/ExecutionTrace'

describe('ConfidenceBadge', () => {
  it('shows the level as text, not colour alone', () => {
    render(
      <ConfidenceBadge confidence={{ level: 'High', score: 0.91, signals: ['strength 1.00'] }} />,
    )
    expect(screen.getByText('High')).toBeInTheDocument()
    expect(screen.getByText('score 0.91')).toBeInTheDocument()
  })

  it('omits the score when confidence cannot be determined', () => {
    render(
      <ConfidenceBadge
        confidence={{ level: 'Unable to determine', score: null, signals: ['no result'] }}
      />,
    )
    expect(screen.getByText('Unable to determine')).toBeInTheDocument()
    expect(screen.queryByText(/score/)).not.toBeInTheDocument()
  })

  it('lists the signals that produced the level so it is auditable', async () => {
    render(
      <ConfidenceBadge
        confidence={{
          level: 'Medium',
          score: 0.55,
          signals: ['Query routing score 0.60', 'Some inputs lack georeferencing'],
        }}
      />,
    )

    await userEvent.click(screen.getByText(/Why this confidence \(2 signals\)/))
    expect(screen.getByText('Some inputs lack georeferencing')).toBeInTheDocument()
  })
})

describe('ExecutionTrace', () => {
  it('renders each step with a textual status label', () => {
    render(
      <ExecutionTrace
        trace={[
          { step: 'Validation', status: 'ok', detail: 'validated' },
          { step: 'Execution (vqa)', status: 'skipped', detail: 'not available yet' },
          { step: 'Workflow', status: 'failed', detail: 'stopped' },
        ]}
      />,
    )

    expect(screen.getByText('Validation')).toBeInTheDocument()
    expect(screen.getByText(/· OK/)).toBeInTheDocument()
    expect(screen.getByText(/· Skipped/)).toBeInTheDocument()
    expect(screen.getByText(/· Failed/)).toBeInTheDocument()
  })

  it('handles an empty trace', () => {
    render(<ExecutionTrace trace={[]} />)
    expect(screen.getByText('No trace was recorded.')).toBeInTheDocument()
  })
})
