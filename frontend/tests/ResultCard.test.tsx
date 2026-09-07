import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { ResultCard } from '@/components/results/ResultCard'

import {
  adaptedResult,
  completedResult,
  rejectedResult,
  unavailableResult,
} from './analysisFixtures'

describe('ResultCard', () => {
  it('shows a completed answer with task, confidence and selected tool', () => {
    render(<ResultCard result={completedResult} />)

    expect(screen.getByText('Completed')).toBeInTheDocument()
    expect(screen.getByText('Metadata Inspection')).toBeInTheDocument()
    expect(screen.getByText(/EPSG:4326/)).toBeInTheDocument()
    expect(screen.getByText('High')).toBeInTheDocument()

    // Scoped to the "Selected tool" line, since the tool id also appears in the trace.
    // The label and value are separate spans, so assert against the containing paragraph.
    const selectedToolLine = screen.getByText(/Selected tool:/).closest('p')
    expect(selectedToolLine?.textContent).toContain('raster_preprocessor')
    expect(selectedToolLine?.textContent).toContain('v1.0.0')
  })

  it('does NOT invent an answer when the specialist is unavailable', () => {
    render(<ResultCard result={unavailableResult} />)

    expect(screen.getByText('Not yet available')).toBeInTheDocument()
    expect(screen.getByText('No answer was produced for this request.')).toBeInTheDocument()
    expect(screen.getByText('Unable to determine')).toBeInTheDocument()
    // It still shows that routing worked and explains what is missing.
    expect(screen.getByText('Visual Question Answering')).toBeInTheDocument()
    expect(screen.getByText(/not implemented yet/)).toBeInTheDocument()
  })

  it('shows no confidence score when confidence cannot be determined', () => {
    render(<ResultCard result={unavailableResult} />)
    expect(screen.queryByText(/^score /)).not.toBeInTheDocument()
  })

  it('explains a rejected incompatible pair and keeps its evidence', () => {
    render(<ResultCard result={rejectedResult} />)

    expect(screen.getByText('Rejected')).toBeInTheDocument()
    // The answer explains the outcome; the issue restates the reason.
    expect(screen.getByText(/cannot be analyzed together/)).toBeInTheDocument()
    expect(screen.getAllByText(/do not overlap geographically/).length).toBeGreaterThan(0)
    expect(screen.getByText('Evidence (1)')).toBeInTheDocument()
  })

  it('surfaces warnings', () => {
    render(<ResultCard result={rejectedResult} />)
    expect(screen.getByText('Spatial overlap could not be confirmed.')).toBeInTheDocument()
  })

  it('exposes the execution trace on demand', async () => {
    render(<ResultCard result={completedResult} />)

    const summary = screen.getByText('Execution trace (8 steps)')
    await userEvent.click(summary)

    expect(screen.getByText('Task classification')).toBeInTheDocument()
    // The trace records the exact tool version and its safe parameter summary.
    expect(
      screen.getByText(/Step 1: raster_preprocessor v1\.0\.0 \(parameters: max_preview_side=1024\)/),
    ).toBeInTheDocument()
  })

  it('never displays reasoning-style content in the trace', () => {
    render(<ResultCard result={completedResult} />)
    const text = document.body.textContent?.toLowerCase() ?? ''
    expect(text).not.toContain('chain-of-thought')
    expect(text).not.toContain('my reasoning')
  })

  it('does not label a baseline result as adapted', () => {
    render(<ResultCard result={completedResult} />)
    expect(screen.queryByText('Adapted model')).not.toBeInTheDocument()
  })

  it('labels a result produced by the Phase 5 adapted model', () => {
    render(<ResultCard result={adaptedResult} />)

    expect(screen.getByText('Adapted model')).toBeInTheDocument()
    // The reason the adapted model was chosen is shown, so the choice is auditable.
    expect(
      screen.getByText(/selected the Phase 5 adapted model/),
    ).toBeInTheDocument()
  })

  it('surfaces the adaptation limitation as a warning', () => {
    render(<ResultCard result={adaptedResult} />)
    // The training subset is named in both the warning and the model evidence.
    expect(screen.getAllByText(/Lithuania\/Summer/).length).toBeGreaterThan(0)
    expect(
      screen.getByText(/benefit on other regions, seasons, or sensors is not established/),
    ).toBeInTheDocument()
  })

  it('offers a downloadable report linked to this analysis', () => {
    render(<ResultCard result={completedResult} />)
    const link = screen.getByRole('link', { name: /download report/i })
    expect(link).toHaveAttribute('href', expect.stringContaining('/analysis/an-1/report'))
    expect(link).toHaveAttribute('download')
  })

  it('offers a report even for a rejected analysis (it explains why)', () => {
    render(<ResultCard result={rejectedResult} />)
    expect(screen.getByRole('link', { name: /download report/i })).toHaveAttribute(
      'href',
      expect.stringContaining('/analysis/an-3/report'),
    )
  })
})
