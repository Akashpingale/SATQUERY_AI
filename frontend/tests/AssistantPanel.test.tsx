import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AssistantPanel } from '@/components/panels/AssistantPanel'

import { completedResult } from './analysisFixtures'

const base = {
  hasImagery: true,
  imageCount: 1,
  isRunning: false,
  result: null,
  error: null,
  onAnalyze: vi.fn(),
  examples: [
    'What CRS and resolution does this image use?',
    'What land cover dominates this image?',
  ],
}

describe('AssistantPanel', () => {
  it('guides the user to upload first when no imagery exists', () => {
    render(<AssistantPanel {...base} hasImagery={false} imageCount={0} />)

    expect(screen.getByText(/Upload a raster first/)).toBeInTheDocument()
    expect(screen.getByLabelText('Your question')).toBeDisabled()
    expect(screen.getByRole('button', { name: /analyze/i })).toBeDisabled()
  })

  it('enables analysis once imagery is present and a question is typed', async () => {
    const onAnalyze = vi.fn()
    render(<AssistantPanel {...base} onAnalyze={onAnalyze} />)

    const analyze = screen.getByRole('button', { name: /analyze/i })
    expect(analyze).toBeDisabled()

    await userEvent.type(screen.getByLabelText('Your question'), 'What CRS is this?')
    expect(analyze).toBeEnabled()

    await userEvent.click(analyze)
    expect(onAnalyze).toHaveBeenCalledWith('What CRS is this?')
  })

  it('does not submit a whitespace-only query', async () => {
    const onAnalyze = vi.fn()
    render(<AssistantPanel {...base} onAnalyze={onAnalyze} />)

    await userEvent.type(screen.getByLabelText('Your question'), '    ')
    expect(screen.getByRole('button', { name: /analyze/i })).toBeDisabled()
    expect(onAnalyze).not.toHaveBeenCalled()
  })

  it('fills the query box from an example question', async () => {
    render(<AssistantPanel {...base} />)

    await userEvent.click(
      screen.getByRole('button', { name: 'What CRS and resolution does this image use?' }),
    )
    expect(screen.getByLabelText('Your question')).toHaveValue(
      'What CRS and resolution does this image use?',
    )
  })

  it('shows a running state while the agent works', () => {
    render(<AssistantPanel {...base} isRunning />)

    expect(screen.getByText('Analyzing…')).toBeInTheDocument()
    expect(screen.getByText(/classifying the query/)).toBeInTheDocument()
  })

  it('renders the result once available', () => {
    render(<AssistantPanel {...base} result={completedResult} />)

    expect(screen.getByText('Completed')).toBeInTheDocument()
    expect(screen.getByText(/EPSG:4326/)).toBeInTheDocument()
  })

  it('shows an accessible error when the request fails', () => {
    render(<AssistantPanel {...base} error="Cannot reach the SatQuery AI backend." />)

    expect(screen.getByRole('alert')).toHaveTextContent('Cannot reach the SatQuery AI backend.')
  })

  it('reflects the image count in the description', () => {
    render(<AssistantPanel {...base} imageCount={2} />)
    expect(screen.getByText(/about your 2 images/)).toBeInTheDocument()
  })
})
