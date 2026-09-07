import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { EvidencePanel } from '@/components/results/EvidencePanel'
import type { EvidenceOut } from '@/lib/api'

const scoresEvidence: EvidenceOut = {
  kind: 'scores',
  label: 'Concept probabilities from the model',
  asset_id: 'a1',
  detail: {
    model_id: 'flax-community/clip-rsicd-v2',
    concepts: [
      { key: 'farmland', label: 'farmland', probability: 0.562 },
      { key: 'water', label: 'water', probability: 0.13 },
      { key: 'forest', label: 'forest', probability: 0.067 },
    ],
    note: 'Softmax over the model concept vocabulary; not calibrated.',
  },
  reference: null,
}

const overlayEvidence: EvidenceOut = {
  kind: 'overlay',
  label: "Detected regions for 'a body of water'",
  asset_id: 'a1',
  detail: { box_count: 2, method: 'clip_patch_scoring', grid: 4 },
  reference: '/api/v1/evidence/abcdef1234567890',
}

const boxEvidence: EvidenceOut = {
  kind: 'box',
  label: 'Region 1 (42%)',
  asset_id: 'a1',
  detail: {
    index: 1,
    score: 0.42,
    analyzed_pixel_box: [10, 20, 80, 90],
    geo: { center_lonlat: [77.63764, 12.93241], crs: 'EPSG:4326' },
  },
  reference: null,
}

const regionEvidence: EvidenceOut = {
  kind: 'region',
  label: "Coarse region for 'a body of water'",
  asset_id: 'a1',
  detail: {
    index: 1,
    method: 'clip_patch_scoring',
    grid: '4x4',
    tiles_merged: 3,
    tile_contrast: 0.7396,
    analyzed_pixel_box: [0, 96, 288, 192],
    geo: { center_lonlat: [77.6376, 12.9324] },
  },
  reference: null,
}

const modelEvidence: EvidenceOut = {
  kind: 'model',
  label: 'Model provenance',
  asset_id: null,
  detail: {
    model_id: 'google/owlvit-base-patch32',
    device: 'cpu',
    note: 'OWL-ViT open-vocabulary detection; boxes are model output.',
  },
  reference: null,
}

describe('EvidencePanel', () => {
  it('states plainly when there is no evidence', () => {
    render(<EvidencePanel evidence={[]} />)
    expect(screen.getByText(/No inspectable evidence/)).toBeInTheDocument()
  })

  it('renders concept scores with percentages as text, not colour alone', () => {
    render(<EvidencePanel evidence={[scoresEvidence]} />)

    expect(screen.getByText('Model scores')).toBeInTheDocument()
    expect(screen.getByText('farmland')).toBeInTheDocument()
    expect(screen.getByText('56%')).toBeInTheDocument()
    expect(screen.getByText('13%')).toBeInTheDocument()
    // Each bar is labelled for assistive tech.
    expect(screen.getByLabelText('farmland: 56 percent')).toBeInTheDocument()
  })

  it('surfaces the uncalibrated-score note', () => {
    render(<EvidencePanel evidence={[scoresEvidence]} />)
    expect(screen.getByText(/not calibrated/)).toBeInTheDocument()
  })

  it('renders the overlay image with alt text', () => {
    render(<EvidencePanel evidence={[overlayEvidence]} />)

    const image = screen.getByRole('img', { name: /Detected regions for 'a body of water'/ })
    expect(image).toHaveAttribute('src', expect.stringContaining('/api/v1/evidence/'))
  })

  it('shows detection score, pixel box and coordinates for a box', () => {
    render(<EvidencePanel evidence={[boxEvidence]} />)

    expect(screen.getByText('Detection score')).toBeInTheDocument()
    expect(screen.getByText('42%')).toBeInTheDocument()
    expect(screen.getByText('10, 20 → 80, 90')).toBeInTheDocument()
    // Latitude first, matching the displayed label.
    expect(screen.getByText('12.93241, 77.63764')).toBeInTheDocument()
  })

  it('labels a coarse region with its tile contrast and grid', () => {
    render(<EvidencePanel evidence={[regionEvidence]} />)

    expect(screen.getByText('Estimated region')).toBeInTheDocument()
    expect(screen.getByText('Tile contrast')).toBeInTheDocument()
    expect(screen.getByText('74%')).toBeInTheDocument()
    expect(screen.getByText('4x4 (3 merged)')).toBeInTheDocument()
  })

  it('states when coordinates are unavailable instead of inventing them', () => {
    const ungeoreferenced: EvidenceOut = {
      ...boxEvidence,
      detail: { index: 1, score: 0.3, analyzed_pixel_box: [1, 2, 3, 4] },
    }
    render(<EvidencePanel evidence={[ungeoreferenced]} />)

    expect(screen.getByText(/Unavailable \(image not georeferenced\)/)).toBeInTheDocument()
  })

  it('records model provenance', () => {
    render(<EvidencePanel evidence={[modelEvidence]} />)

    expect(screen.getByText('Model')).toBeInTheDocument()
    expect(screen.getByText('google/owlvit-base-patch32')).toBeInTheDocument()
    expect(screen.getByText('cpu')).toBeInTheDocument()
  })

  it('shows the overlay before other evidence so visuals lead', () => {
    render(<EvidencePanel evidence={[modelEvidence, boxEvidence, overlayEvidence]} />)

    const list = screen.getByRole('list', { name: 'Evidence' })
    const items = Array.from(list.children)
    expect(items[0].textContent).toContain('Detected regions')
  })

  it('renders every supplied evidence item', () => {
    render(
      <EvidencePanel
        evidence={[overlayEvidence, boxEvidence, regionEvidence, scoresEvidence, modelEvidence]}
      />,
    )
    // Count only top-level entries: score bars render their own nested list items.
    const list = screen.getByRole('list', { name: 'Evidence' })
    expect(list.children).toHaveLength(5)
  })
})
