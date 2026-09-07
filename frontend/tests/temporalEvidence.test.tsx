import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { EvidencePanel } from '@/components/results/EvidencePanel'
import type { EvidenceOut } from '@/lib/api'

/**
 * Phase 6 temporal evidence, shaped exactly as the backend emits it.
 *
 * These fixtures mirror real measured payloads (a 1200-pixel changed block on a 10 m grid,
 * so 120,000 m²). They are clearly labelled test fixtures, not model output.
 */
const changeMapEvidence: EvidenceOut = {
  kind: 'change_map',
  label: 'Measured change map (later image with flagged areas)',
  asset_id: 'asset-t2',
  detail: {
    method: 'change_vector_analysis',
    algorithm_version: 'cva-robust-1.1.0',
    changed_pixels: 1200,
    valid_pixels: 9216,
    changed_percent: 13.02,
    changed_area_m2: 120000,
    region_count: 2,
    coherence: 1,
    threshold: 0.2541,
    threshold_method: 'noise_floor',
    registration_peak_correlation: 0.994,
    registration_peak_offset: [0, 0],
    co_registration: ['resampled both images onto a shared 96 x 96 grid using average resampling'],
    georeferenced: true,
  },
  reference: '/api/v1/evidence/aaaabbbbccccdddd',
}

const stripEvidence: EvidenceOut = {
  kind: 'temporal_compare',
  label: 'T1 / T2 / change on the shared grid',
  asset_id: null,
  detail: {
    panels: ['T1 (earlier)', 'T2 (later)', 'measured change'],
    note: 'All three panels share one grid, so the same position is the same ground location in each.',
  },
  reference: '/api/v1/evidence/1111222233334444',
}

const changeRegionEvidence: EvidenceOut = {
  kind: 'change_region',
  label: 'Changed region 1 - 1200 px in the northern western quadrant',
  asset_id: 'asset-t2',
  detail: {
    index: 1,
    pixel_count: 1200,
    box: [30, 20, 69, 49],
    centroid: [49.5, 34.5],
    mean_magnitude: 0.9791,
    brightness_delta: 3364.2,
    fraction_of_scene: 0.1302,
    area_m2: 120000,
    centroid_lonlat: [75.00483, 13.56441],
    position: 'northern western quadrant',
    units: 'square metres',
  },
  reference: null,
}

const measuredIndices: EvidenceOut = {
  kind: 'indices',
  label: 'Spectral index change',
  asset_id: null,
  detail: {
    indices: [
      {
        name: 'NDVI',
        available: true,
        reason: '',
        mean_before: 0.622,
        mean_after: 0.489,
        delta: -0.133,
        delta_in_changed: -0.872,
      },
    ],
    note: 'Computed from bands the rasters explicitly declare, on unnormalised reflectance.',
  },
  reference: null,
}

const unavailableIndices: EvidenceOut = {
  kind: 'indices',
  label: 'Spectral indices not available',
  asset_id: null,
  detail: {
    indices: [
      {
        name: 'NDVI',
        available: false,
        reason: 'red and near-infrared bands are not both declared',
        mean_before: null,
        mean_after: null,
        delta: null,
        delta_in_changed: null,
      },
    ],
    note: 'Indices are only computed when the raster declares which band is which.',
  },
  reference: null,
}

const semanticReading: EvidenceOut = {
  kind: 'scores',
  label: 'Land cover of the largest changed region, before and after',
  asset_id: null,
  detail: {
    available: true,
    reason: '',
    variant: 'baseline',
    before_label: 'farmland',
    before_probability: 0.512,
    after_label: 'built-up area',
    after_probability: 0.447,
    transition_supported: true,
    note: 'Zero-shot land-cover scoring of the changed region, before and after. Indicative only: no change-type classifier was trained or evaluated.',
  },
  reference: null,
}

describe('EvidencePanel: temporal change evidence', () => {
  it('renders the change map as an image with alt text', () => {
    render(<EvidencePanel evidence={[changeMapEvidence]} />)

    expect(screen.getByText('Change map')).toBeInTheDocument()
    const image = screen.getByRole('img', { name: /Measured change map/ })
    expect(image).toHaveAttribute('src', expect.stringContaining('/api/v1/evidence/'))
  })

  it('shows the measured extent with its real units', () => {
    render(<EvidencePanel evidence={[changeMapEvidence]} />)

    expect(screen.getByText('Area changed')).toBeInTheDocument()
    expect(screen.getByText('13.02% (120,000 m²)')).toBeInTheDocument()
    expect(screen.getByText('Coherent regions')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
  })

  it('reports the threshold and how it was chosen, not just a number', () => {
    render(<EvidencePanel evidence={[changeMapEvidence]} />)
    expect(screen.getByText('0.2541 (noise_floor)')).toBeInTheDocument()
  })

  it('reports registration quality so a reader can judge the comparison', () => {
    render(<EvidencePanel evidence={[changeMapEvidence]} />)

    expect(screen.getByText('Registration correlation')).toBeInTheDocument()
    expect(screen.getByText(/0\.994 at offset 0, 0 px/)).toBeInTheDocument()
  })

  it('discloses the co-registration operations that were applied', () => {
    render(<EvidencePanel evidence={[changeMapEvidence]} />)
    expect(screen.getByText(/resampled both images onto a shared 96 x 96 grid/)).toBeInTheDocument()
  })

  it('omits a square-metre area when the CRS does not support one', () => {
    const geographic: EvidenceOut = {
      ...changeMapEvidence,
      detail: { ...changeMapEvidence.detail, changed_area_m2: null },
    }
    render(<EvidencePanel evidence={[geographic]} />)

    expect(screen.getByText('13.02% of the shared area')).toBeInTheDocument()
    expect(screen.queryByText(/m²/)).not.toBeInTheDocument()
  })

  it('renders the T1/T2/change strip with its shared-grid note', () => {
    render(<EvidencePanel evidence={[stripEvidence]} />)

    expect(screen.getByText('T1 / T2 / change')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /T1 \/ T2 \/ change/ })).toBeInTheDocument()
    expect(screen.getByText(/same position is the same ground location/)).toBeInTheDocument()
  })

  it('shows a changed region with its size, position and coordinates', () => {
    render(<EvidencePanel evidence={[changeRegionEvidence]} />)

    expect(screen.getByText('Changed region')).toBeInTheDocument()
    expect(screen.getByText('northern western quadrant')).toBeInTheDocument()
    expect(screen.getByText(/1,200 px/)).toBeInTheDocument()
    expect(screen.getByText(/120,000 m²/)).toBeInTheDocument()
    // Latitude first, matching the label.
    expect(screen.getByText('13.56441, 75.00483')).toBeInTheDocument()
  })

  it('says brighter or darker rather than showing a bare number', () => {
    render(<EvidencePanel evidence={[changeRegionEvidence]} />)
    expect(screen.getByText(/brighter at T2/)).toBeInTheDocument()
  })

  it('states when a changed region has no coordinates instead of inventing them', () => {
    const ungeoreferenced: EvidenceOut = {
      ...changeRegionEvidence,
      detail: { ...changeRegionEvidence.detail, centroid_lonlat: null, area_m2: null },
    }
    render(<EvidencePanel evidence={[ungeoreferenced]} />)

    expect(screen.getByText('Unavailable (not georeferenced)')).toBeInTheDocument()
  })

  it('renders a measured spectral index with before, after and delta', () => {
    render(<EvidencePanel evidence={[measuredIndices]} />)

    expect(screen.getByText('NDVI')).toBeInTheDocument()
    expect(screen.getByText('measured')).toBeInTheDocument()
    expect(screen.getByText(/0\.622 → 0\.489/)).toBeInTheDocument()
    expect(screen.getByText(/-0\.872 in changed areas/)).toBeInTheDocument()
  })

  it('shows an unavailable index with its reason, so no trend is implied', () => {
    render(<EvidencePanel evidence={[unavailableIndices]} />)

    expect(screen.getByText('not available')).toBeInTheDocument()
    expect(
      screen.getByText('red and near-infrared bands are not both declared'),
    ).toBeInTheDocument()
  })

  it('labels the land-cover reading as indicative rather than a verdict', () => {
    render(<EvidencePanel evidence={[semanticReading]} />)

    expect(screen.getByText(/farmland/)).toBeInTheDocument()
    expect(screen.getByText(/built-up area/)).toBeInTheDocument()
    expect(screen.getByText(/no change-type classifier was trained or evaluated/)).toBeInTheDocument()
  })

  it('states plainly when the land-cover cause could not be determined', () => {
    const unavailable: EvidenceOut = {
      ...semanticReading,
      detail: {
        available: false,
        reason: 'the changed region is too small for a land-cover reading to be meaningful',
        transition_supported: false,
      },
    }
    render(<EvidencePanel evidence={[unavailable]} />)

    expect(screen.getByText(/reported as undetermined/)).toBeInTheDocument()
  })

  it('puts the change map before the tabular evidence so visuals lead', () => {
    render(
      <EvidencePanel
        evidence={[changeRegionEvidence, measuredIndices, changeMapEvidence]}
      />,
    )

    const list = screen.getByRole('list', { name: 'Evidence' })
    expect(list.children[0].textContent).toContain('Change map')
  })
})
