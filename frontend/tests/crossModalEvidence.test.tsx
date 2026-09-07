import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { EvidencePanel } from '@/components/results/EvidencePanel'
import type { EvidenceOut } from '@/lib/api'

/** Phase 7 optical+SAR cross-modal evidence, shaped as the backend emits it (test fixtures). */
const composite: EvidenceOut = {
  kind: 'fusion_composite',
  label: 'Optical+SAR false-colour composite (R=SAR, G/B=optical)',
  asset_id: null,
  detail: {
    correlation: 0.85,
    complementarity: 0.15,
    sar_representation: 'log10-scaled backscatter (uncalibrated; relative intensity only)',
    note: 'Reddish = SAR-strong (rough/structured); cyan = optically bright, low SAR return.',
  },
  reference: '/api/v1/evidence/aaaabbbbccccdddd',
}

const modalityComparison: EvidenceOut = {
  kind: 'modality_comparison',
  label: 'Optical vs SAR measured features',
  asset_id: null,
  detail: {
    optical: {
      modality: 'optical',
      mean_level: 0.76,
      spread: 0.331,
      texture: 0.0376,
      structured_fraction: 0.0625,
      spectral_variance: 0.668,
      ndvi_mean: 0.5011,
    },
    sar: {
      modality: 'sar',
      mean_level: 0.5006,
      spread: 0.2524,
      texture: 0.0594,
      structured_fraction: 0.1013,
      dark_fraction: 0.25,
    },
    correlation: 0.85,
    complementarity: 0.15,
    sar_representation: 'log10-scaled backscatter (uncalibrated; relative intensity only)',
    sar_calibrated: false,
    note: 'SAR features are measured backscatter/roughness statistics, not land-cover classes.',
  },
  reference: null,
}

const water: EvidenceOut = {
  kind: 'cross_modal_water',
  label: 'Optical vs SAR water agreement',
  asset_id: null,
  detail: {
    available: true,
    optical_water_fraction: 0.156,
    sar_dark_fraction: 0.25,
    agreement_iou: 0.625,
    agree_fraction: 0.156,
    optical_only_fraction: 0.0,
    sar_only_fraction: 0.094,
  },
  reference: null,
}

const region: EvidenceOut = {
  kind: 'complementary_region',
  label: 'Complementary region 1 (sar structure) - 636 px in the northern part',
  asset_id: 'sar-1',
  detail: {
    index: 1,
    kind: 'sar_structure',
    pixel_count: 636,
    box: [25, 20, 70, 55],
    fraction_of_scene: 0.039,
    position: 'northern part',
    area_m2: 63600,
    centroid_lonlat: [75.00434, 13.5651],
    units: 'square metres',
  },
  reference: null,
}

describe('EvidencePanel: cross-modal evidence', () => {
  it('renders the optical+SAR composite as an image with its legend', () => {
    render(<EvidencePanel evidence={[composite]} />)
    expect(screen.getByText('Optical+SAR composite')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /false-colour composite/i })).toBeInTheDocument()
    expect(screen.getByText(/Reddish = SAR-strong/)).toBeInTheDocument()
  })

  it('shows optical and SAR features side by side with their distinct keys', () => {
    render(<EvidencePanel evidence={[modalityComparison]} />)
    expect(screen.getByText('Optical vs SAR')).toBeInTheDocument()
    expect(screen.getByText('NDVI (optical)')).toBeInTheDocument()
    expect(screen.getByText('Dark fraction (SAR)')).toBeInTheDocument()
    expect(screen.getByText(/Correlation 0.85/)).toBeInTheDocument()
  })

  it('states the SAR representation is uncalibrated', () => {
    render(<EvidencePanel evidence={[modalityComparison]} />)
    expect(screen.getByText(/uncalibrated/)).toBeInTheDocument()
  })

  it('reports water agreement as measured fractions and IoU', () => {
    render(<EvidencePanel evidence={[water]} />)
    expect(screen.getByText('Water agreement')).toBeInTheDocument()
    expect(screen.getByText('Agreement (IoU)')).toBeInTheDocument()
    expect(screen.getByText('Agree on water')).toBeInTheDocument()
    expect(screen.getByText('15.6% of the scene')).toBeInTheDocument()
    expect(screen.getByText('SAR-dark, not optical water')).toBeInTheDocument()
  })

  it('states plainly when water agreement is not computable', () => {
    const unavailable: EvidenceOut = {
      ...water,
      detail: { available: false, reason: 'NDWI is not computable (bands not declared)' },
    }
    render(<EvidencePanel evidence={[unavailable]} />)
    expect(screen.getByText(/NDWI is not computable/)).toBeInTheDocument()
  })

  it('shows a complementary region with its type, size and coordinates', () => {
    render(<EvidencePanel evidence={[region]} />)
    expect(screen.getByText('Complementary region')).toBeInTheDocument()
    expect(screen.getByText('sar structure')).toBeInTheDocument()
    // "636 px" appears in both the item label and the size cell, so assert on the size cell.
    expect(screen.getByText('636 px · 63,600 m²')).toBeInTheDocument()
    expect(screen.getByText('13.56510, 75.00434')).toBeInTheDocument()
  })

  it('states when a complementary region has no coordinates instead of inventing them', () => {
    const ungeoreferenced: EvidenceOut = {
      ...region,
      detail: { ...region.detail, centroid_lonlat: null, area_m2: null },
    }
    render(<EvidencePanel evidence={[ungeoreferenced]} />)
    expect(screen.getByText('Unavailable (not georeferenced)')).toBeInTheDocument()
  })

  it('leads with the fused composite when several evidence kinds are present', () => {
    render(<EvidencePanel evidence={[modalityComparison, water, region, composite]} />)
    const list = screen.getByRole('list', { name: 'Evidence' })
    expect(list.children[0].textContent).toContain('Optical+SAR composite')
  })
})
