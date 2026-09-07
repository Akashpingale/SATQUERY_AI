import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { VisualizationPanel } from '@/components/panels/VisualizationPanel'
import type { EvidenceOut, RasterAsset } from '@/lib/api'

import { georeferencedAsset, plainAsset } from './fixtures'

const t1: RasterAsset = { ...georeferencedAsset, id: 'asset-t1', role: 't1', original_filename: 'site_2019.tif', preview_url: '/api/v1/uploads/asset-t1/preview' }
const t2: RasterAsset = { ...georeferencedAsset, id: 'asset-t2', role: 't2', original_filename: 'site_2024.tif', preview_url: '/api/v1/uploads/asset-t2/preview' }

const changeMap: EvidenceOut = {
  kind: 'change_map',
  label: 'Measured change map (later image with flagged areas)',
  asset_id: 'asset-t2',
  detail: { changed_percent: 13.02, region_count: 1 },
  reference: '/api/v1/evidence/aaaabbbbccccdddd',
}

const strip: EvidenceOut = {
  kind: 'temporal_compare',
  label: 'T1 / T2 / change on the shared grid',
  asset_id: null,
  detail: {},
  reference: '/api/v1/evidence/1111222233334444',
}

describe('VisualizationPanel', () => {
  it('shows only the single-image view when there is nothing to compare', () => {
    render(<VisualizationPanel asset={georeferencedAsset} />)

    expect(screen.queryByRole('tablist')).not.toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Preview of scene_optical.tif/ })).toBeInTheDocument()
  })

  it('offers a T1/T2 comparison once both temporal slots are filled', async () => {
    render(<VisualizationPanel asset={t1} t1={t1} t2={t2} />)

    const tablist = screen.getByRole('tablist', { name: 'Visualization mode' })
    expect(tablist).toBeInTheDocument()

    await userEvent.click(screen.getByRole('tab', { name: 'Compare T1 / T2' }))

    expect(screen.getByRole('img', { name: /T1 · earlier: preview of site_2019.tif/ })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /T2 · later: preview of site_2024.tif/ })).toBeInTheDocument()
  })

  it('labels which date is earlier so the direction of change is unambiguous', async () => {
    render(<VisualizationPanel asset={t1} t1={t1} t2={t2} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Compare T1 / T2' }))

    expect(screen.getByText('T1 · earlier')).toBeInTheDocument()
    expect(screen.getByText('T2 · later')).toBeInTheDocument()
  })

  it('states when a temporal image has no preview instead of showing a broken image', async () => {
    const noPreview: RasterAsset = { ...t2, preview_url: null }
    render(<VisualizationPanel asset={t1} t1={t1} t2={noPreview} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Compare T1 / T2' }))

    expect(screen.getByText(/No preview was rendered/)).toBeInTheDocument()
  })

  it('reports missing georeferencing per date rather than assuming it', async () => {
    const plainT2: RasterAsset = { ...plainAsset, id: 'asset-t2', role: 't2', preview_url: '/p' }
    render(<VisualizationPanel asset={t1} t1={t1} t2={plainT2} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Compare T1 / T2' }))

    expect(screen.getByText('No georeferencing')).toBeInTheDocument()
  })

  it('surfaces the change evidence automatically when an analysis produces it', () => {
    render(<VisualizationPanel asset={t1} t1={t1} t2={t2} evidence={[changeMap, strip]} />)

    const changeTab = screen.getByRole('tab', { name: 'Change evidence' })
    expect(changeTab).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('img', { name: /Measured change map/ })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /T1 \/ T2 \/ change/ })).toBeInTheDocument()
  })

  it('explains what the tinted areas on the change map mean', () => {
    render(<VisualizationPanel asset={t1} t1={t1} t2={t2} evidence={[changeMap]} />)

    expect(screen.getByText(/exceeded the measured change threshold/)).toBeInTheDocument()
  })

  it('does not offer a change view when the analysis produced no change artifacts', () => {
    const nonTemporal: EvidenceOut = {
      kind: 'scores',
      label: 'Concept probabilities',
      asset_id: 'asset-t1',
      detail: {},
      reference: null,
    }
    render(<VisualizationPanel asset={t1} t1={t1} t2={t2} evidence={[nonTemporal]} />)

    expect(screen.queryByRole('tab', { name: 'Change evidence' })).not.toBeInTheDocument()
  })

  it('lets the user switch back to a single image from the change view', async () => {
    render(<VisualizationPanel asset={t1} t1={t1} t2={t2} evidence={[changeMap]} />)

    await userEvent.click(screen.getByRole('tab', { name: 'Single image' }))

    expect(screen.getByRole('tab', { name: 'Single image' })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    expect(screen.queryByRole('img', { name: /Measured change map/ })).not.toBeInTheDocument()
  })
})


describe('VisualizationPanel: optical + SAR', () => {
  const fusionComposite: EvidenceOut = {
    kind: 'fusion_composite',
    label: 'Optical+SAR false-colour composite',
    asset_id: null,
    detail: { note: 'Reddish = SAR-strong.' },
    reference: '/api/v1/evidence/ffff0000ffff0000',
  }
  const alignedSar: EvidenceOut = {
    kind: 'aligned_sar',
    label: 'Aligned SAR (scene_sar.tif)',
    asset_id: null,
    detail: {},
    reference: '/api/v1/evidence/1234abcd1234abcd',
  }

  it('surfaces an Optical + SAR view automatically when fusion evidence is present', () => {
    render(<VisualizationPanel asset={georeferencedAsset} evidence={[fusionComposite, alignedSar]} />)
    const tab = screen.getByRole('tab', { name: 'Optical + SAR' })
    expect(tab).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('img', { name: /false-colour composite/i })).toBeInTheDocument()
  })

  it('explains the composite channels', () => {
    render(<VisualizationPanel asset={georeferencedAsset} evidence={[fusionComposite]} />)
    expect(screen.getByText(/red is SAR backscatter/i)).toBeInTheDocument()
  })

  it('does not offer an Optical + SAR view without fusion evidence', () => {
    render(<VisualizationPanel asset={georeferencedAsset} />)
    expect(screen.queryByRole('tab', { name: 'Optical + SAR' })).not.toBeInTheDocument()
  })
})
