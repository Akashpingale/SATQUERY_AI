import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { MetadataPanel } from '@/components/upload/MetadataPanel'

import { georeferencedAsset, plainAsset } from './fixtures'

describe('MetadataPanel', () => {
  it('shows real raster metadata for a georeferenced image', () => {
    render(<MetadataPanel asset={georeferencedAsset} />)

    expect(screen.getByText('scene_optical.tif')).toBeInTheDocument()
    expect(screen.getByText('512 x 512 px')).toBeInTheDocument()
    expect(screen.getByText('EPSG:4326')).toBeInTheDocument()
    expect(screen.getByText('GTiff')).toBeInTheDocument()
    expect(screen.getByText(/Georeferenced/)).toBeInTheDocument()
  })

  it('states plainly when an image has no georeferencing instead of inventing values', () => {
    render(<MetadataPanel asset={plainAsset} />)

    expect(screen.getByText(/No georeferencing/)).toBeInTheDocument()
    expect(screen.getByText('None')).toBeInTheDocument()
    // Resolution and bounds both report the absence rather than a fabricated number.
    expect(screen.getAllByText('Not georeferenced').length).toBeGreaterThanOrEqual(2)
  })

  it('surfaces validation warnings to the user', () => {
    render(<MetadataPanel asset={plainAsset} />)

    expect(screen.getByText(/Warning:/)).toBeInTheDocument()
    expect(screen.getByText(/coordinates, areas, and pair alignment/)).toBeInTheDocument()
  })

  it('lists per-band details', () => {
    render(<MetadataPanel asset={georeferencedAsset} />)
    expect(screen.getByText('Band details (3)')).toBeInTheDocument()
  })
})
