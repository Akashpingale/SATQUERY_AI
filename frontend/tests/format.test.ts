import { describe, expect, it } from 'vitest'

import {
  formatArea,
  formatBounds,
  formatBytes,
  formatCount,
  formatCrs,
  formatDimensions,
  formatResolution,
} from '@/lib/format'

import { georeferencedAsset, plainAsset } from './fixtures'

describe('format helpers', () => {
  it('formats byte sizes', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
    expect(formatBytes(786_432)).toBe('768 KB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })

  it('formats dimensions and falls back when unknown', () => {
    expect(formatDimensions(georeferencedAsset)).toBe('512 x 512 px')
    expect(formatDimensions({ ...georeferencedAsset, width: null })).toBe('—')
  })

  it('labels degree-based resolution without converting to metres', () => {
    // Converting degrees to metres depends on latitude, so the unit is preserved.
    expect(formatResolution(georeferencedAsset)).toContain('°')
    expect(formatResolution(plainAsset)).toBe('Not georeferenced')
  })

  it('formats projected resolution in metres', () => {
    const utm = {
      ...georeferencedAsset,
      georeference: { ...georeferencedAsset.georeference, crs_epsg: 32643, resolution: [10, 10] },
    }
    expect(formatResolution(utm)).toBe('10.00m')
  })

  it('formats bounds and reports their absence honestly', () => {
    expect(formatBounds(georeferencedAsset)).toBe('77.5000, 12.7340 — 77.7560, 12.9900')
    expect(formatBounds(plainAsset)).toBe('Not georeferenced')
  })

  it('formats CRS', () => {
    expect(formatCrs(georeferencedAsset)).toBe('EPSG:4326')
    expect(formatCrs(plainAsset)).toBe('None')
  })
})

describe('formatCount', () => {
  it('groups thousands deterministically, independent of machine locale', () => {
    expect(formatCount(120000)).toBe('120,000')
    expect(formatCount(1200)).toBe('1,200')
    expect(formatCount(999)).toBe('999')
    expect(formatCount(1234567)).toBe('1,234,567')
  })

  it('rounds a fractional count rather than showing decimals', () => {
    expect(formatCount(1200.6)).toBe('1,201')
  })
})

describe('formatArea', () => {
  it('reports square metres below a square kilometre', () => {
    expect(formatArea(120000)).toBe('120,000 m²')
  })

  it('switches to square kilometres once the figure is large', () => {
    expect(formatArea(2_500_000)).toBe('2.50 km²')
  })

  it('keeps the boundary case in square kilometres', () => {
    expect(formatArea(1_000_000)).toBe('1.00 km²')
  })
})
