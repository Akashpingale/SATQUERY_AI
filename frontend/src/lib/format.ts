import type { RasterAsset } from '@/lib/api'

/** Human-readable file size. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`
}

/** Pixel dimensions, or a dash when unknown. */
export function formatDimensions(asset: RasterAsset): string {
  if (asset.width == null || asset.height == null) return '—'
  return `${asset.width} x ${asset.height} px`
}

/**
 * Ground resolution. Degrees are labelled as such rather than converted, because a
 * degree-to-metre conversion depends on latitude and would be misleading here.
 */
export function formatResolution(asset: RasterAsset): string {
  const res = asset.georeference.resolution
  if (!res || res.length < 2) return 'Not georeferenced'
  const [x, y] = res
  const isDegrees = asset.georeference.crs_epsg === 4326
  const unit = isDegrees ? '°' : 'm'
  const round = (v: number) => (isDegrees ? v.toPrecision(4) : v.toFixed(2))
  return x === y ? `${round(x)}${unit}` : `${round(x)}${unit} x ${round(y)}${unit}`
}

/** Bounding box as "W, S — E, N", or a clear absence message. */
export function formatBounds(asset: RasterAsset): string {
  const b = asset.georeference.bounds
  if (!b || b.length < 4) return 'Not georeferenced'
  const [left, bottom, right, top] = b
  const f = (v: number) => v.toFixed(4)
  return `${f(left)}, ${f(bottom)} — ${f(right)}, ${f(top)}`
}

/**
 * Integer with thousands separators.
 *
 * Grouping is applied explicitly rather than via `toLocaleString`, which is locale-dependent:
 * the same measurement renders as `120,000` or `1,20,000` depending on the machine, which is
 * wrong for a reproducible measurement readout and makes tests environment-sensitive.
 */
export function formatCount(value: number): string {
  return Math.round(value)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ',')
}

/**
 * Ground area in the unit that suits its magnitude.
 *
 * Only ever called with a real square-metre figure: when the CRS is geographic the backend
 * sends `null` instead, because an area in degrees squared is not an area.
 */
export function formatArea(squareMetres: number): string {
  if (squareMetres >= 1_000_000) return `${(squareMetres / 1_000_000).toFixed(2)} km²`
  return `${formatCount(squareMetres)} m²`
}

export function formatCrs(asset: RasterAsset): string {
  const { crs_epsg: epsg, crs } = asset.georeference
  if (epsg) return `EPSG:${epsg}`
  if (crs) return crs
  return 'None'
}

const ROLE_LABELS: Record<string, string> = {
  single: 'Single image',
  optical: 'Optical / Multispectral',
  sar: 'SAR',
  t1: 'Temporal T1',
  t2: 'Temporal T2',
}

export function formatRole(role: string): string {
  return ROLE_LABELS[role] ?? role
}

const MODALITY_LABELS: Record<string, string> = {
  optical: 'Optical',
  sar: 'SAR',
  unknown: 'Unknown',
}

export function formatModality(modality: string): string {
  return MODALITY_LABELS[modality] ?? modality
}
