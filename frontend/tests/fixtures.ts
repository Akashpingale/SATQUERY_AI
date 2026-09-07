import type { RasterAsset } from '@/lib/api'

/** A georeferenced optical asset shaped exactly like the backend response. */
export const georeferencedAsset: RasterAsset = {
  id: 'asset-geo-1',
  created_at: '2026-01-01T00:00:00Z',
  original_filename: 'scene_optical.tif',
  size_bytes: 786_432,
  role: 'optical',
  modality: 'optical',
  detected_modality: 'optical',
  width: 512,
  height: 512,
  band_count: 3,
  dtype: 'uint8',
  driver: 'GTiff',
  georeference: {
    is_georeferenced: true,
    crs: 'EPSG:4326',
    crs_epsg: 4326,
    transform: [0.0005, 0, 77.5, 0, -0.0005, 12.99],
    bounds: [77.5, 12.734, 77.756, 12.99],
    resolution: [0.0005, 0.0005],
  },
  bands: [
    { index: 1, dtype: 'uint8', nodata: null, description: null },
    { index: 2, dtype: 'uint8', nodata: null, description: null },
    { index: 3, dtype: 'uint8', nodata: null, description: null },
  ],
  nodata: null,
  tags: { SENSOR: 'Demo optical' },
  preview_url: '/api/v1/uploads/asset-geo-1/preview',
  warnings: [],
}

/** A plain PNG benchmark image: valid, but with no georeferencing. */
export const plainAsset: RasterAsset = {
  ...georeferencedAsset,
  id: 'asset-plain-1',
  original_filename: 'benchmark.png',
  driver: 'PNG',
  georeference: {
    is_georeferenced: false,
    crs: null,
    crs_epsg: null,
    transform: null,
    bounds: null,
    resolution: null,
  },
  tags: {},
  preview_url: null,
  warnings: [
    {
      code: 'not_georeferenced',
      message:
        'The image has no CRS/transform. Analysis will work, but coordinates, areas, and pair alignment cannot be computed.',
      severity: 'warning',
    },
  ],
}
