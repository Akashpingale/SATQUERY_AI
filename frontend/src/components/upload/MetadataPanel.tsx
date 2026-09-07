import { Globe, GlobeLock } from 'lucide-react'

import { IssueList } from '@/components/upload/IssueList'
import { Badge } from '@/components/ui/badge'
import type { RasterAsset } from '@/lib/api'
import {
  formatBounds,
  formatBytes,
  formatCrs,
  formatDimensions,
  formatModality,
  formatResolution,
  formatRole,
} from '@/lib/format'

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="break-words font-medium">{value}</dd>
    </>
  )
}

/**
 * Full raster metadata, including an explicit georeferencing state. When a raster has
 * no CRS/transform this says so plainly instead of showing invented coordinates.
 */
export function MetadataPanel({ asset }: { asset: RasterAsset }) {
  const geo = asset.georeference

  return (
    <section className="space-y-3" aria-label="Raster metadata">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">{formatRole(asset.role)}</Badge>
        <Badge variant="outline">{formatModality(asset.modality)}</Badge>
        {geo.is_georeferenced ? (
          <Badge variant="success">
            <Globe className="h-3.5 w-3.5" aria-hidden="true" />
            Georeferenced
          </Badge>
        ) : (
          <Badge variant="warning">
            <GlobeLock className="h-3.5 w-3.5" aria-hidden="true" />
            No georeferencing
          </Badge>
        )}
      </div>

      <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1.5 text-xs">
        <Row label="File" value={asset.original_filename} />
        <Row label="Format" value={asset.driver ?? '—'} />
        <Row label="Dimensions" value={formatDimensions(asset)} />
        <Row label="Bands" value={String(asset.band_count ?? '—')} />
        <Row label="Data type" value={asset.dtype ?? '—'} />
        <Row label="No-data" value={asset.nodata == null ? 'Not set' : String(asset.nodata)} />
        <Row label="File size" value={formatBytes(asset.size_bytes)} />
        <Row label="CRS" value={formatCrs(asset)} />
        <Row label="Resolution" value={formatResolution(asset)} />
        <Row label="Bounds" value={formatBounds(asset)} />
      </dl>

      {asset.bands.length > 0 && (
        <details className="rounded-md border bg-background/60 p-2">
          <summary className="cursor-pointer text-xs font-medium">
            Band details ({asset.bands.length})
          </summary>
          <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
            {asset.bands.map((band) => (
              <li key={band.index}>
                Band {band.index}: {band.dtype}
                {band.nodata != null && ` · no-data ${band.nodata}`}
                {band.description && ` · ${band.description}`}
              </li>
            ))}
          </ul>
        </details>
      )}

      {Object.keys(asset.tags).length > 0 && (
        <details className="rounded-md border bg-background/60 p-2">
          <summary className="cursor-pointer text-xs font-medium">
            Advanced metadata tags ({Object.keys(asset.tags).length})
          </summary>
          <dl className="mt-2 grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-xs">
            {Object.entries(asset.tags).map(([key, value]) => (
              <Row key={key} label={key} value={value} />
            ))}
          </dl>
        </details>
      )}

      {asset.warnings.length > 0 && <IssueList issues={asset.warnings} />}
    </section>
  )
}
