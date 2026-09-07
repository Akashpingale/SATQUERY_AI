import { type PointerEvent, useCallback, useEffect, useRef, useState } from 'react'

import { ImageOff, Maximize, Minus, Plus } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { type RasterAsset, previewUrl } from '@/lib/api'
import { formatCrs, formatDimensions, formatResolution } from '@/lib/format'
import { cn } from '@/lib/utils'

const MIN_ZOOM = 0.25
const MAX_ZOOM = 8
const ZOOM_STEP = 1.25

interface ImageViewerProps {
  asset: RasterAsset | null
  /** Rendered above the image; used for evidence overlays in later phases. */
  overlay?: React.ReactNode
}

/**
 * Pan/zoom viewer for a raster preview.
 *
 * The image shown is a server-rendered, downsampled PNG - the full-resolution raster
 * is never sent to the browser. Zoom/pan are keyboard operable via the toolbar. The
 * overlay chips report only metadata the backend actually returned; nothing is invented.
 */
export function ImageViewer({ asset, overlay }: ImageViewerProps) {
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const [failed, setFailed] = useState(false)
  const panStart = useRef({ x: 0, y: 0, offsetX: 0, offsetY: 0 })

  const url = asset ? previewUrl(asset) : null

  const reset = useCallback(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [])

  // Reset the view whenever a different raster is shown.
  useEffect(() => {
    reset()
    setFailed(false)
  }, [asset?.id, reset])

  const changeZoom = (factor: number) =>
    setZoom((current) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current * factor)))

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!url) return
    setIsPanning(true)
    panStart.current = {
      x: event.clientX,
      y: event.clientY,
      offsetX: offset.x,
      offsetY: offset.y,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!isPanning) return
    setOffset({
      x: panStart.current.offsetX + (event.clientX - panStart.current.x),
      y: panStart.current.offsetY + (event.clientY - panStart.current.y),
    })
  }

  const endPan = () => setIsPanning(false)

  if (!asset || !url) {
    return (
      <div className="flex min-h-[300px] flex-1 flex-col items-center justify-center gap-2 rounded-md border border-dashed bg-secondary/30 p-8 text-center">
        <ImageOff className="h-8 w-8 text-muted-foreground" aria-hidden="true" />
        <p className="text-sm font-medium">No imagery selected</p>
        <p className="max-w-sm text-xs text-muted-foreground">
          Upload a raster and choose <span className="font-medium">View</span> to inspect it here.
        </p>
      </div>
    )
  }

  const geoLabel = asset.georeference.is_georeferenced
    ? `Georeferenced · ${
        asset.georeference.crs_epsg ? `EPSG:${asset.georeference.crs_epsg}` : asset.georeference.crs
      }`
    : 'No georeferencing: coordinates and areas are unavailable for this image.'

  return (
    <div className="flex flex-1 flex-col gap-2">
      <div
        className="flex flex-wrap items-center justify-between gap-2"
        role="toolbar"
        aria-label="Image viewer controls"
      >
        <p className="truncate text-xs font-medium text-muted-foreground" title={asset.original_filename}>
          {asset.original_filename}
        </p>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="icon"
            onClick={() => changeZoom(1 / ZOOM_STEP)}
            aria-label="Zoom out"
            disabled={zoom <= MIN_ZOOM}
          >
            <Minus className="h-4 w-4" aria-hidden="true" />
          </Button>
          <span className="min-w-[3.5rem] text-center text-xs tabular-nums" aria-live="polite">
            {Math.round(zoom * 100)}%
          </span>
          <Button
            variant="outline"
            size="icon"
            onClick={() => changeZoom(ZOOM_STEP)}
            aria-label="Zoom in"
            disabled={zoom >= MAX_ZOOM}
          >
            <Plus className="h-4 w-4" aria-hidden="true" />
          </Button>
          <Button variant="outline" size="icon" onClick={reset} aria-label="Fit to screen">
            <Maximize className="h-4 w-4" aria-hidden="true" />
          </Button>
        </div>
      </div>

      {/*
       * The minimum height MUST live on this in-flow container, not on the absolutely
       * positioned wrapper inside it: an out-of-flow child contributes no intrinsic height,
       * so a `flex-1` (flex-basis 0) parent would collapse to ~0px inside a scrolling flex
       * column and the preview would render as a thin strip. A definite height here also
       * lets the image's `max-h-full` resolve correctly, which is what preserves the source
       * aspect ratio via `object-contain`.
       */}
      <div
        className={cn(
          'relative min-h-[24rem] flex-1 overflow-hidden rounded-md border bg-[#0b1220]',
          isPanning ? 'cursor-grabbing' : 'cursor-grab',
        )}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endPan}
        onPointerLeave={endPan}
      >
        {/* Metadata overlay: real values only, sourced from the backend. */}
        <div className="pointer-events-none absolute left-2 top-2 z-10 flex flex-wrap gap-1">
          <span className="data-chip">{formatDimensions(asset)}</span>
          <span className="data-chip">{formatCrs(asset)}</span>
          {asset.georeference.resolution && (
            <span className="data-chip">{formatResolution(asset)}</span>
          )}
        </div>

        <div className="absolute inset-0 flex items-center justify-center p-2">
          {failed ? (
            <p role="alert" className="max-w-xs p-6 text-center text-xs text-destructive-foreground">
              The preview image could not be loaded.
            </p>
          ) : (
            <img
              src={url}
              alt={`Preview of ${asset.original_filename}`}
              draggable={false}
              onError={() => setFailed(true)}
              style={{
                transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
              }}
              // `object-contain` + both max dimensions letterbox the raster inside the box
              // without cropping, so a 512x512 source stays square and a non-square source
              // keeps its own ratio. Zoom is a transform, so it never resizes this box.
              className="max-h-full max-w-full select-none object-contain transition-transform duration-75"
            />
          )}
          {overlay}
        </div>
      </div>

      <p className="text-xs text-muted-foreground">{geoLabel}</p>
    </div>
  )
}
