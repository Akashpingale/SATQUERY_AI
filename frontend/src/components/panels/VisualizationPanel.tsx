import { useEffect, useMemo, useState } from 'react'

import { Layers, Map as MapIcon, Radar, Satellite, SunMedium } from 'lucide-react'

import { Panel } from '@/components/ui/panel'
import { MetadataPanel } from '@/components/upload/MetadataPanel'
import { ImageViewer } from '@/components/viewer/ImageViewer'
import { API_BASE_URL, type EvidenceOut, type RasterAsset, previewUrl } from '@/lib/api'
import { cn } from '@/lib/utils'

type ViewMode = 'single' | 'compare' | 'change' | 'fusion'

interface VisualizationPanelProps {
  asset: RasterAsset | null
  /** Temporal slots, so the panel can offer a T1/T2 comparison when both exist. */
  t1?: RasterAsset | null
  t2?: RasterAsset | null
  /** Evidence from the latest analysis; change artifacts are picked out of it. */
  evidence?: EvidenceOut[]
}

function absoluteReference(reference: string): string {
  if (reference.startsWith('http')) return reference
  const origin = API_BASE_URL.replace(/\/api\/v1\/?$/, '')
  return `${origin}${reference}`
}

/** Polished, intentional empty state shown before any imagery is loaded. */
function EmptyWorkspace() {
  return (
    <div className="flex min-h-[360px] flex-1 flex-col items-center justify-center rounded-md border border-dashed bg-secondary/30 p-8 text-center">
      <span className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/15 to-accent/15 text-primary">
        <Satellite className="h-8 w-8" aria-hidden="true" />
      </span>
      <p className="text-base font-semibold">Your analysis workspace</p>
      <p className="mt-1 max-w-sm text-sm text-muted-foreground">
        Upload a remote-sensing image to begin. Previews, overlays, and change maps render here.
      </p>
      <div className="mt-4 flex flex-wrap items-center justify-center gap-1.5">
        {['Optical', 'SAR', 'Temporal', 'Multimodal'].map((m) => (
          <span key={m} className="data-chip">
            {m}
          </span>
        ))}
      </div>
    </div>
  )
}

/** Side-by-side previews of the two dates on equal footing. */
function TemporalCompare({ t1, t2 }: { t1: RasterAsset; t2: RasterAsset }) {
  const panels: [string, RasterAsset][] = [
    ['T1 · earlier', t1],
    ['T2 · later', t2],
  ]
  return (
    <div className="grid flex-1 grid-cols-1 gap-3 sm:grid-cols-2">
      {panels.map(([label, asset]) => {
        const url = previewUrl(asset)
        return (
          <figure key={asset.id} className="flex flex-col gap-1.5">
            <figcaption className="flex items-baseline justify-between gap-2 text-xs">
              <span className="font-medium">{label}</span>
              <span className="truncate text-muted-foreground" title={asset.original_filename}>
                {asset.original_filename}
              </span>
            </figcaption>
            {url ? (
              <img
                src={url}
                alt={`${label}: preview of ${asset.original_filename}`}
                className="min-h-[220px] w-full rounded-md border bg-[#0b1220] object-contain"
              />
            ) : (
              <p className="flex min-h-[220px] items-center justify-center rounded-md border border-dashed bg-secondary/30 p-4 text-center text-xs text-muted-foreground">
                No preview was rendered for this raster.
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              {asset.georeference.is_georeferenced
                ? `Georeferenced · ${
                    asset.georeference.crs_epsg
                      ? `EPSG:${asset.georeference.crs_epsg}`
                      : asset.georeference.crs
                  }`
                : 'No georeferencing'}
            </p>
          </figure>
        )
      })}
    </div>
  )
}

/**
 * Rendered change/fusion artifacts from the last analysis.
 *
 * These are server-rendered from the measured mask, so what is shown here is the same
 * data the answer is derived from.
 */
function ChangeView({ items }: { items: EvidenceOut[] }) {
  return (
    <div className="flex flex-1 flex-col gap-4">
      {items.map((item, index) => (
        <figure key={`${item.kind}-${index}`} className="flex flex-col gap-1.5">
          <figcaption className="text-xs font-medium">{item.label}</figcaption>
          {item.reference && (
            <img
              src={absoluteReference(item.reference)}
              alt={item.label}
              className="w-full rounded-md border bg-[#0b1220] object-contain"
            />
          )}
          {item.kind === 'change_map' && (
            <p className="text-xs text-muted-foreground">
              Tinted areas exceeded the measured change threshold and survived spatial-coherence
              filtering. Outlined boxes are the reported regions, numbered as in the evidence
              list.
            </p>
          )}
          {item.kind === 'fusion_composite' && (
            <p className="text-xs text-muted-foreground">
              False-colour composite: red is SAR backscatter, green/blue is optical brightness.
              Reddish areas are SAR-strong (rough/structured); cyan areas are optically bright
              with low SAR return.
            </p>
          )}
          {item.kind === 'fusion_complementary' && (
            <p className="text-xs text-muted-foreground">
              Highlights areas where one modality carries information the other does not. Every
              tinted pixel satisfies a measured condition; boxes are the reported regions.
            </p>
          )}
        </figure>
      ))}
    </div>
  )
}

const MODE_ICON: Record<ViewMode, typeof Layers> = {
  single: SunMedium,
  compare: Layers,
  change: MapIcon,
  fusion: Radar,
}

/**
 * The main viewer: a single raster, the two temporal dates side by side, or the rendered
 * change/fusion evidence.
 *
 * Modes appear only when their data exists, so the control never offers an empty view. The
 * mode buttons are a real tablist so they are keyboard operable and announced correctly.
 */
export function VisualizationPanel({ asset, t1, t2, evidence = [] }: VisualizationPanelProps) {
  const changeItems = useMemo(
    () =>
      evidence.filter(
        (item) =>
          (item.kind === 'change_map' || item.kind === 'temporal_compare') && item.reference,
      ),
    [evidence],
  )
  const fusionItems = useMemo(
    () =>
      evidence.filter(
        (item) =>
          (item.kind === 'fusion_composite' ||
            item.kind === 'fusion_complementary' ||
            item.kind === 'aligned_optical' ||
            item.kind === 'aligned_sar') &&
          item.reference,
      ),
    [evidence],
  )

  const hasPair = Boolean(t1 && t2)
  const hasChange = changeItems.length > 0
  const hasFusion = fusionItems.length > 0
  const [mode, setMode] = useState<ViewMode>('single')

  // Surface a new analysis result automatically, and never leave the panel on a mode whose
  // data has since disappeared.
  useEffect(() => {
    if (hasFusion) {
      setMode('fusion')
      return
    }
    if (hasChange) {
      setMode('change')
      return
    }
    setMode((current) =>
      current === 'change' || current === 'fusion' ? (hasPair ? 'compare' : 'single') : current,
    )
  }, [hasChange, hasFusion, hasPair])

  useEffect(() => {
    if (!hasPair) setMode((current) => (current === 'compare' ? 'single' : current))
  }, [hasPair])

  const modes: { id: ViewMode; label: string; enabled: boolean }[] = [
    { id: 'single', label: 'Single image', enabled: true },
    { id: 'compare', label: 'Compare T1 / T2', enabled: hasPair },
    { id: 'change', label: 'Change evidence', enabled: hasChange },
    { id: 'fusion', label: 'Optical + SAR', enabled: hasFusion },
  ]
  const available = modes.filter((entry) => entry.enabled)

  const isEmpty = !asset && !hasPair && !hasChange && !hasFusion

  return (
    <Panel
      title="Visualization"
      description="Server-rendered previews; full-resolution data stays on the server"
      icon={<MapIcon className="h-4 w-4" />}
      fill
      // A min-height floor keeps the viewer from being squeezed to nothing when the panel
      // body scrolls (metadata below the image); the body still scrolls independently.
      bodyClassName="flex min-h-[32rem] flex-col gap-4"
    >
      {available.length > 1 && (
        <div
          role="tablist"
          aria-label="Visualization mode"
          className="flex flex-wrap gap-1 rounded-md border bg-secondary/40 p-1"
        >
          {available.map((entry) => {
            const selected = mode === entry.id
            const Icon = MODE_ICON[entry.id]
            return (
              <button
                key={entry.id}
                type="button"
                role="tab"
                aria-selected={selected}
                onClick={() => setMode(entry.id)}
                className={cn(
                  'inline-flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium transition-colors',
                  selected
                    ? 'bg-primary text-primary-foreground shadow-panel'
                    : 'text-muted-foreground hover:bg-card hover:text-foreground',
                )}
              >
                <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                {entry.label}
              </button>
            )
          })}
        </div>
      )}

      {isEmpty ? (
        <EmptyWorkspace />
      ) : mode === 'compare' && t1 && t2 ? (
        <TemporalCompare t1={t1} t2={t2} />
      ) : mode === 'change' && hasChange ? (
        <ChangeView items={changeItems} />
      ) : mode === 'fusion' && hasFusion ? (
        <ChangeView items={fusionItems} />
      ) : (
        <>
          <ImageViewer asset={asset} />
          {asset && (
            <div className="border-t pt-3">
              <MetadataPanel asset={asset} />
            </div>
          )}
        </>
      )}
    </Panel>
  )
}
