import { useState } from 'react'

import {
  Activity,
  Crosshair,
  Cpu,
  GitCompareArrows,
  Image as ImageIcon,
  Layers,
  Leaf,
  ListOrdered,
  MapPin,
  Radar,
  Table,
  Waves,
} from 'lucide-react'

import { API_BASE_URL, type EvidenceOut } from '@/lib/api'
import { formatArea, formatCount } from '@/lib/format'

const KIND_META: Record<string, { label: string; Icon: typeof ImageIcon }> = {
  metadata: { label: 'Metadata', Icon: Table },
  preview: { label: 'Image', Icon: ImageIcon },
  overlay: { label: 'Detected regions', Icon: Crosshair },
  box: { label: 'Bounding box', Icon: Crosshair },
  region: { label: 'Estimated region', Icon: MapPin },
  mask: { label: 'Mask', Icon: Crosshair },
  scores: { label: 'Model scores', Icon: ListOrdered },
  model: { label: 'Model', Icon: Cpu },
  comparison: { label: 'Pair comparison', Icon: Table },
  // Phase 6 temporal evidence.
  change_map: { label: 'Change map', Icon: Layers },
  temporal_compare: { label: 'T1 / T2 / change', Icon: GitCompareArrows },
  change_region: { label: 'Changed region', Icon: Activity },
  indices: { label: 'Spectral indices', Icon: Leaf },
  // Phase 7 optical+SAR cross-modal evidence.
  fusion_composite: { label: 'Optical+SAR composite', Icon: Layers },
  fusion_complementary: { label: 'Complementary map', Icon: Radar },
  aligned_optical: { label: 'Aligned optical', Icon: ImageIcon },
  aligned_sar: { label: 'Aligned SAR', Icon: Radar },
  modality_comparison: { label: 'Optical vs SAR', Icon: Table },
  cross_modal_water: { label: 'Water agreement', Icon: Waves },
  complementary_region: { label: 'Complementary region', Icon: MapPin },
}

/** Evidence kinds whose `reference` is a rendered image the panel should display. */
const IMAGE_KINDS = new Set([
  'overlay',
  'preview',
  'change_map',
  'temporal_compare',
  'fusion_composite',
  'fusion_complementary',
  'aligned_optical',
  'aligned_sar',
])

function absoluteReference(reference: string): string {
  if (reference.startsWith('http')) return reference
  const origin = API_BASE_URL.replace(/\/api\/v1\/?$/, '')
  return `${origin}${reference}`
}

/**
 * A rendered evidence image with an explicit failure fallback.
 *
 * Evidence artifacts are files on the server; a token can expire or a file can be missing. When
 * the image fails to load we show a labelled placeholder instead of a broken-image icon, so the
 * evidence card stays meaningful.
 */
function EvidenceImage({ reference, alt }: { reference: string; alt: string }) {
  const [failed, setFailed] = useState(false)
  if (failed) {
    return (
      <p className="mt-2 rounded border border-dashed bg-secondary/20 p-3 text-center text-xs text-muted-foreground">
        This visual evidence could not be loaded. It may have expired or been removed from the
        server.
      </p>
    )
  }
  return (
    <img
      src={absoluteReference(reference)}
      alt={alt}
      onError={() => setFailed(true)}
      className="mt-2 max-h-56 w-full rounded border bg-[#0b1220] object-contain"
    />
  )
}

interface ConceptScore {
  key: string
  label: string
  probability: number
}

/**
 * Ranked concept probabilities with a proportional bar.
 *
 * The percentage is always shown as text next to the bar, so the information does not
 * depend on reading bar length or colour.
 */
function ScoreBars({ detail }: { detail: Record<string, unknown> }) {
  const concepts = (detail.concepts as ConceptScore[] | undefined) ?? []
  if (concepts.length === 0) return <DetailList detail={detail} />

  const note = typeof detail.note === 'string' ? detail.note : null
  const method = typeof detail.method === 'string' ? detail.method : null

  return (
    <div className="mt-2 space-y-1.5">
      <ul className="space-y-1">
        {concepts.map((concept) => {
          const percent = Math.round(concept.probability * 100)
          return (
            <li key={concept.key} className="flex items-center gap-2 text-xs">
              <span className="w-28 shrink-0 truncate" title={concept.label}>
                {concept.label}
              </span>
              <span
                className="h-2 flex-1 overflow-hidden rounded bg-secondary"
                role="img"
                aria-label={`${concept.label}: ${percent} percent`}
              >
                <span
                  className="block h-full rounded bg-primary"
                  style={{ width: `${Math.max(2, percent)}%` }}
                />
              </span>
              <span className="w-10 shrink-0 text-right tabular-nums">{percent}%</span>
            </li>
          )
        })}
      </ul>
      {method && <p className="text-xs text-muted-foreground">Method: {method}</p>}
      {note && <p className="text-xs text-muted-foreground">{note}</p>}
    </div>
  )
}

/** Geographic and pixel geometry for a detected box or estimated region. */
function GeometryDetail({ detail }: { detail: Record<string, unknown> }) {
  const geo = detail.geo as Record<string, unknown> | undefined
  const lonlat = geo?.center_lonlat as [number, number] | undefined
  const pixelBox = detail.analyzed_pixel_box as number[] | undefined
  const score = typeof detail.score === 'number' ? detail.score : null
  const contrast = typeof detail.tile_contrast === 'number' ? detail.tile_contrast : null

  return (
    <dl className="mt-1.5 grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs">
      {score !== null && (
        <>
          <dt className="text-muted-foreground">Detection score</dt>
          <dd className="tabular-nums">{Math.round(score * 100)}%</dd>
        </>
      )}
      {contrast !== null && (
        <>
          <dt className="text-muted-foreground">Tile contrast</dt>
          <dd className="tabular-nums">{Math.round(contrast * 100)}%</dd>
        </>
      )}
      {typeof detail.grid === 'string' && (
        <>
          <dt className="text-muted-foreground">Tile grid</dt>
          <dd>
            {detail.grid}
            {typeof detail.tiles_merged === 'number' && ` (${detail.tiles_merged} merged)`}
          </dd>
        </>
      )}
      {pixelBox && pixelBox.length === 4 && (
        <>
          <dt className="text-muted-foreground">Pixel box</dt>
          <dd className="tabular-nums">
            {pixelBox[0]}, {pixelBox[1]} → {pixelBox[2]}, {pixelBox[3]}
          </dd>
        </>
      )}
      {lonlat ? (
        <>
          <dt className="text-muted-foreground">Centre (lat, lon)</dt>
          <dd className="tabular-nums">
            {lonlat[1].toFixed(5)}, {lonlat[0].toFixed(5)}
          </dd>
        </>
      ) : (
        <>
          <dt className="text-muted-foreground">Coordinates</dt>
          <dd>Unavailable (image not georeferenced)</dd>
        </>
      )}
    </dl>
  )
}

/**
 * The measured numbers behind a change map.
 *
 * Only the figures that support the answer are surfaced, each labelled with its unit, so a
 * reader can check the claim rather than take it on trust. Area is omitted when the CRS does
 * not permit a metric figure, which is why `changed_area_m2` may be null.
 */
function ChangeStats({ detail }: { detail: Record<string, unknown> }) {
  const number = (key: string): number | null =>
    typeof detail[key] === 'number' ? (detail[key] as number) : null

  const percent = number('changed_percent')
  const areaM2 = number('changed_area_m2')
  const regions = number('region_count')
  const coherence = number('coherence')
  const threshold = number('threshold')
  const correlation = number('registration_peak_correlation')
  const shift = detail.registration_peak_offset as number[] | undefined
  const method = typeof detail.method === 'string' ? detail.method : null
  const version = typeof detail.algorithm_version === 'string' ? detail.algorithm_version : null
  const thresholdMethod =
    typeof detail.threshold_method === 'string' ? detail.threshold_method : null
  const operations = detail.co_registration as string[] | undefined

  const rows: [string, string][] = []
  if (percent !== null) {
    rows.push([
      'Area changed',
      areaM2 !== null
        ? `${percent.toFixed(2)}% (${formatArea(areaM2)})`
        : `${percent.toFixed(2)}% of the shared area`,
    ])
  }
  if (regions !== null) rows.push(['Coherent regions', String(regions)])
  if (coherence !== null) rows.push(['Spatial coherence', `${Math.round(coherence * 100)}%`])
  if (threshold !== null) {
    rows.push([
      'Change threshold',
      thresholdMethod ? `${threshold.toFixed(4)} (${thresholdMethod})` : threshold.toFixed(4),
    ])
  }
  if (correlation !== null) {
    const offset = shift && shift.length === 2 ? ` at offset ${shift[1]}, ${shift[0]} px` : ''
    rows.push(['Registration correlation', `${correlation.toFixed(3)}${offset}`])
  }
  if (method) rows.push(['Method', `${method.replace(/_/g, ' ')}${version ? ` · ${version}` : ''}`])

  return (
    <div className="mt-1.5 space-y-1.5">
      <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs">
        {rows.map(([term, value]) => (
          <div key={term} className="col-span-2 grid grid-cols-[auto,1fr] gap-x-3">
            <dt className="text-muted-foreground">{term}</dt>
            <dd className="tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
      {operations && operations.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Co-registration: {operations.join('; ')}.
        </p>
      )}
    </div>
  )
}

/** Geometry and measured strength of one spatially coherent changed area. */
function ChangeRegionDetail({ detail }: { detail: Record<string, unknown> }) {
  const pixels = typeof detail.pixel_count === 'number' ? detail.pixel_count : null
  const areaM2 = typeof detail.area_m2 === 'number' ? detail.area_m2 : null
  const position = typeof detail.position === 'string' ? detail.position : null
  const lonlat = detail.centroid_lonlat as [number, number] | null | undefined
  const magnitude = typeof detail.mean_magnitude === 'number' ? detail.mean_magnitude : null
  const brightness =
    typeof detail.brightness_delta === 'number' ? detail.brightness_delta : null
  const box = detail.box as number[] | undefined
  const fraction =
    typeof detail.fraction_of_scene === 'number' ? detail.fraction_of_scene : null

  return (
    <dl className="mt-1.5 grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs">
      {position && (
        <div className="col-span-2 grid grid-cols-[auto,1fr] gap-x-3">
          <dt className="text-muted-foreground">Position</dt>
          <dd>{position}</dd>
        </div>
      )}
      {pixels !== null && (
        <div className="col-span-2 grid grid-cols-[auto,1fr] gap-x-3">
          <dt className="text-muted-foreground">Size</dt>
          <dd className="tabular-nums">
            {formatCount(pixels)} px
            {areaM2 !== null && ` · ${formatArea(areaM2)}`}
            {fraction !== null && ` · ${(fraction * 100).toFixed(2)}% of the scene`}
          </dd>
        </div>
      )}
      {box && box.length === 4 && (
        <div className="col-span-2 grid grid-cols-[auto,1fr] gap-x-3">
          <dt className="text-muted-foreground">Pixel box</dt>
          <dd className="tabular-nums">
            {box[0]}, {box[1]} → {box[2]}, {box[3]}
          </dd>
        </div>
      )}
      <div className="col-span-2 grid grid-cols-[auto,1fr] gap-x-3">
        <dt className="text-muted-foreground">Centre (lat, lon)</dt>
        <dd className="tabular-nums">
          {lonlat ? `${lonlat[1].toFixed(5)}, ${lonlat[0].toFixed(5)}` : 'Unavailable (not georeferenced)'}
        </dd>
      </div>
      {magnitude !== null && (
        <div className="col-span-2 grid grid-cols-[auto,1fr] gap-x-3">
          <dt className="text-muted-foreground">Mean change magnitude</dt>
          <dd className="tabular-nums">{magnitude.toFixed(3)}</dd>
        </div>
      )}
      {brightness !== null && (
        <div className="col-span-2 grid grid-cols-[auto,1fr] gap-x-3">
          <dt className="text-muted-foreground">Brightness</dt>
          <dd className="tabular-nums">
            {brightness >= 0 ? 'brighter' : 'darker'} at T2 ({brightness >= 0 ? '+' : ''}
            {brightness.toFixed(1)})
          </dd>
        </div>
      )}
    </dl>
  )
}

interface IndexDetail {
  name: string
  available: boolean
  reason?: string
  mean_before?: number | null
  mean_after?: number | null
  delta?: number | null
  delta_in_changed?: number | null
}

/**
 * Spectral index change, including the ones that could not be computed.
 *
 * Showing unavailable indices with their reason is deliberate: it tells the user that no
 * vegetation or water trend is being claimed, and why.
 */
function IndexList({ detail }: { detail: Record<string, unknown> }) {
  const indices = (detail.indices as IndexDetail[] | undefined) ?? []
  const note = typeof detail.note === 'string' ? detail.note : null
  if (indices.length === 0) return <DetailList detail={detail} />

  return (
    <div className="mt-2 space-y-1.5">
      <ul className="space-y-1.5">
        {indices.map((index) => (
          <li key={index.name} className="text-xs">
            <p className="flex items-center gap-1.5 font-medium">
              {index.name}
              <span
                className={
                  index.available
                    ? 'rounded bg-secondary px-1 py-0.5 text-[10px] font-normal'
                    : 'rounded border border-dashed px-1 py-0.5 text-[10px] font-normal text-muted-foreground'
                }
              >
                {index.available ? 'measured' : 'not available'}
              </span>
            </p>
            {index.available ? (
              <p className="tabular-nums text-muted-foreground">
                {typeof index.mean_before === 'number' && typeof index.mean_after === 'number'
                  ? `${index.mean_before.toFixed(3)} → ${index.mean_after.toFixed(3)}`
                  : null}
                {typeof index.delta === 'number' &&
                  ` (${index.delta >= 0 ? '+' : ''}${index.delta.toFixed(3)} overall`}
                {typeof index.delta_in_changed === 'number' &&
                  `, ${index.delta_in_changed >= 0 ? '+' : ''}${index.delta_in_changed.toFixed(
                    3,
                  )} in changed areas`}
                {typeof index.delta === 'number' && ')'}
              </p>
            ) : (
              <p className="text-muted-foreground">{index.reason}</p>
            )}
          </li>
        ))}
      </ul>
      {note && <p className="text-xs text-muted-foreground">{note}</p>}
    </div>
  )
}

interface ModalityFeature {
  modality: string
  mean_level?: number
  spread?: number
  texture?: number
  structured_fraction?: number
  [key: string]: unknown
}

/**
 * Optical vs SAR measured features side by side.
 *
 * The point of the panel is that the two modalities carry *different* keys (NDVI for optical,
 * dark-fraction for SAR), which is exactly why fusion is informative.
 */
function ModalityComparison({ detail }: { detail: Record<string, unknown> }) {
  const optical = detail.optical as ModalityFeature | undefined
  const sar = detail.sar as ModalityFeature | undefined
  const correlation = typeof detail.correlation === 'number' ? detail.correlation : null
  const complementarity =
    typeof detail.complementarity === 'number' ? detail.complementarity : null
  const sarRep = typeof detail.sar_representation === 'string' ? detail.sar_representation : null

  const row = (label: string, o: unknown, s: unknown) => (
    <tr key={label} className="border-t">
      <td className="py-0.5 pr-2 text-muted-foreground">{label}</td>
      <td className="py-0.5 pr-2 text-right tabular-nums">{fmt(o)}</td>
      <td className="py-0.5 text-right tabular-nums">{fmt(s)}</td>
    </tr>
  )
  const fmt = (v: unknown) => (typeof v === 'number' ? v.toFixed(3) : v == null ? '—' : String(v))

  return (
    <div className="mt-2 space-y-1.5 text-xs">
      {(correlation !== null || complementarity !== null) && (
        <p>
          {correlation !== null && `Correlation ${correlation.toFixed(2)}`}
          {complementarity !== null && ` · complementarity ${complementarity.toFixed(2)}`}
        </p>
      )}
      <table className="w-full">
        <thead>
          <tr>
            <th className="text-left font-medium">Feature</th>
            <th className="text-right font-medium">Optical</th>
            <th className="text-right font-medium">SAR</th>
          </tr>
        </thead>
        <tbody>
          {row('Mean level', optical?.mean_level, sar?.mean_level)}
          {row('Dynamic range', optical?.spread, sar?.spread)}
          {row('Texture', optical?.texture, sar?.texture)}
          {row('Structured fraction', optical?.structured_fraction, sar?.structured_fraction)}
          {row('NDVI (optical)', optical?.ndvi_mean, '—')}
          {row('Dark fraction (SAR)', '—', sar?.dark_fraction)}
        </tbody>
      </table>
      {sarRep && <p className="text-muted-foreground">SAR: {sarRep}</p>}
      {typeof detail.note === 'string' && (
        <p className="text-muted-foreground">{detail.note}</p>
      )}
    </div>
  )
}

/** Cross-modal water agreement between optical NDWI and SAR dark surfaces. */
function WaterAgreementDetail({ detail }: { detail: Record<string, unknown> }) {
  if (detail.available !== true) {
    return (
      <p className="mt-1.5 text-xs text-muted-foreground">
        Not available: {typeof detail.reason === 'string' ? detail.reason : 'not computed'}.
      </p>
    )
  }
  const pct = (key: string) =>
    typeof detail[key] === 'number' ? `${((detail[key] as number) * 100).toFixed(1)}%` : '—'
  const iou = typeof detail.agreement_iou === 'number' ? detail.agreement_iou.toFixed(2) : '—'

  return (
    <dl className="mt-1.5 grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs">
      <dt className="text-muted-foreground">Agreement (IoU)</dt>
      <dd className="tabular-nums">{iou}</dd>
      <dt className="text-muted-foreground">Agree on water</dt>
      <dd className="tabular-nums">{pct('agree_fraction')} of the scene</dd>
      <dt className="text-muted-foreground">Optical says water</dt>
      <dd className="tabular-nums">{pct('optical_water_fraction')}</dd>
      <dt className="text-muted-foreground">SAR smooth/dark</dt>
      <dd className="tabular-nums">{pct('sar_dark_fraction')}</dd>
      <dt className="text-muted-foreground">SAR-dark, not optical water</dt>
      <dd className="tabular-nums">{pct('sar_only_fraction')}</dd>
    </dl>
  )
}

/** A coherent area where one modality carries information the other does not. */
function ComplementaryRegionDetail({ detail }: { detail: Record<string, unknown> }) {
  const kind = typeof detail.kind === 'string' ? detail.kind.replace(/_/g, ' ') : null
  const pixels = typeof detail.pixel_count === 'number' ? detail.pixel_count : null
  const areaM2 = typeof detail.area_m2 === 'number' ? detail.area_m2 : null
  const position = typeof detail.position === 'string' ? detail.position : null
  const lonlat = detail.centroid_lonlat as [number, number] | null | undefined

  return (
    <dl className="mt-1.5 grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs">
      {kind && (
        <>
          <dt className="text-muted-foreground">Type</dt>
          <dd>{kind}</dd>
        </>
      )}
      {position && (
        <>
          <dt className="text-muted-foreground">Position</dt>
          <dd>{position}</dd>
        </>
      )}
      {pixels !== null && (
        <>
          <dt className="text-muted-foreground">Size</dt>
          <dd className="tabular-nums">
            {formatCount(pixels)} px{areaM2 !== null && ` · ${formatArea(areaM2)}`}
          </dd>
        </>
      )}
      <dt className="text-muted-foreground">Centre (lat, lon)</dt>
      <dd className="tabular-nums">
        {lonlat ? `${lonlat[1].toFixed(5)}, ${lonlat[0].toFixed(5)}` : 'Unavailable (not georeferenced)'}
      </dd>
    </dl>
  )
}

/** Before/after land-cover reading of a changed region. */
function SemanticReading({ detail }: { detail: Record<string, unknown> }) {
  const available = detail.available === true
  const note = typeof detail.note === 'string' ? detail.note : null
  if (!available) {
    return (
      <p className="mt-1.5 text-xs text-muted-foreground">
        Land-cover reading unavailable:{' '}
        {typeof detail.reason === 'string' ? detail.reason : 'not computed'}. The cause of the
        change is reported as undetermined.
      </p>
    )
  }

  const before = String(detail.before_label ?? '')
  const after = String(detail.after_label ?? '')
  const beforeP = typeof detail.before_probability === 'number' ? detail.before_probability : null
  const afterP = typeof detail.after_probability === 'number' ? detail.after_probability : null
  const supported = detail.transition_supported === true

  return (
    <div className="mt-1.5 space-y-1 text-xs">
      <p className="tabular-nums">
        <span className="text-muted-foreground">Before: </span>
        {before}
        {beforeP !== null && ` (${Math.round(beforeP * 100)}%)`}
        <span className="text-muted-foreground"> → After: </span>
        {after}
        {afterP !== null && ` (${Math.round(afterP * 100)}%)`}
      </p>
      <p className="text-muted-foreground">
        {supported
          ? 'The scores support describing this as a land-cover transition.'
          : 'The scores do not support naming a specific land-cover cause.'}
      </p>
      {note && <p className="text-muted-foreground">{note}</p>}
    </div>
  )
}

function DetailList({ detail }: { detail: Record<string, unknown> }) {
  const entries = Object.entries(detail).filter(
    ([, value]) => value !== null && value !== undefined && value !== '',
  )
  if (entries.length === 0) return null

  return (
    <dl className="mt-1.5 grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs">
      {entries.map(([key, value]) => (
        <div key={key} className="col-span-2 grid grid-cols-[auto,1fr] gap-x-3">
          <dt className="text-muted-foreground">{key.replace(/_/g, ' ')}</dt>
          <dd className="break-words">
            {typeof value === 'object' ? JSON.stringify(value) : String(value)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

/**
 * Evidence supporting a result: rendered overlays, detection geometry, model scores,
 * measured metadata, and provenance.
 *
 * Every item is linked to the analysis that produced it, so a claim can always be checked
 * against something inspectable.
 */
export function EvidencePanel({ evidence }: { evidence: EvidenceOut[] }) {
  if (evidence.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No inspectable evidence was produced for this result.
      </p>
    )
  }

  // Rendered visual evidence first: the change map and the T1/T2 strip are the primary
  // evidence for a temporal result, as the overlay is for grounding. Plain input previews
  // sort last because they show what went in, not what was found.
  const ordered = [...evidence].sort((a, b) => {
    const lead = new Set([
      'change_map',
      'temporal_compare',
      'overlay',
      'fusion_composite',
      'fusion_complementary',
    ])
    const rank = (kind: string) => {
      if (lead.has(kind)) return 0
      if (kind === 'preview') return 2
      return 1
    }
    return rank(a.kind) - rank(b.kind)
  })

  return (
    <ul className="space-y-2.5" aria-label="Evidence">
      {ordered.map((item, index) => {
        const meta = KIND_META[item.kind] ?? { label: item.kind, Icon: Table }
        const { label, Icon } = meta
        const isImage = IMAGE_KINDS.has(item.kind)
        const isSemantic = item.kind === 'scores' && 'transition_supported' in item.detail

        return (
          <li key={`${item.kind}-${index}`} className="rounded-md border bg-card/60 p-2.5">
            <p className="flex items-center gap-1.5 text-xs font-medium">
              <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
              {label}
              <span className="truncate font-normal text-muted-foreground">· {item.label}</span>
            </p>

            {isImage && item.reference && (
              <EvidenceImage reference={item.reference} alt={item.label} />
            )}

            {item.kind === 'scores' &&
              (isSemantic ? (
                <SemanticReading detail={item.detail} />
              ) : (
                <ScoreBars detail={item.detail} />
              ))}
            {item.kind === 'change_map' && <ChangeStats detail={item.detail} />}
            {item.kind === 'indices' && <IndexList detail={item.detail} />}
            {item.kind === 'change_region' && <ChangeRegionDetail detail={item.detail} />}
            {item.kind === 'modality_comparison' && <ModalityComparison detail={item.detail} />}
            {item.kind === 'cross_modal_water' && <WaterAgreementDetail detail={item.detail} />}
            {item.kind === 'complementary_region' && (
              <ComplementaryRegionDetail detail={item.detail} />
            )}
            {(item.kind === 'fusion_composite' || item.kind === 'fusion_complementary') &&
              typeof item.detail.note === 'string' && (
                <p className="mt-1.5 text-xs text-muted-foreground">{item.detail.note}</p>
              )}
            {(item.kind === 'box' || item.kind === 'region') && (
              <GeometryDetail detail={item.detail} />
            )}
            {item.kind === 'temporal_compare' && typeof item.detail.note === 'string' && (
              <p className="mt-1.5 text-xs text-muted-foreground">{item.detail.note}</p>
            )}
            {item.kind === 'overlay' && typeof item.detail.method === 'string' && (
              <p className="mt-1.5 text-xs text-muted-foreground">
                Method: {String(item.detail.method).replace(/_/g, ' ')}
              </p>
            )}
            {(item.kind === 'metadata' || item.kind === 'model' || item.kind === 'comparison') && (
              <DetailList detail={item.detail} />
            )}
          </li>
        )
      })}
    </ul>
  )
}
