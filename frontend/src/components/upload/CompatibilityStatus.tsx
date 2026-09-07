import { CheckCircle2, XCircle } from 'lucide-react'

import { IssueList } from '@/components/upload/IssueList'
import { Badge } from '@/components/ui/badge'
import type { CompatibilityResponse } from '@/lib/api'

const PURPOSE_LABELS: Record<string, string> = {
  temporal: 'Temporal pair (T1 / T2)',
  optical_sar: 'Optical + SAR pair',
}

/**
 * Result of the backend pair-compatibility check.
 *
 * This is what prevents change/fusion analysis from running on images that do not
 * describe the same area. The verdict comes from the server, not the browser.
 */
export function CompatibilityStatus({
  result,
  error,
}: {
  result: CompatibilityResponse | null
  error?: string | null
}) {
  if (error) {
    return (
      <div role="alert" className="rounded-md border border-destructive/50 p-3">
        <p className="text-xs text-destructive">{error}</p>
      </div>
    )
  }
  if (!result) return null

  return (
    <div className="space-y-2 rounded-md border bg-background/60 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium">
          {PURPOSE_LABELS[result.purpose] ?? result.purpose}
        </p>
        {result.compatible ? (
          <Badge variant="success">
            <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
            Compatible
          </Badge>
        ) : (
          <Badge variant="destructive">
            <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
            Not compatible
          </Badge>
        )}
      </div>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <dt className="text-muted-foreground">Spatial overlap</dt>
        <dd>
          {result.overlap_ratio == null
            ? 'Cannot be verified'
            : `${Math.round(result.overlap_ratio * 100)}%`}
        </dd>
        <dt className="text-muted-foreground">Same CRS</dt>
        <dd>{result.same_crs == null ? 'Unknown' : result.same_crs ? 'Yes' : 'No'}</dd>
        <dt className="text-muted-foreground">Resampling</dt>
        <dd>{result.resampling_required ? 'Required' : 'Not required'}</dd>
        <dt className="text-muted-foreground">Alignment</dt>
        <dd>{result.alignment_required ? 'Required' : 'Not required'}</dd>
      </dl>

      {result.issues.length > 0 && <IssueList issues={result.issues} />}

      {result.notes.length > 0 && (
        <ul className="space-y-1 text-xs text-muted-foreground">
          {result.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
