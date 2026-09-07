import { Clock, History as HistoryIcon } from 'lucide-react'

import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Panel } from '@/components/ui/panel'
import { useHistory } from '@/hooks/useAnalysis'

const STATUS_VARIANT: Record<string, BadgeProps['variant']> = {
  completed: 'success',
  rejected: 'destructive',
  unavailable: 'warning',
  failed: 'destructive',
}

const STATUS_LABEL: Record<string, string> = {
  completed: 'Completed',
  rejected: 'Rejected',
  unavailable: 'Unavailable',
  failed: 'Failed',
}

const TASK_LABELS: Record<string, string> = {
  vqa: 'VQA',
  vqa_adapted: 'VQA',
  captioning: 'Captioning',
  captioning_adapted: 'Captioning',
  grounding: 'Grounding',
  temporal_change: 'Temporal change',
  change_vqa: 'Change VQA',
  optical_sar_fusion: 'Optical-SAR',
  metadata_inspection: 'Metadata',
  unsupported: 'Unsupported',
}

function formatTime(iso: string): string {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString()
}

export function HistoryPanel({ onReopen }: { onReopen?: (id: string) => void }) {
  const { data, isLoading, isError } = useHistory()
  const items = data?.items ?? []

  return (
    <Panel
      title="History"
      description="Past analyses, most recent first"
      icon={<HistoryIcon className="h-4 w-4" />}
      actions={
        items.length > 0 ? (
          <Badge variant="secondary" className="tabular-nums">
            {items.length}
          </Badge>
        ) : undefined
      }
    >
      {isLoading && <p className="text-xs text-muted-foreground">Loading history…</p>}

      {isError && (
        <p role="alert" className="text-xs text-destructive">
          History could not be loaded.
        </p>
      )}

      {!isLoading && !isError && items.length === 0 && (
        <div className="flex flex-col items-center gap-2 rounded-md border border-dashed bg-secondary/30 px-3 py-6 text-center">
          <Clock className="h-5 w-5 text-muted-foreground" aria-hidden="true" />
          <p className="text-sm font-medium">No analyses yet</p>
          <p className="text-xs text-muted-foreground">
            Your completed analyses will appear here.
          </p>
        </div>
      )}

      {items.length > 0 && (
        <ul className="max-h-72 space-y-1.5 overflow-y-auto">
          {items.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onReopen?.(item.id)}
                disabled={!onReopen}
                className="w-full rounded-md border bg-card/60 px-2.5 py-2 text-left transition-colors hover:border-primary/40 hover:bg-primary/[0.04] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:hover:border-border disabled:hover:bg-card/60"
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-medium" title={item.query ?? ''}>
                    {item.query || 'Untitled analysis'}
                  </span>
                  <Badge variant={STATUS_VARIANT[item.status] ?? 'secondary'}>
                    {STATUS_LABEL[item.status] ?? item.status}
                  </Badge>
                </span>
                <span className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
                  <span className="rounded border bg-secondary/60 px-1.5 py-0.5 font-medium">
                    {TASK_LABELS[item.task ?? ''] ?? item.task ?? 'unknown'}
                  </span>
                  {item.confidence_level && <span>· {item.confidence_level}</span>}
                  <span>· {formatTime(item.created_at)}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}
