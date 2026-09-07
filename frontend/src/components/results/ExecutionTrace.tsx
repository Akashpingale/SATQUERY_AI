import { AlertTriangle, Check, MinusCircle, X } from 'lucide-react'

import type { TraceEntryOut } from '@/lib/api'

const STATUS_META: Record<string, { Icon: typeof Check; label: string; className: string }> = {
  ok: { Icon: Check, label: 'OK', className: 'text-success' },
  warning: { Icon: AlertTriangle, label: 'Warning', className: 'text-warning' },
  failed: { Icon: X, label: 'Failed', className: 'text-destructive' },
  skipped: { Icon: MinusCircle, label: 'Skipped', className: 'text-muted-foreground' },
}

/**
 * The observable execution trace: what the system did, in order.
 *
 * This is an audit log of concrete steps and outcomes. It intentionally contains no
 * model reasoning or chain-of-thought.
 */
export function ExecutionTrace({ trace }: { trace: TraceEntryOut[] }) {
  if (trace.length === 0) {
    return <p className="text-xs text-muted-foreground">No trace was recorded.</p>
  }

  return (
    <ol className="space-y-2" aria-label="Execution trace">
      {trace.map((entry, index) => {
        const meta = STATUS_META[entry.status] ?? STATUS_META.ok
        const { Icon, label, className } = meta
        return (
          <li key={`${entry.step}-${index}`} className="flex items-start gap-2 text-xs">
            <span className="mt-0.5 flex w-4 shrink-0 justify-center">
              <Icon className={`h-3.5 w-3.5 ${className}`} aria-hidden="true" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="font-medium">{entry.step}</span>
              <span className="text-muted-foreground"> · {label}</span>
              <span className="block break-words text-muted-foreground">{entry.detail}</span>
            </span>
          </li>
        )
      })}
    </ol>
  )
}
