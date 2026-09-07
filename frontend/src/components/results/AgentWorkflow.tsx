import { AlertTriangle, Check, Loader2, MinusCircle, X } from 'lucide-react'

import type { TraceEntryOut } from '@/lib/api'
import { cn } from '@/lib/utils'

type StepState = 'done' | 'active' | 'pending' | 'warn' | 'fail' | 'skip'

interface CanonicalStep {
  key: string
  label: string
  /** Trace step names that map to this pipeline stage. */
  match: string[]
  /** When true, a trace step name is matched by prefix rather than equality. */
  prefix?: boolean
}

/** The six observable pipeline stages, in execution order. */
const STEPS: CanonicalStep[] = [
  { key: 'understand', label: 'Understand query', match: ['Query understanding', 'Task classification'] },
  { key: 'validate', label: 'Validate inputs', match: ['Input inspection', 'Validation'] },
  { key: 'select', label: 'Select tool', match: ['Tool selection'] },
  { key: 'analyze', label: 'Analyze imagery', match: ['Execution'], prefix: true },
  { key: 'evidence', label: 'Build evidence', match: ['Evidence'] },
  { key: 'confidence', label: 'Confidence', match: ['Confidence'] },
]

function statusFor(step: CanonicalStep, trace: TraceEntryOut[]): StepState {
  const matches = trace.filter((entry) =>
    step.prefix
      ? step.match.some((name) => entry.step.startsWith(name))
      : step.match.includes(entry.step),
  )
  if (matches.length === 0) return 'pending'
  if (matches.some((m) => m.status === 'failed')) return 'fail'
  if (matches.some((m) => m.status === 'warning')) return 'warn'
  if (matches.every((m) => m.status === 'skipped')) return 'skip'
  return 'done'
}

const STATE_META: Record<
  StepState,
  { Icon: typeof Check; ring: string; text: string; label: string }
> = {
  done: { Icon: Check, ring: 'border-success bg-success text-success-foreground', text: 'text-foreground', label: 'done' },
  active: {
    Icon: Loader2,
    ring: 'border-primary bg-primary text-primary-foreground',
    text: 'text-foreground',
    label: 'in progress',
  },
  pending: { Icon: MinusCircle, ring: 'border-border bg-secondary text-muted-foreground', text: 'text-muted-foreground', label: 'not run' },
  warn: { Icon: AlertTriangle, ring: 'border-warning bg-warning text-warning-foreground', text: 'text-foreground', label: 'warning' },
  fail: { Icon: X, ring: 'border-destructive bg-destructive text-destructive-foreground', text: 'text-foreground', label: 'failed' },
  skip: { Icon: MinusCircle, ring: 'border-border bg-secondary text-muted-foreground', text: 'text-muted-foreground', label: 'skipped' },
}

interface AgentWorkflowProps {
  /** Real execution trace; when present, per-stage status is derived from it. */
  trace?: TraceEntryOut[]
  /** While the pipeline runs (no trace yet), show an indeterminate in-progress state. */
  running?: boolean
  className?: string
}

/**
 * A compact visualization of the agent pipeline.
 *
 * - Idle: the six stages as an informational overview.
 * - Running: an honest indeterminate state (work happens server-side; we do not claim a
 *   specific stage completed until the trace says so).
 * - With a result: each stage's status is derived from the real execution trace.
 */
export function AgentWorkflow({ trace, running, className }: AgentWorkflowProps) {
  const states: StepState[] = STEPS.map((step) => {
    if (trace && trace.length > 0) return statusFor(step, trace)
    if (running) return 'active'
    return 'pending'
  })

  return (
    <ol className={cn('space-y-0', className)} aria-label="Agent pipeline">
      {STEPS.map((step, index) => {
        const state = states[index]
        const meta = STATE_META[state]
        const { Icon } = meta
        const isLast = index === STEPS.length - 1
        return (
          <li key={step.key} className="relative flex gap-2.5 pb-2 last:pb-0">
            {!isLast && (
              <span
                className="absolute left-[11px] top-6 h-[calc(100%-1rem)] w-px bg-border"
                aria-hidden="true"
              />
            )}
            <span
              className={cn(
                'relative z-10 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-[11px] font-semibold',
                meta.ring,
              )}
            >
              {running && !trace ? (
                <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
              ) : state === 'pending' ? (
                index + 1
              ) : (
                <Icon className="h-3.5 w-3.5" aria-hidden="true" />
              )}
            </span>
            <span className="flex min-w-0 flex-col pt-0.5">
              <span className={cn('text-xs font-medium leading-tight', meta.text)}>
                {step.label}
              </span>
              <span className="text-[11px] text-muted-foreground">{meta.label}</span>
            </span>
          </li>
        )
      })}
    </ol>
  )
}
