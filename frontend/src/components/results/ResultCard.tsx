import {
  CircleAlert,
  CircleCheck,
  Clock,
  Download,
  ListChecks,
  Sparkles,
  TriangleAlert,
  Wrench,
} from 'lucide-react'

import { motion } from 'framer-motion'

import { AgentWorkflow } from '@/components/results/AgentWorkflow'
import { ConfidenceBadge } from '@/components/results/ConfidenceBadge'
import { EvidencePanel } from '@/components/results/EvidencePanel'
import { ExecutionTrace } from '@/components/results/ExecutionTrace'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { IssueList } from '@/components/upload/IssueList'
import { type AnalysisResult, reportUrl } from '@/lib/api'

const TASK_LABELS: Record<string, string> = {
  vqa: 'Visual Question Answering',
  captioning: 'Captioning',
  vqa_adapted: 'Visual Question Answering',
  captioning_adapted: 'Captioning',
  grounding: 'Grounding',
  temporal_change: 'Temporal Change',
  change_vqa: 'Change VQA',
  optical_sar_fusion: 'Optical-SAR Fusion',
  metadata_inspection: 'Metadata Inspection',
  unsupported: 'Unsupported',
}

const STATUS_META: Record<
  string,
  { label: string; variant: BadgeProps['variant']; Icon: typeof CircleCheck }
> = {
  completed: { label: 'Completed', variant: 'success', Icon: CircleCheck },
  rejected: { label: 'Rejected', variant: 'destructive', Icon: CircleAlert },
  unavailable: { label: 'Not yet available', variant: 'warning', Icon: Clock },
  failed: { label: 'Failed', variant: 'destructive', Icon: TriangleAlert },
}

/**
 * Presents an analysis outcome.
 *
 * All four outcomes are shown truthfully. In particular, `unavailable` means the query
 * was understood and routed correctly but the specialist has not been built yet - the UI
 * says so and shows the routing evidence instead of displaying a stand-in answer.
 */
export function ResultCard({ result }: { result: AnalysisResult }) {
  const status = STATUS_META[result.status] ?? STATUS_META.failed
  const { label, variant, Icon } = status
  const taskLabel = TASK_LABELS[result.task] ?? result.task
  const selectedTool = result.plan?.steps.at(-1)
  // A result from the fine-tuned model is labelled, so baseline and adapted answers are
  // never confused. Derived from the model-variant evidence the backend attaches.
  const isAdapted = result.evidence.some((item) => item.detail?.variant === 'adapted')
  // Prototype: the team's trained BLIP models are highlighted so a demo can point at them.
  const TRAINED_TOOL_IDS = new Set(['captioning_blip', 'change_vqa_blip'])
  const isTrained = selectedTool ? TRAINED_TOOL_IDS.has(selectedTool.tool_id) : false

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className="space-y-3"
      aria-label="Analysis result"
    >
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={variant}>
          <Icon className="h-3.5 w-3.5" aria-hidden="true" />
          {label}
        </Badge>
        <Badge variant="outline">{taskLabel}</Badge>
      </div>

      <div className="rounded-md border border-primary/25 bg-card p-3 shadow-[0_0_24px_-10px_hsl(190_95%_60%_/_0.5)]">
        {result.query && (
          <p className="mb-2 border-l-2 border-primary/40 pl-2.5 text-xs italic text-muted-foreground">
            {result.query}
          </p>
        )}
        {result.answer ? (
          <>
            <p className="hud-title mb-1 text-[11px] font-semibold text-primary">Answer</p>
            <p className="glow-cyan whitespace-pre-line text-sm leading-relaxed text-foreground">
              {result.answer}
            </p>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">No answer was produced for this request.</p>
        )}
      </div>

      {result.issues.length > 0 && <IssueList issues={result.issues} />}

      {result.warnings.length > 0 && (
        <ul className="space-y-1 text-xs text-muted-foreground">
          {result.warnings.map((warning) => (
            <li key={warning} className="flex items-start gap-2">
              <TriangleAlert
                className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning"
                aria-hidden="true"
              />
              <span>{warning}</span>
            </li>
          ))}
        </ul>
      )}

      <ConfidenceBadge confidence={result.confidence} />

      {selectedTool && (
        <div className="space-y-1 rounded-md border bg-secondary/30 p-2.5">
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <p className="flex flex-wrap items-center gap-1.5">
              <Wrench className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Selected tool:</span>
              <span className="font-mono font-medium text-foreground">{selectedTool.tool_id}</span>
              {selectedTool.tool_version ? (
                <span className="tabular-nums">v{selectedTool.tool_version}</span>
              ) : null}
            </p>
            {isTrained && (
              <Badge variant="default">
                <Sparkles className="h-3 w-3" aria-hidden="true" />
                Our trained model
              </Badge>
            )}
            {isAdapted && (
              <Badge variant="default">
                <Sparkles className="h-3 w-3" aria-hidden="true" />
                Adapted model
              </Badge>
            )}
          </div>
          {result.plan?.selection_notes?.map((note) => (
            <p key={note} className="text-xs text-muted-foreground">
              {note}
            </p>
          ))}
        </div>
      )}

      {result.trace.length > 0 && (
        <div className="rounded-md border bg-secondary/30 p-2.5">
          <p className="mb-2 flex items-center gap-1.5 text-xs font-medium">
            <ListChecks className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
            Agent plan
          </p>
          <AgentWorkflow trace={result.trace} />
        </div>
      )}

      <details className="rounded-md border bg-card p-2.5" open>
        <summary className="cursor-pointer text-xs font-medium">
          Evidence ({result.evidence.length})
        </summary>
        <div className="mt-2">
          <EvidencePanel evidence={result.evidence} />
        </div>
      </details>

      <details className="rounded-md border bg-card p-2.5">
        <summary className="cursor-pointer text-xs font-medium">
          Execution trace ({result.trace.length} steps)
        </summary>
        <div className="mt-2">
          <ExecutionTrace trace={result.trace} />
        </div>
      </details>

      {result.id && (
        <a
          href={reportUrl(result.id)}
          download
          className="inline-flex items-center gap-1.5 rounded-md border bg-card px-3 py-1.5 text-xs font-medium transition-colors hover:border-primary/40 hover:bg-primary/[0.04] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <Download className="h-3.5 w-3.5" aria-hidden="true" />
          Download report
        </a>
      )}
    </motion.section>
  )
}
