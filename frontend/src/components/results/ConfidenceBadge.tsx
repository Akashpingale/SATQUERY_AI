import { CircleHelp, ShieldAlert, ShieldCheck, ShieldQuestion } from 'lucide-react'

import { Badge, type BadgeProps } from '@/components/ui/badge'
import type { ConfidenceOut } from '@/lib/api'

const CONFIG: Record<
  string,
  { variant: BadgeProps['variant']; Icon: typeof ShieldCheck; bar: string }
> = {
  High: { variant: 'success', Icon: ShieldCheck, bar: 'bg-success' },
  Medium: { variant: 'warning', Icon: ShieldQuestion, bar: 'bg-warning' },
  Low: { variant: 'destructive', Icon: ShieldAlert, bar: 'bg-destructive' },
  'Unable to determine': { variant: 'secondary', Icon: CircleHelp, bar: 'bg-muted-foreground' },
}

/**
 * Confidence level with the signals that produced it.
 *
 * The level is always shown as text alongside an icon (never colour alone). When a numeric
 * score exists it is drawn as a meter with the percentage in text, and the contributing
 * signals are listed so the value is auditable rather than asserted.
 */
export function ConfidenceBadge({ confidence }: { confidence: ConfidenceOut }) {
  const config = CONFIG[confidence.level] ?? CONFIG['Unable to determine']
  const { variant, Icon, bar } = config
  const percent =
    confidence.score != null ? Math.round(Math.max(0, Math.min(1, confidence.score)) * 100) : null

  return (
    <div className="space-y-1.5 rounded-md border bg-secondary/30 p-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">Confidence</span>
        <Badge variant={variant}>
          <Icon className="h-3.5 w-3.5" aria-hidden="true" />
          {confidence.level}
        </Badge>
        {confidence.score != null && (
          <span className="text-xs tabular-nums text-muted-foreground">
            score {confidence.score.toFixed(2)}
          </span>
        )}
      </div>

      {percent != null && (
        <div className="flex items-center gap-2">
          <span
            className="h-2 flex-1 overflow-hidden rounded-full bg-secondary"
            role="img"
            aria-label={`Confidence score ${percent} percent`}
          >
            <span className={`block h-full rounded-full ${bar}`} style={{ width: `${percent}%` }} />
          </span>
          <span className="w-9 shrink-0 text-right text-xs font-medium tabular-nums">
            {percent}%
          </span>
        </div>
      )}

      {confidence.signals.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">
            Why this confidence ({confidence.signals.length} signal
            {confidence.signals.length === 1 ? '' : 's'})
          </summary>
          <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-muted-foreground">
            {confidence.signals.map((signal) => (
              <li key={signal}>{signal}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}
