import { AlertTriangle, Info, XCircle } from 'lucide-react'

import type { Issue } from '@/lib/api'

/**
 * Renders validation issues in plain language. Severity is shown with an icon plus a
 * text prefix, never colour alone.
 */
export function IssueList({ issues, className }: { issues: Issue[]; className?: string }) {
  if (issues.length === 0) return null

  return (
    <ul className={className ?? 'space-y-1.5'}>
      {issues.map((issue) => {
        const isError = issue.severity === 'error'
        const Icon = isError ? XCircle : issue.severity === 'warning' ? AlertTriangle : Info
        const label = isError ? 'Error' : issue.severity === 'warning' ? 'Warning' : 'Note'
        return (
          <li key={`${issue.code}-${issue.message}`} className="flex items-start gap-2 text-xs">
            <Icon
              className={
                isError
                  ? 'mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive'
                  : 'mt-0.5 h-3.5 w-3.5 shrink-0 text-warning'
              }
              aria-hidden="true"
            />
            <span className={isError ? 'text-destructive' : 'text-muted-foreground'}>
              <span className="font-medium">{label}: </span>
              {issue.message}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
