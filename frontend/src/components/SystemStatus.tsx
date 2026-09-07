import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'

import { Badge, type BadgeProps } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

export type SystemStatusState = 'online' | 'offline' | 'checking'

interface StatusConfig {
  label: string
  variant: BadgeProps['variant']
  Icon: typeof CheckCircle2
  spin?: boolean
}

const CONFIG: Record<SystemStatusState, StatusConfig> = {
  online: { label: 'Online', variant: 'success', Icon: CheckCircle2 },
  offline: { label: 'Offline', variant: 'destructive', Icon: AlertCircle },
  checking: { label: 'Checking', variant: 'secondary', Icon: Loader2, spin: true },
}

/**
 * System status indicator. Conveys state with BOTH an icon and a text label (never
 * color alone) and announces changes to assistive tech via role="status".
 */
export function SystemStatus({ status }: { status: SystemStatusState }) {
  const { label, variant, Icon, spin } = CONFIG[status]
  return (
    <Badge variant={variant} role="status" aria-live="polite">
      <Icon className={cn('h-3.5 w-3.5', spin && 'animate-spin')} aria-hidden="true" />
      <span>{label}</span>
      <span className="sr-only">Backend status: {label}</span>
    </Badge>
  )
}
