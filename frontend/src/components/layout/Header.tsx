import { type ReactNode } from 'react'

import { Cpu, HelpCircle, Layers3, Satellite, Settings } from 'lucide-react'

import { SystemStatus, type SystemStatusState } from '@/components/SystemStatus'
import { Button } from '@/components/ui/button'
import { useHealth } from '@/hooks/useHealth'
import { useModels } from '@/hooks/useModels'

function deriveStatus(isLoading: boolean, isError: boolean, ok: boolean): SystemStatusState {
  if (isLoading) return 'checking'
  if (isError || !ok) return 'offline'
  return 'online'
}

/** A small read-only header chip: icon + label + value, never colour-only. */
function InfoChip({
  icon,
  label,
  value,
}: {
  icon: ReactNode
  label: string
  value: string
}) {
  return (
    <span
      className="hidden items-center gap-1.5 rounded-md border bg-card/70 px-2.5 py-1 text-xs text-muted-foreground sm:inline-flex"
      title={label}
    >
      <span className="text-muted-foreground/80" aria-hidden="true">
        {icon}
      </span>
      <span className="sr-only">{label}: </span>
      <span className="font-medium text-foreground">{value}</span>
    </span>
  )
}

export function Header() {
  const { data, isLoading, isError } = useHealth()
  const status = deriveStatus(isLoading, isError, data?.status === 'ok')

  const { data: models } = useModels()
  const availableTools = models?.tools.filter((tool) => tool.status === 'available').length
  const toolsValue =
    models == null ? '—' : `${availableTools ?? 0}/${models.count} tools`

  return (
    <header className="sticky top-0 z-30 border-b bg-panel-header/95 backdrop-blur supports-[backdrop-filter]:bg-panel-header/80">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-2.5">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent text-primary-foreground shadow-[0_0_18px_-2px_hsl(190_95%_60%_/_0.7)]">
            <Satellite className="h-5 w-5" aria-hidden="true" />
          </span>
          <div className="leading-tight">
            <span className="hud-title block text-[15px] font-semibold text-foreground">
              SatQuery AI
            </span>
            <span className="block text-xs text-muted-foreground">
              Remote-sensing intelligence workspace
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {data?.device && (
            <InfoChip
              icon={<Cpu className="h-3.5 w-3.5" />}
              label="Compute device"
              value={data.device.toUpperCase()}
            />
          )}
          <InfoChip icon={<Layers3 className="h-3.5 w-3.5" />} label="Registry" value={toolsValue} />
          <SystemStatus status={status} />
          <nav className="flex items-center gap-0.5 pl-1" aria-label="Utility navigation">
            <Button variant="ghost" size="icon" aria-label="Help and about">
              <HelpCircle className="h-5 w-5" aria-hidden="true" />
            </Button>
            <Button variant="ghost" size="icon" aria-label="Settings">
              <Settings className="h-5 w-5" aria-hidden="true" />
            </Button>
          </nav>
        </div>
      </div>
    </header>
  )
}
