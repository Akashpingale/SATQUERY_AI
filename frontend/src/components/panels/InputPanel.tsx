import { type ReactNode } from 'react'

import {
  CalendarClock,
  ImageIcon,
  Layers,
  Radar,
  SatelliteDish,
  SunMedium,
} from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Panel } from '@/components/ui/panel'
import { CompatibilityStatus } from '@/components/upload/CompatibilityStatus'
import { UploadSlot } from '@/components/upload/UploadSlot'
import type { UploadRole } from '@/lib/api'
import type { useUploads } from '@/hooks/useUploads'

interface SlotDef {
  role: UploadRole
  label: string
  hint: string
  icon: ReactNode
}

const ICON_CLASS = 'h-4 w-4'

/** Groups of upload slots, in the order they are presented. */
const GROUPS: { title: string; slots: SlotDef[] }[] = [
  {
    title: 'Optical / SAR',
    slots: [
      {
        role: 'optical',
        label: 'Optical / Multispectral',
        hint: 'Single-image analysis',
        icon: <SunMedium className={ICON_CLASS} />,
      },
      {
        role: 'sar',
        label: 'SAR',
        hint: 'Single-image or fusion',
        icon: <Radar className={ICON_CLASS} />,
      },
    ],
  },
  {
    title: 'Temporal pair',
    slots: [
      {
        role: 't1',
        label: 'Temporal T1',
        hint: 'Earlier date',
        icon: <CalendarClock className={ICON_CLASS} />,
      },
      {
        role: 't2',
        label: 'Temporal T2',
        hint: 'Later date',
        icon: <CalendarClock className={ICON_CLASS} />,
      },
    ],
  },
  {
    title: 'Optional / advanced',
    slots: [
      {
        role: 'single',
        label: 'Single image',
        hint: 'Any raster; the agent detects the modality',
        icon: <ImageIcon className={ICON_CLASS} />,
      },
    ],
  },
]

interface InputPanelProps {
  uploads: ReturnType<typeof useUploads>
  onInspect: (role: UploadRole) => void
  /** The slot currently shown in the viewer, for a subtle active marker. */
  viewedRole?: UploadRole | null
}

/**
 * The analysis inputs: modality-differentiated upload cards, a live summary of what the
 * agent will analyze, and the backend pair-compatibility check for temporal / fusion pairs.
 */
export function InputPanel({ uploads, onInspect, viewedRole }: InputPanelProps) {
  const { slots, upload, remove, checkPair, compatibility, compatibilityError } = uploads

  const hasTemporalPair = Boolean(slots.t1.asset && slots.t2.asset)
  const hasFusionPair = Boolean(slots.optical.asset && slots.sar.asset)
  const readyCount = (['optical', 'sar', 't1', 't2', 'single'] as UploadRole[]).filter(
    (role) => slots[role].status === 'ready',
  ).length

  return (
    <Panel
      title="Input workspace"
      description="Upload imagery; every file is validated on the server"
      icon={<SatelliteDish className="h-4 w-4" />}
      actions={
        readyCount > 0 ? (
          <Badge variant="secondary" className="tabular-nums">
            {readyCount} loaded
          </Badge>
        ) : undefined
      }
      bodyClassName="space-y-4"
    >
      <InputSummary
        hasTemporalPair={hasTemporalPair}
        hasFusionPair={hasFusionPair}
        opticalReady={slots.optical.status === 'ready'}
        sarReady={slots.sar.status === 'ready'}
        singleReady={slots.single.status === 'ready'}
      />

      {GROUPS.map((group) => (
        <div key={group.title} className="space-y-2">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            {group.title}
          </p>
          <div className="space-y-2.5">
            {group.slots.map((slot) => (
              <UploadSlot
                key={slot.role}
                role={slot.role}
                label={slot.label}
                hint={slot.hint}
                icon={slot.icon}
                state={slots[slot.role]}
                onSelect={upload}
                onRemove={remove}
                onInspect={onInspect}
                active={viewedRole === slot.role}
              />
            ))}
          </div>
        </div>
      ))}

      {(hasTemporalPair || hasFusionPair) && (
        <div className="space-y-2 border-t pt-3">
          <p className="flex items-center gap-1.5 text-xs font-medium">
            <Layers className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
            Pair compatibility
          </p>
          <div className="flex flex-wrap gap-2">
            {hasTemporalPair && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => checkPair('t1', 't2', 'temporal')}
              >
                Check T1 / T2
              </Button>
            )}
            {hasFusionPair && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => checkPair('optical', 'sar', 'optical_sar')}
              >
                Check Optical / SAR
              </Button>
            )}
          </div>
          <CompatibilityStatus result={compatibility} error={compatibilityError} />
        </div>
      )}
    </Panel>
  )
}

interface SummaryProps {
  hasTemporalPair: boolean
  hasFusionPair: boolean
  opticalReady: boolean
  sarReady: boolean
  singleReady: boolean
}

/**
 * A compact, always-honest read-out of the configuration the agent will analyze. It only
 * reflects what has actually been uploaded; the backend still decides the task.
 */
function InputSummary({
  hasTemporalPair,
  hasFusionPair,
  opticalReady,
  sarReady,
  singleReady,
}: SummaryProps) {
  const tags: string[] = []
  if (hasTemporalPair) tags.push('Temporal T1 + T2')
  if (hasFusionPair) tags.push('Optical + SAR')
  if (!hasFusionPair && opticalReady) tags.push('Single optical')
  if (!hasFusionPair && sarReady) tags.push('Single SAR')
  if (singleReady) tags.push('Single image')

  return (
    <div className="rounded-md border bg-secondary/40 px-3 py-2.5">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        Analysis configuration
      </p>
      {tags.length > 0 ? (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {tags.map((tag) => (
            <Badge key={tag} variant="default">
              {tag}
            </Badge>
          ))}
        </div>
      ) : (
        <p className="mt-1 text-xs text-muted-foreground">
          No imagery yet. Upload a raster to begin.
        </p>
      )}
    </div>
  )
}
