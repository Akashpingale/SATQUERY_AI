import { type DragEvent, type KeyboardEvent, type ReactNode, useId, useRef, useState } from 'react'

import { CheckCircle2, Loader2, Trash2, Upload, XCircle } from 'lucide-react'

import { IssueList } from '@/components/upload/IssueList'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { UploadRole } from '@/lib/api'
import { formatBytes, formatCrs, formatDimensions } from '@/lib/format'
import type { SlotState } from '@/hooks/useUploads'
import { cn } from '@/lib/utils'

const ACCEPT = '.tif,.tiff,.geotiff,.png,.jpg,.jpeg'

interface UploadSlotProps {
  role: UploadRole
  label: string
  hint: string
  state: SlotState
  onSelect: (role: UploadRole, file: File) => void
  onRemove: (role: UploadRole) => void
  onInspect?: (role: UploadRole) => void
  /** Modality glyph shown in the card header. */
  icon?: ReactNode
  /** Whether this slot is currently the one shown in the viewer. */
  active?: boolean
}

/**
 * A single upload target supporting drag-and-drop and the keyboard/file picker.
 *
 * The card visibly transforms from an empty drop zone into a validated summary once a
 * raster is accepted. Accessibility: the drop zone is a real <button> so it is focusable
 * and operable with Enter/Space; status is conveyed with an icon plus text, never colour
 * alone.
 */
export function UploadSlot({
  role,
  label,
  hint,
  state,
  onSelect,
  onRemove,
  onInspect,
  icon,
  active,
}: UploadSlotProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const describedById = useId()

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0]
    if (file) onSelect(role, file)
  }

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault()
    setIsDragging(false)
    handleFiles(event.dataTransfer.files)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      inputRef.current?.click()
    }
  }

  const { status, asset, error, issues } = state
  const isBusy = status === 'uploading'
  const isReady = status === 'ready' && Boolean(asset)

  return (
    <div
      className={cn(
        'rounded-md border bg-card transition-colors',
        isReady ? 'border-primary/30 bg-primary/[0.03]' : 'bg-card/60',
        status === 'error' && 'border-destructive/40',
        active && isReady && 'ring-1 ring-primary/40',
      )}
    >
      <div className="flex items-center gap-2.5 px-3 pt-3">
        <span
          className={cn(
            'flex h-8 w-8 shrink-0 items-center justify-center rounded-md border',
            isReady
              ? 'border-primary/30 bg-primary/10 text-primary'
              : 'border-border bg-secondary/60 text-muted-foreground',
          )}
          aria-hidden="true"
        >
          {icon ?? <Upload className="h-4 w-4" />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium leading-tight">{label}</p>
          <p className="text-xs text-muted-foreground">{hint}</p>
        </div>
        {isReady && (
          <Badge variant="success">
            <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
            Valid
          </Badge>
        )}
        {status === 'error' && (
          <Badge variant="destructive">
            <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
            Rejected
          </Badge>
        )}
        {isBusy && (
          <Badge variant="secondary">
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            Checking
          </Badge>
        )}
      </div>

      <div className="p-3">
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          aria-label={`${label} file`}
          onChange={(event) => {
            handleFiles(event.target.files)
            event.target.value = ''
          }}
        />

        {isReady && asset ? (
          <div className="space-y-2.5">
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
              <dt className="text-muted-foreground">File</dt>
              <dd className="truncate font-medium" title={asset.original_filename}>
                {asset.original_filename}
              </dd>
              <dt className="text-muted-foreground">Size</dt>
              <dd className="tabular-nums">{formatDimensions(asset)}</dd>
              <dt className="text-muted-foreground">Bands</dt>
              <dd className="tabular-nums">
                {asset.band_count ?? '—'} · {asset.dtype ?? '—'}
              </dd>
              <dt className="text-muted-foreground">On disk</dt>
              <dd className="tabular-nums">{formatBytes(asset.size_bytes)}</dd>
            </dl>

            <div className="flex flex-wrap gap-1.5">
              <span className="data-chip">{formatCrs(asset)}</span>
              <span className="data-chip">
                {asset.georeference.is_georeferenced ? 'Georeferenced' : 'No georeferencing'}
              </span>
            </div>

            {asset.warnings.length > 0 && <IssueList issues={asset.warnings} />}

            <div className="flex flex-wrap gap-2 pt-0.5">
              {onInspect && (
                <Button variant="outline" size="sm" onClick={() => onInspect(role)}>
                  View
                </Button>
              )}
              <Button variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
                Replace
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onRemove(role)}
                aria-label={`Remove ${label}`}
              >
                <Trash2 className="h-4 w-4" aria-hidden="true" />
                Remove
              </Button>
            </div>
          </div>
        ) : (
          <>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              onKeyDown={handleKeyDown}
              onDragOver={(event) => {
                event.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              disabled={isBusy}
              aria-describedby={describedById}
              className={cn(
                'flex w-full flex-col items-center justify-center gap-1.5 rounded-md border border-dashed px-3 py-6 text-center transition-colors',
                'hover:border-primary hover:bg-primary/[0.04] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
                isDragging && 'border-primary bg-primary/[0.06]',
                isBusy && 'cursor-wait opacity-70',
                status === 'error' && 'border-destructive/60',
              )}
            >
              {isBusy ? (
                <Loader2
                  className="h-5 w-5 animate-spin text-muted-foreground"
                  aria-hidden="true"
                />
              ) : (
                <Upload className="h-5 w-5 text-muted-foreground" aria-hidden="true" />
              )}
              <span className="text-xs font-medium">
                {isBusy ? 'Validating…' : 'Drop a file or click to browse'}
              </span>
            </button>
            <p id={describedById} className="mt-1.5 text-center text-[11px] text-muted-foreground">
              GeoTIFF, TIFF, PNG, or JPEG.
            </p>

            {status === 'error' && (
              <div role="alert" className="mt-2 space-y-1.5">
                <p className="flex items-start gap-2 text-xs text-destructive">
                  <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span>{error}</span>
                </p>
                {issues && issues.length > 0 && <IssueList issues={issues} />}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
