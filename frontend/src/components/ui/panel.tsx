import * as React from 'react'

import { cn } from '@/lib/utils'

interface PanelProps extends Omit<React.HTMLAttributes<HTMLElement>, 'title'> {
  /** Section title, rendered as a heading for accessible document structure. */
  title: React.ReactNode
  /** Short supporting line under the title. */
  description?: React.ReactNode
  /** Leading glyph shown in a tinted square, communicating the panel's purpose. */
  icon?: React.ReactNode
  /** Right-aligned controls in the header (toolbar, counts, badges). */
  actions?: React.ReactNode
  /** Heading tag for the title (keeps a sensible document outline). */
  as?: 'h2' | 'h3'
  /** Extra classes for the scrollable body. */
  bodyClassName?: string
  /** Render the panel as a flex column that fills its parent height. */
  fill?: boolean
}

/**
 * A titled instrument panel: a card with a distinct header row (icon, title,
 * description, actions) and a content body.
 *
 * This is the primary structural unit of the workspace. Using a real heading for the
 * title keeps the page navigable by screen-reader landmarks and headings.
 */
export const Panel = React.forwardRef<HTMLElement, PanelProps>(
  (
    { title, description, icon, actions, as = 'h2', bodyClassName, fill, className, children, ...props },
    ref,
  ) => {
    const Heading = as
    return (
      <section
        ref={ref}
        className={cn(
          'flex flex-col overflow-hidden rounded-lg border bg-panel text-panel-foreground shadow-panel',
          fill && 'h-full',
          className,
        )}
        {...props}
      >
        <div className="flex shrink-0 items-center gap-3 border-b bg-panel-header px-4 py-3">
          {icon && (
            <span
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-primary/20 bg-primary/10 text-primary"
              aria-hidden="true"
            >
              {icon}
            </span>
          )}
          <div className="min-w-0 flex-1 leading-tight">
            <Heading className="truncate text-sm font-semibold tracking-tight">{title}</Heading>
            {description && (
              <p className="truncate text-xs text-muted-foreground">{description}</p>
            )}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </div>
        {/*
         * When `fill` is set the panel occupies its column's full height and the BODY
         * scrolls internally, so the header stays pinned and no content is ever clipped
         * out of reach. Without `fill` the body grows naturally and any scrolling is
         * handled by the surrounding column.
         */}
        <div className={cn('flex-1 p-4', fill && 'min-h-0 overflow-y-auto', bodyClassName)}>
          {children}
        </div>
      </section>
    )
  },
)
Panel.displayName = 'Panel'
