import { type FormEvent, useState } from 'react'

import { Bot, CornerDownLeft, Loader2, MessageSquareText, Sparkles } from 'lucide-react'

import { AgentWorkflow } from '@/components/results/AgentWorkflow'
import { ResultCard } from '@/components/results/ResultCard'
import { Button } from '@/components/ui/button'
import { Panel } from '@/components/ui/panel'
import type { AnalysisResult } from '@/lib/api'

const MAX_QUERY_LENGTH = 1000

interface AssistantPanelProps {
  hasImagery: boolean
  imageCount: number
  isRunning: boolean
  result: AnalysisResult | null
  error: string | null
  onAnalyze: (query: string) => void
  /** Example questions adapted to the current input configuration. */
  examples: readonly string[]
}

export function AssistantPanel({
  hasImagery,
  imageCount,
  isRunning,
  result,
  error,
  onAnalyze,
  examples,
}: AssistantPanelProps) {
  const [query, setQuery] = useState('')

  const trimmed = query.trim()
  const canSubmit = hasImagery && trimmed.length > 0 && !isRunning

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (canSubmit) onAnalyze(trimmed)
  }

  const description = hasImagery
    ? `Ask a question about your ${imageCount === 1 ? 'image' : `${imageCount} images`}.`
    : 'Upload a raster first, then ask a question about it.'

  return (
    <Panel
      title="AI Assistant"
      description={description}
      icon={<Bot className="h-4 w-4" />}
      fill
      bodyClassName="flex flex-col gap-4"
    >
      <form onSubmit={handleSubmit} className="space-y-2">
        <label htmlFor="query" className="sr-only">
          Your question
        </label>
        <div className="rounded-md border bg-background focus-within:ring-2 focus-within:ring-ring">
          <textarea
            id="query"
            rows={3}
            value={query}
            maxLength={MAX_QUERY_LENGTH}
            disabled={!hasImagery || isRunning}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. Has the built-up area increased between these two dates?"
            aria-describedby="query-help"
            className="w-full resize-none rounded-md bg-transparent px-3 py-2.5 text-sm placeholder:text-muted-foreground focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-70"
          />
          <div className="flex items-center justify-between gap-2 border-t px-3 py-1.5">
            <span className="text-[11px] tabular-nums text-muted-foreground">
              {trimmed.length}/{MAX_QUERY_LENGTH}
            </span>
            <span className="hidden items-center gap-1 text-[11px] text-muted-foreground sm:inline-flex">
              <CornerDownLeft className="h-3 w-3" aria-hidden="true" />
              agent picks the task
            </span>
          </div>
        </div>
        <p id="query-help" className="text-xs text-muted-foreground">
          {hasImagery
            ? 'The agent picks the task and tools automatically.'
            : 'Disabled until at least one raster is uploaded.'}
        </p>

        <Button type="submit" disabled={!canSubmit} className="w-full">
          {isRunning ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <Sparkles className="h-4 w-4" aria-hidden="true" />
          )}
          {isRunning ? 'Analyzing…' : 'Analyze'}
        </Button>
      </form>

      {examples.length > 0 && (
        <div className="space-y-1.5">
          <p id="examples-label" className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            {hasImagery ? 'Suggested for your inputs' : 'Example questions'}
          </p>
          <ul className="space-y-1" aria-labelledby="examples-label">
            {examples.map((question) => (
              <li key={question}>
                <button
                  type="button"
                  onClick={() => setQuery(question)}
                  disabled={isRunning}
                  className="group flex w-full items-center gap-2 rounded-md border bg-card/60 px-2.5 py-1.5 text-left text-xs transition-colors hover:border-primary/40 hover:bg-primary/[0.04] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-60"
                >
                  <MessageSquareText
                    className="h-3.5 w-3.5 shrink-0 text-muted-foreground group-hover:text-primary"
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1">{question}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div aria-live="polite" aria-busy={isRunning} className="flex-1 space-y-3">
        {error && (
          <div role="alert" className="rounded-md border border-destructive/50 bg-destructive/5 p-3">
            <p className="text-xs font-medium text-destructive">{error}</p>
          </div>
        )}

        {isRunning && (
          <div className="rounded-md border bg-secondary/30 p-3">
            <p className="mb-2.5 text-xs text-muted-foreground">
              Running the agent pipeline: classifying the query, validating inputs, and selecting
              tools…
            </p>
            <AgentWorkflow running />
          </div>
        )}

        {!isRunning && result && <ResultCard result={result} />}

        {!isRunning && !result && !error && (
          <div className="rounded-md border bg-secondary/30 p-3">
            <p className="mb-2.5 flex items-center gap-1.5 text-xs font-medium">
              <Sparkles className="h-3.5 w-3.5 text-primary" aria-hidden="true" />
              Analysis pipeline
            </p>
            <AgentWorkflow />
          </div>
        )}
      </div>
    </Panel>
  )
}
