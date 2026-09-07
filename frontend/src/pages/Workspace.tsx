import { useCallback, useEffect, useState } from 'react'

import { motion } from 'framer-motion'

import { Header } from '@/components/layout/Header'
import { AssistantPanel } from '@/components/panels/AssistantPanel'
import { HistoryPanel } from '@/components/panels/HistoryPanel'
import { InputPanel } from '@/components/panels/InputPanel'
import { VisualizationPanel } from '@/components/panels/VisualizationPanel'
import { useAnalysis } from '@/hooks/useAnalysis'
import { useUploads } from '@/hooks/useUploads'
import type { UploadRole } from '@/lib/api'
import { exampleQuestions } from '@/lib/examples'

/** Roles the agent should analyze, in the order they are offered to the backend. */
const ANALYSIS_ROLES: UploadRole[] = ['t1', 't2', 'optical', 'sar', 'single']

export function Workspace() {
  const uploads = useUploads()
  const analysis = useAnalysis()
  const [viewedRole, setViewedRole] = useState<UploadRole | null>(null)

  // Show the most recent upload automatically so the viewer is never empty when a
  // valid raster exists.
  useEffect(() => {
    if (viewedRole && uploads.slots[viewedRole].asset) return
    const next = (['optical', 'sar', 't1', 't2', 'single'] as UploadRole[]).find(
      (role) => uploads.slots[role].asset,
    )
    setViewedRole(next ?? null)
  }, [uploads.slots, viewedRole])

  const viewedAsset = viewedRole ? (uploads.slots[viewedRole].asset ?? null) : null

  /**
   * Collect the asset ids to analyze.
   *
   * A temporal pair takes precedence, then optical+SAR, otherwise the single available
   * image. The backend accepts at most two, and it - not the UI - decides which task the
   * combination supports.
   */
  const collectAssetIds = useCallback((): string[] => {
    const { slots } = uploads
    if (slots.t1.asset && slots.t2.asset) {
      return [slots.t1.asset.id, slots.t2.asset.id]
    }
    if (slots.optical.asset && slots.sar.asset) {
      return [slots.optical.asset.id, slots.sar.asset.id]
    }
    const first = ANALYSIS_ROLES.map((role) => slots[role].asset).find(Boolean)
    return first ? [first.id] : []
  }, [uploads])

  const assetIds = collectAssetIds()

  const examples = exampleQuestions({
    hasSingle: Boolean(uploads.slots.single.asset),
    hasOptical: Boolean(uploads.slots.optical.asset),
    hasSar: Boolean(uploads.slots.sar.asset),
    hasTemporalPair: Boolean(uploads.slots.t1.asset && uploads.slots.t2.asset),
    hasOpticalSarPair: Boolean(uploads.slots.optical.asset && uploads.slots.sar.asset),
  })

  const handleAnalyze = useCallback(
    (query: string) => {
      const ids = collectAssetIds()
      if (ids.length > 0) void analysis.analyze(query, ids)
    },
    [analysis, collectAssetIds],
  )

  return (
    // On large screens the shell is locked to the viewport height so each workspace
    // column scrolls independently (no page scroll swallows the panels). On small
    // screens it falls back to normal document flow so everything is reachable by
    // scrolling the page.
    <div className="flex min-h-screen flex-col bg-background lg:h-screen lg:min-h-0">
      <Header />
      <main className="space-backdrop relative flex-1 lg:min-h-0 lg:overflow-hidden">
        <div className="mx-auto w-full max-w-[1600px] px-4 py-4 lg:h-full">
          <h1 className="sr-only">SatQuery AI analysis workspace</h1>
          <div className="grid grid-cols-1 gap-4 lg:h-full lg:grid-cols-12">
            <motion.aside
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
              className="space-y-4 lg:col-span-3 lg:h-full lg:min-h-0 lg:overflow-y-auto lg:pr-1"
              aria-label="Inputs and history"
            >
              <InputPanel uploads={uploads} onInspect={setViewedRole} viewedRole={viewedRole} />
              <HistoryPanel onReopen={(id) => void analysis.load(id)} />
            </motion.aside>
            <motion.section
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, ease: 'easeOut', delay: 0.05 }}
              className="lg:col-span-6 lg:h-full lg:min-h-0 lg:overflow-hidden"
              aria-label="Visualization"
            >
              <VisualizationPanel
                asset={viewedAsset}
                t1={uploads.slots.t1.asset ?? null}
                t2={uploads.slots.t2.asset ?? null}
                evidence={analysis.result?.evidence ?? []}
              />
            </motion.section>
            <motion.aside
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, ease: 'easeOut', delay: 0.1 }}
              className="lg:col-span-3 lg:h-full lg:min-h-0 lg:overflow-hidden lg:pl-1"
              aria-label="Assistant"
            >
              <AssistantPanel
                hasImagery={assetIds.length > 0}
                imageCount={assetIds.length}
                isRunning={analysis.isRunning}
                result={analysis.result}
                error={analysis.error}
                onAnalyze={handleAnalyze}
                examples={examples}
              />
            </motion.aside>
          </div>
        </div>
      </main>
      <footer className="border-t bg-panel-header">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-2 px-4 py-3 text-xs text-muted-foreground">
          <span>SatQuery AI · SIH26167 · Agentic remote-sensing analysis</span>
          <span>Optical · SAR · Temporal change · Evidence &amp; trace</span>
        </div>
      </footer>
    </div>
  )
}
