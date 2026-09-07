import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import { type ReactElement } from 'react'
import { axe } from 'vitest-axe'
import { describe, expect, it } from 'vitest'

import { AssistantPanel } from '@/components/panels/AssistantPanel'
import { HistoryPanel } from '@/components/panels/HistoryPanel'
import { VisualizationPanel } from '@/components/panels/VisualizationPanel'
import { ConfidenceBadge } from '@/components/results/ConfidenceBadge'
import { EvidencePanel } from '@/components/results/EvidencePanel'
import { ExecutionTrace } from '@/components/results/ExecutionTrace'
import { ResultCard } from '@/components/results/ResultCard'
import type { EvidenceOut } from '@/lib/api'

import { adaptedResult, completedResult, rejectedResult, unavailableResult } from './analysisFixtures'
import { georeferencedAsset } from './fixtures'

/**
 * Automated accessibility checks (axe-core) over the important application states.
 *
 * `color-contrast` is disabled: jsdom does not compute layout or resolve CSS colours, so axe
 * cannot evaluate contrast here. Contrast is verified manually against the Tailwind palette and
 * documented in `docs/accessibility.md`. Every other WCAG rule axe supports is enforced.
 */
const AXE_OPTIONS = { rules: { 'color-contrast': { enabled: false } } }

async function expectNoViolations(ui: ReactElement) {
  const { container } = render(ui)
  const results = await axe(container, AXE_OPTIONS)
  // Assert on rule ids so a failure names the exact violated rule.
  expect(results.violations.map((v) => v.id)).toEqual([])
}

const examples = ['Describe this scene', 'What land cover dominates this image?']
const base = {
  hasImagery: true,
  imageCount: 1,
  isRunning: false,
  result: null,
  error: null,
  onAnalyze: () => {},
  examples,
}

const fusionEvidence: EvidenceOut[] = [
  {
    kind: 'fusion_composite',
    label: 'Optical+SAR composite',
    asset_id: null,
    detail: { note: 'Reddish = SAR-strong.', correlation: 0.85, complementarity: 0.15 },
    reference: '/api/v1/evidence/aaaabbbbccccdddd',
  },
  {
    kind: 'cross_modal_water',
    label: 'Water agreement',
    asset_id: null,
    detail: { available: true, agreement_iou: 0.62, agree_fraction: 0.15 },
    reference: null,
  },
]

function withQueryClient(ui: ReactElement): ReactElement {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>
}

describe('accessibility (axe)', () => {
  it('query panel — empty state has no violations', async () => {
    await expectNoViolations(<AssistantPanel {...base} hasImagery={false} imageCount={0} />)
  })

  it('query panel — running state has no violations', async () => {
    await expectNoViolations(<AssistantPanel {...base} isRunning />)
  })

  it('query panel — error state has no violations', async () => {
    await expectNoViolations(
      <AssistantPanel {...base} error="Cannot reach the SatQuery AI backend." />,
    )
  })

  it('completed result has no violations', async () => {
    await expectNoViolations(<AssistantPanel {...base} result={completedResult} />)
  })

  it('rejected result has no violations', async () => {
    await expectNoViolations(<ResultCard result={rejectedResult} />)
  })

  it('unavailable result has no violations', async () => {
    await expectNoViolations(<ResultCard result={unavailableResult} />)
  })

  it('adapted result has no violations', async () => {
    await expectNoViolations(<ResultCard result={adaptedResult} />)
  })

  it('evidence panel (cross-modal) has no violations', async () => {
    await expectNoViolations(<EvidencePanel evidence={fusionEvidence} />)
  })

  it('execution trace has no violations', async () => {
    await expectNoViolations(<ExecutionTrace trace={completedResult.trace} />)
  })

  it('confidence badge has no violations', async () => {
    await expectNoViolations(<ConfidenceBadge confidence={completedResult.confidence} />)
  })

  it('visualization panel (single image) has no violations', async () => {
    await expectNoViolations(<VisualizationPanel asset={georeferencedAsset} />)
  })

  it('visualization panel (optical+SAR) has no violations', async () => {
    await expectNoViolations(
      <VisualizationPanel asset={georeferencedAsset} evidence={fusionEvidence} />,
    )
  })

  it('history panel (empty) has no violations', async () => {
    await expectNoViolations(withQueryClient(<HistoryPanel />))
  })
})
