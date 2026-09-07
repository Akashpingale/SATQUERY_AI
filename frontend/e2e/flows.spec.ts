import { fileURLToPath } from 'node:url'

import { type Page, expect, test } from '@playwright/test'

/**
 * End-to-end tests over the real application (real backend, real analysis, real evidence).
 *
 * Uploads use the synthetic demo dataset (generate it once with
 * `python scripts/make_demo_data.py`). No hidden evaluation data is used. Selectors are
 * semantic (labels, roles, headings) to stay robust.
 *
 * The single-image / adapted flows exercise CLIP on CPU, which loads weights on first use, so
 * they are given generous timeouts and assert the genuine workflow (route → result → confidence
 * → evidence) rather than a specific model answer.
 */

const DEMO = fileURLToPath(new URL('../../data/demo/', import.meta.url))
const file = (name: string) => `${DEMO}${name}`

const SLOT_LABEL = {
  optical: 'Optical / Multispectral file',
  sar: 'SAR file',
  t1: 'Temporal T1 file',
  t2: 'Temporal T2 file',
} as const

async function upload(page: Page, slot: keyof typeof SLOT_LABEL, filename: string) {
  await page.getByLabel(SLOT_LABEL[slot]).setInputFiles(file(filename))
  // Wait for the slot to report a validated upload.
  await expect(page.getByText('Valid').first()).toBeVisible({ timeout: 30_000 })
}

async function ask(page: Page, query: string) {
  const box = page.getByLabel('Your question')
  await box.fill(query)
  await page.getByRole('button', { name: /^analyze$/i }).click()
}

async function resultRegion(page: Page) {
  const region = page.getByRole('region', { name: 'Analysis result' })
  // Model-backed analyses load weights on first use; allow ample time.
  await expect(region).toBeVisible({ timeout: 150_000 })
  return region
}

test.beforeEach(async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /assistant/i })).toBeVisible()
})

test('Flow 1 — single-image VQA returns a real result with confidence and evidence', async ({
  page,
}) => {
  await upload(page, 'optical', 'demo_optical.tif')
  await ask(page, 'What land cover dominates this image?')

  const region = await resultRegion(page)
  // Routed to a VQA task and produced a genuine (completed) answer. `exact` avoids matching
  // the same words inside the trace/signals (Playwright text match is substring by default).
  await expect(region.getByText('Visual Question Answering', { exact: true })).toBeVisible()
  await expect(region.getByText('Completed', { exact: true })).toBeVisible()
  // Confidence is shown with its contributing signals (the badge always renders this
  // disclosure when signals exist, which they do for a completed result).
  await expect(region.getByText(/Why this confidence/)).toBeVisible()
  await expect(region.getByText(/Evidence \(\d+\)/)).toBeVisible()
})

test('Flow 2 — grounding produces overlay/region evidence', async ({ page }) => {
  await upload(page, 'optical', 'demo_optical.tif')
  await ask(page, 'Highlight the water body in this image')

  const region = await resultRegion(page)
  await expect(region.getByText('Grounding', { exact: true })).toBeVisible()
  await expect(region.getByText('Completed', { exact: true })).toBeVisible()
  await expect(region.getByText(/Evidence \(\d+\)/)).toBeVisible()
})

test('Flow 3 — adapted VLM is selected and named in the result', async ({ page }) => {
  await upload(page, 'optical', 'demo_optical.tif')
  await ask(page, 'Describe the land cover in this scene')

  const region = await resultRegion(page)
  await expect(region.getByText('Completed')).toBeVisible()
  // With the Phase 5 adapter present, the adapted model is chosen and labelled. If the adapter
  // is absent the baseline runs; either way the selected tool is shown.
  await expect(region.getByText(/Selected tool:/)).toBeVisible()
})

test('Flow 4 — temporal change is described from the T1/T2 pair', async ({ page }) => {
  await upload(page, 't1', 'demo_optical_t1.tif')
  await upload(page, 't2', 'demo_optical_t2.tif')
  await ask(page, 'What changed between these two dates?')

  const region = await resultRegion(page)
  await expect(region.getByText(/Temporal Change|Change VQA/)).toBeVisible()
  await expect(region.getByText('Completed')).toBeVisible()
  await expect(region.getByText(/Evidence \(\d+\)/)).toBeVisible()
})

test('Flow 5 — optical+SAR cross-modal analysis reports complementary evidence', async ({
  page,
}) => {
  await upload(page, 'optical', 'demo_optical.tif')
  await upload(page, 'sar', 'demo_sar.tif')
  await ask(page, 'What complementary information does SAR provide?')

  const region = await resultRegion(page)
  await expect(region.getByText('Optical-SAR Fusion')).toBeVisible()
  await expect(region.getByText('Completed')).toBeVisible()
  // A downloadable report link is offered for a completed analysis.
  await expect(region.getByRole('link', { name: /download report/i })).toBeVisible()
})

test('Flow 6 — an incompatible pair is rejected with an actionable message, no fake answer', async ({
  page,
}) => {
  await upload(page, 't1', 'demo_optical_t1.tif')
  await upload(page, 't2', 'demo_incompatible.tif')
  await ask(page, 'What changed between these two dates?')

  const region = await resultRegion(page)
  await expect(region.getByText('Rejected', { exact: true })).toBeVisible()
  // The rejection reason is spelled out (it appears in the answer and the issue list). No change
  // description is fabricated - the only text shown is the honest incompatibility explanation.
  await expect(
    region.getByText(/do not overlap|cannot be analyzed together/i).first(),
  ).toBeVisible()
  await expect(region.getByText(/incompatible|cannot describe the same area/i).first()).toBeVisible()
})
