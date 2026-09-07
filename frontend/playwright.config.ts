import path from 'node:path'

import { defineConfig, devices } from '@playwright/test'

// The config lives in frontend/; the repo root is one level up. Absolute paths are used for the
// backend launcher because Playwright runs webServer commands through cmd.exe on Windows, which
// does not accept a leading "../" forward-slash path.
const REPO_ROOT = path.resolve(process.cwd(), '..')
const PYTHON = path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
const E2E_BACKEND = path.join(REPO_ROOT, 'scripts', 'e2e_backend.py')

/**
 * Playwright end-to-end configuration.
 *
 * Two real servers are started for the run: the FastAPI backend (isolated SQLite + storage, via
 * scripts/e2e_backend.py) and the Vite dev server. The tests drive the actual application - real
 * uploads, real analysis, real evidence - so a passing E2E run is meaningful, not mocked.
 *
 * Browser binaries must be installed once: `npx playwright install chromium`.
 */
export default defineConfig({
  testDir: './e2e',
  // Model-backed flows (CLIP/OWL-ViT on CPU) load weights on first use, so allow generous
  // timeouts. Grounding loads two models, which is the slowest path.
  timeout: 200_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
    actionTimeout: 20_000,
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      // Isolated backend (throwaway SQLite + storage), started from the repo root.
      command: `"${PYTHON}" "${E2E_BACKEND}"`,
      cwd: REPO_ROOT,
      url: 'http://127.0.0.1:8000/api/v1/health',
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      command: 'npm run dev',
      url: 'http://localhost:5173',
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
    },
  ],
})
