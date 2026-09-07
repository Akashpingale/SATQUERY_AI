/**
 * Typed API client. This is the ONLY place the frontend talks to the backend, so
 * error handling and the base URL are centralized. No secrets live here - only the
 * public, build-time `VITE_API_BASE_URL`.
 */

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1'

export interface DatabaseStatus {
  connected: boolean
  dialect: string
}

export interface HealthResponse {
  status: string
  app: string
  version: string
  environment: string
  device: string
  database: DatabaseStatus
}

export interface ToolSpec {
  id: string
  name: string
  version: string
  task: string
  modalities: string[]
  input_contract: string
  output_contract: string
  device: string
  checkpoint: string | null
  license: string
  status: string
  phase: number
}

export interface ModelsResponse {
  count: number
  tools: ToolSpec[]
  health: Record<string, string>
}

export interface TaskSpec {
  type: string
  title: string
  description: string
  min_images: number
  max_images: number
  modalities: string[]
  phase: number
}

export interface TasksResponse {
  count: number
  tasks: TaskSpec[]
}

// --- Raster uploads -------------------------------------------------------------

/** Slot a raster was uploaded into. */
export type UploadRole = 'single' | 'optical' | 'sar' | 't1' | 't2'
export type Modality = 'optical' | 'sar' | 'unknown'
export type CompatibilityPurpose = 'temporal' | 'optical_sar'

export interface Issue {
  code: string
  message: string
  severity: string
}

export interface BandInfo {
  index: number
  dtype: string
  nodata: number | null
  description: string | null
}

/** Georeferencing block. Every field is null for plain (non-georeferenced) images. */
export interface Georeference {
  is_georeferenced: boolean
  crs: string | null
  crs_epsg: number | null
  transform: number[] | null
  /** left, bottom, right, top */
  bounds: number[] | null
  resolution: number[] | null
}

export interface RasterAsset {
  id: string
  created_at: string
  original_filename: string
  size_bytes: number
  role: UploadRole
  modality: Modality
  detected_modality: Modality
  width: number | null
  height: number | null
  band_count: number | null
  dtype: string | null
  driver: string | null
  georeference: Georeference
  bands: BandInfo[]
  nodata: number | null
  tags: Record<string, string>
  preview_url: string | null
  warnings: Issue[]
}

export interface UploadListResponse {
  count: number
  assets: RasterAsset[]
}

export interface CompatibilityResponse {
  compatible: boolean
  purpose: string
  overlap_ratio: number | null
  same_crs: boolean | null
  resampling_required: boolean
  alignment_required: boolean
  issues: Issue[]
  notes: string[]
}

// --- Analysis ---------------------------------------------------------------------

/**
 * Outcome of an analysis run.
 * - `completed`   a genuine result was produced
 * - `rejected`    inputs or query could not be analyzed (with reasons)
 * - `unavailable` task understood and routed, but its specialist is not built yet
 * - `failed`      unexpected execution error
 */
export type AnalysisStatus = 'completed' | 'rejected' | 'unavailable' | 'failed'

export type ConfidenceLevel = 'High' | 'Medium' | 'Low' | 'Unable to determine'

export interface EvidenceOut {
  kind: string
  label: string
  asset_id: string | null
  detail: Record<string, unknown>
  reference: string | null
}

export interface ConfidenceOut {
  level: ConfidenceLevel | string
  score: number | null
  signals: string[]
}

export interface TraceEntryOut {
  step: string
  status: string
  detail: string
}

export interface PlanStepOut {
  order: number
  tool_id: string
  tool_version: string
  description: string
  parameters: Record<string, unknown>
}

export interface PlanOut {
  task: string
  steps: PlanStepOut[]
  /** Why each model was chosen (adapted preferred, or baseline fallback). */
  selection_notes?: string[]
}

export interface AnalysisResult {
  id: string
  created_at: string
  status: AnalysisStatus
  task: string
  query: string
  /** null whenever no genuine result could be produced. Never a placeholder. */
  answer: string | null
  confidence: ConfidenceOut
  evidence: EvidenceOut[]
  trace: TraceEntryOut[]
  plan: PlanOut | null
  issues: Issue[]
  warnings: string[]
  input_asset_ids: string[]
}

export interface HistoryItem {
  id: string
  created_at: string
  status: AnalysisStatus
  task: string | null
  query: string | null
  confidence_level: string | null
}

export interface HistoryResponse {
  count: number
  items: HistoryItem[]
}

export class ApiError extends Error {
  status: number

  /** Validation issues returned by the backend, when present. */
  issues: Issue[]

  constructor(message: string, status: number, issues: Issue[] = []) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.issues = issues
  }
}

/** Normalize a backend error body into a friendly message plus structured issues. */
async function toApiError(response: Response, path: string): Promise<ApiError> {
  let message = `Request to ${path} failed (${response.status}).`
  let issues: Issue[] = []
  try {
    const body = await response.json()
    const detail = body?.detail
    if (typeof detail === 'string') {
      message = detail
    } else if (detail && typeof detail === 'object') {
      if (typeof detail.message === 'string') message = detail.message
      if (Array.isArray(detail.issues)) issues = detail.issues as Issue[]
    } else if (body?.error?.message) {
      message = body.error.message as string
    }
  } catch {
    // Non-JSON body: keep the generic message.
  }
  return new ApiError(message, response.status, issues)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    // Network / CORS / server-down: normalized into a friendly message.
    throw new ApiError('Cannot reach the SatQuery AI backend.', 0)
  }
  if (!response.ok) {
    throw await toApiError(response, path)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

/** Absolute URL to download the self-contained HTML report for an analysis. */
export function reportUrl(analysisId: string): string {
  return `${API_BASE_URL}/analysis/${encodeURIComponent(analysisId)}/report`
}

/** Absolute URL for an asset preview, or null when no preview exists. */
export function previewUrl(asset: RasterAsset): string | null {
  if (!asset.preview_url) return null
  // The backend returns a path already prefixed with the API base (e.g. /api/v1/...).
  const origin = API_BASE_URL.replace(/\/api\/v1\/?$/, '')
  return asset.preview_url.startsWith('http')
    ? asset.preview_url
    : `${origin}${asset.preview_url}`
}

export const api = {
  getHealth: () => request<HealthResponse>('/health'),
  getModels: () => request<ModelsResponse>('/models'),
  getTasks: () => request<TasksResponse>('/tasks'),

  listUploads: () => request<UploadListResponse>('/uploads'),
  getUpload: (id: string) => request<RasterAsset>(`/uploads/${id}`),

  uploadRaster: async (
    file: File,
    role: UploadRole,
    modality: Modality = 'unknown',
  ): Promise<RasterAsset> => {
    const form = new FormData()
    form.append('file', file)
    form.append('role', role)
    form.append('modality', modality)
    const body = await request<{ asset: RasterAsset }>('/uploads', {
      method: 'POST',
      body: form,
    })
    return body.asset
  },

  deleteUpload: (id: string) => request<void>(`/uploads/${id}`, { method: 'DELETE' }),

  checkCompatibility: (assetIdA: string, assetIdB: string, purpose: CompatibilityPurpose) =>
    request<CompatibilityResponse>('/uploads/compatibility', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ asset_id_a: assetIdA, asset_id_b: assetIdB, purpose }),
    }),

  analyze: (query: string, assetIds: string[]) =>
    request<AnalysisResult>('/analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, asset_ids: assetIds }),
    }),

  getAnalysis: (id: string) => request<AnalysisResult>(`/analysis/${id}`),
  getHistory: () => request<HistoryResponse>('/history'),
}
