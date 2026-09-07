import type { AnalysisResult } from '@/lib/api'

/** A genuinely completed metadata-inspection result (measured, not modelled). */
export const completedResult: AnalysisResult = {
  id: 'an-1',
  created_at: '2026-01-01T10:00:00Z',
  status: 'completed',
  task: 'metadata_inspection',
  query: 'What CRS and resolution does this image use?',
  answer:
    'scene_optical.tif is a 512 x 512 pixel GTiff with 3 band(s) of type uint8, georeferenced in EPSG:4326.',
  confidence: {
    level: 'High',
    score: 0.949,
    signals: [
      'Specialist result strength 1.00',
      'Query routing score 0.86',
      'All inputs georeferenced',
      '2 evidence item(s) attached',
    ],
  },
  evidence: [
    {
      kind: 'metadata',
      label: 'Metadata for scene_optical.tif',
      asset_id: 'asset-geo-1',
      detail: { width: 512, height: 512, crs_epsg: 4326, band_count: 3 },
      reference: null,
    },
    {
      kind: 'preview',
      label: 'Preview of scene_optical.tif',
      asset_id: 'asset-geo-1',
      detail: {},
      reference: '/api/v1/uploads/asset-geo-1/preview',
    },
  ],
  trace: [
    { step: 'Input inspection', status: 'ok', detail: '1 image(s) - optical:scene_optical.tif' },
    { step: 'Query understanding', status: 'ok', detail: 'Query accepted (42 characters).' },
    {
      step: 'Task classification',
      status: 'ok',
      detail: 'Classified as metadata_inspection (routing score 0.86)',
    },
    { step: 'Validation', status: 'ok', detail: 'Inputs, task shape and parameters validated.' },
    {
      step: 'Tool selection',
      status: 'ok',
      detail: 'Step 1: raster_preprocessor v1.0.0 (parameters: max_preview_side=1024)',
    },
    { step: 'Execution (raster_preprocessor)', status: 'ok', detail: 'Metadata read.' },
    { step: 'Evidence', status: 'ok', detail: '2 item(s): 1 metadata, 1 preview' },
    { step: 'Confidence', status: 'ok', detail: 'High (score 0.95)' },
  ],
  plan: {
    task: 'metadata_inspection',
    steps: [
      {
        order: 1,
        tool_id: 'raster_preprocessor',
        tool_version: '1.0.0',
        description: 'Read raster metadata and render a preview',
        parameters: { max_preview_side: 1024 },
      },
    ],
  },
  issues: [],
  warnings: [],
  input_asset_ids: ['asset-geo-1'],
}

/** Correctly routed, but the specialist does not exist yet. No answer is invented. */
export const unavailableResult: AnalysisResult = {
  id: 'an-2',
  created_at: '2026-01-01T10:05:00Z',
  status: 'unavailable',
  task: 'vqa',
  query: 'What land cover dominates this image?',
  answer: null,
  confidence: {
    level: 'Unable to determine',
    score: null,
    signals: ['No result was produced because the specialist is unavailable.'],
  },
  evidence: [],
  trace: [
    { step: 'Input inspection', status: 'ok', detail: '1 image(s)' },
    { step: 'Task classification', status: 'ok', detail: 'Classified as vqa (routing score 0.60)' },
    {
      step: 'Execution (vqa)',
      status: 'skipped',
      detail: "The 'VQA' specialist is not available yet (status: planned).",
    },
  ],
  plan: {
    task: 'vqa',
    steps: [
      {
        order: 1,
        tool_id: 'vqa',
        tool_version: '0.0.0',
        description: 'Answer the question about the image',
        parameters: { max_new_tokens: 64 },
      },
    ],
  },
  issues: [
    {
      code: 'tool_unavailable',
      message:
        'Visual Question Answering is understood and correctly routed, but its specialist is not implemented yet (arrives in phase 4).',
      severity: 'error',
    },
  ],
  warnings: [],
  input_asset_ids: ['asset-geo-1'],
}

/** An incompatible temporal pair: rejected before any change analysis runs. */
export const rejectedResult: AnalysisResult = {
  id: 'an-3',
  created_at: '2026-01-01T10:10:00Z',
  status: 'rejected',
  task: 'change_vqa',
  query: 'What changed between these two dates?',
  answer:
    't1.tif and elsewhere.tif cannot be analyzed together. The images do not overlap geographically, so they cannot describe the same area.',
  confidence: {
    level: 'High',
    score: 0.62,
    signals: ['Specialist reported a definite negative finding'],
  },
  evidence: [
    {
      kind: 'comparison',
      label: 'Pair compatibility assessment',
      asset_id: null,
      detail: { compatible: false, overlap_ratio: 0 },
      reference: null,
    },
  ],
  trace: [
    { step: 'Execution (registration_validator)', status: 'ok', detail: 'Checked pair.' },
    { step: 'Workflow', status: 'failed', detail: 'Stopped after compatibility check.' },
  ],
  plan: null,
  issues: [
    {
      code: 'incompatible_inputs',
      message: 'The images do not overlap geographically.',
      severity: 'error',
    },
  ],
  warnings: ['Spatial overlap could not be confirmed.'],
  input_asset_ids: ['a', 'b'],
}

/**
 * A result produced by the Phase 5 adapted model, shaped like the real API response
 * (including the model-variant evidence the backend attaches).
 */
export const adaptedResult: AnalysisResult = {
  ...completedResult,
  id: 'an-4',
  task: 'vqa',
  query: 'What land cover dominates this image?',
  answer: 'The scene is most consistent with water (70%). The next closest match is a river (16%).',
  evidence: [
    ...completedResult.evidence,
    {
      kind: 'model',
      label: 'Model variant',
      asset_id: null,
      detail: {
        variant: 'adapted',
        version: 'clip-rsicd-v2+lora-ben-txt-r16a32',
        selection: "registry tool 'vqa_adapted'",
        base_model: 'flax-community/clip-rsicd-v2',
        adapter_path: 'models/phase5/adapter',
        training_dataset: 'BigEarthNet.txt (Lithuania/Summer captioning subset)',
        training_method: 'LoRA, symmetric contrastive (InfoNCE) objective',
        lora_rank: 16,
        lora_alpha: 32,
      },
      reference: null,
    },
  ],
  plan: {
    task: 'vqa',
    steps: [
      {
        order: 1,
        tool_id: 'vqa_adapted',
        tool_version: 'clip-rsicd-v2+lora-ben-txt',
        description: 'Answer the question about the image',
        parameters: {},
      },
    ],
    selection_notes: ['vqa: selected the Phase 5 adapted model (adapter present)'],
  },
  warnings: [
    'This answer used the Phase 5 adapted model (LoRA on BigEarthNet.txt captions). Adaptation was trained on a Lithuania/Summer Sentinel-2 subset, so its benefit on other regions, seasons, or sensors is not established.',
  ],
}
