/**
 * Input-adaptive example questions.
 *
 * The examples shown next to the query box adapt to what the user has actually uploaded, so
 * they only ever suggest questions the current inputs can satisfy. This mirrors the backend's
 * feasibility filter (a temporal question needs two images; a cross-modal question needs one
 * optical + one SAR), so a chip never proposes something the agent would reject.
 *
 * The examples are just prompts - the real backend still classifies and answers them. Nothing
 * here hard-codes an answer.
 */

export interface InputContext {
  /** A single image in the generic slot. */
  hasSingle: boolean
  hasOptical: boolean
  hasSar: boolean
  /** Both temporal slots filled (T1 + T2). */
  hasTemporalPair: boolean
  /** One optical + one SAR filled. */
  hasOpticalSarPair: boolean
}

const NONE: readonly string[] = [
  'What CRS and resolution does this image use?',
  'What bands and data type does this image have?',
]

const OPTICAL: readonly string[] = [
  'Describe this scene',
  'What land cover dominates this image?',
  'Where is water visible in this image?',
  'Highlight the water body in this image',
]

const SAR: readonly string[] = [
  'Describe the SAR scene',
  'Where are the strong backscatter regions?',
  'What structures are visible in this SAR image?',
  'What does the SAR image reveal?',
]

const OPTICAL_SAR: readonly string[] = [
  'Analyze these optical and SAR images together',
  'What complementary information does SAR provide?',
  'Where do optical and SAR agree on water?',
  'What does SAR reveal that is less apparent in the optical image?',
  'Which modality provides stronger evidence of vegetation?',
]

const TEMPORAL: readonly string[] = [
  'What changed between these two dates?',
  'Where did the strongest changes occur?',
  'How much of the shared area changed?',
  'Has vegetation decreased between these dates?',
]

/**
 * Choose example questions for the current input configuration.
 *
 * Pair configurations take precedence over single-image ones, because when a user has uploaded
 * a pair they almost certainly want a pair analysis. Returns at most `limit` questions.
 */
export function exampleQuestions(context: InputContext, limit = 5): readonly string[] {
  let pool: readonly string[]
  if (context.hasOpticalSarPair) {
    pool = OPTICAL_SAR
  } else if (context.hasTemporalPair) {
    pool = TEMPORAL
  } else if (context.hasSar && !context.hasOptical && !context.hasSingle) {
    pool = SAR
  } else if (context.hasOptical || context.hasSingle || context.hasSar) {
    pool = OPTICAL
  } else {
    pool = NONE
  }
  return pool.slice(0, limit)
}
