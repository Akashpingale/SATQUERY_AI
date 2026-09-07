# Spec 08 — Single-Image VQA · Design (as built, Phase 4)

## Approach
Zero-shot visual question answering with `flax-community/clip-rsicd-v2` (CLIP fine-tuned on
RSICD, a remote-sensing captioning dataset). The image is scored against a remote-sensing
concept vocabulary; the question's *type* determines how those scores become an answer.

Modules: `ml/inference/{vocabulary,clip_backend}.py`,
`backend/app/agent/specialists/vqa.py`.

## Pipeline
```
question -> analyze_question()  -> QuestionKind + referenced concepts
image    -> load_display_image() -> decimated RGB (512 px max side)
         -> score_concepts()     -> softmax probabilities over the vocabulary
         -> answer strategy per QuestionKind
```

## Question kinds and strategies
| Kind | Trigger | Strategy |
|---|---|---|
| `PRESENCE` | "is water present", "are there buildings" | Report the target concept's probability and rank; verdict from a threshold **or** rank 1 |
| `DOMINANT` | "what land cover dominates", "classify the land use" | Top-scoring land-cover concept, plus runner-up for context |
| `WHAT_VISIBLE` | "what is visible", "what objects" | List concepts above 5 %, up to 4 |
| `COUNT` | "how many", "number of" | **Refused** — see below |
| `UNSUPPORTED` | empty question | Reported as such |

## The counting refusal (central honesty requirement)
A contrastive image-text model has no counting mechanism. Asked "how many buildings?", the
specialist returns an explanation and `score=None`, which makes confidence
`Unable to determine`. A test asserts the answer contains **no digits at all**, so nothing
in it can be misread as a count.

## Vocabulary
24 concepts drawn from common land-cover taxonomies (EuroSAT / UCM / RSICD style):
water, river, dense residential, residential, industrial, commercial, farmland, forest,
grassland, shrubland, bare land, desert, beach, mountain, road, railway, bridge, parking lot,
airport, harbor, storage tanks, sports field, snow, cloud.

Deliberately general rather than tuned to any demo image, so unseen scenes are described
with the same concepts. Each concept carries trigger words for question matching and a flag
for whether it can serve as a "dominant land cover" answer.

## Prompt ensembling
Each concept is expanded over 4 aerial framings ("an aerial photograph of {}", "a satellite
image of {}", "an aerial view of {}", "a remote sensing image of {}"); their text embeddings
are averaged and re-normalised. This is a standard zero-shot robustness technique.

**Text embeddings are cached** per concept set: they depend only on the vocabulary, never
on the image. This cut warm latency from ~3.0 s to ~0.06 s.

## Uncertainty handling
- Probabilities are a softmax over the model's own vocabulary — relative, not calibrated.
  Every result carries a warning saying so.
- A top concept below 20 % is reported as a weak reading, with a warning.
- Two close top classes produce "the scene is likely mixed" plus a warning.
- The confidence score handed to the confidence layer is the model's **measured
  probability**, never a constant.

## Evidence
- `scores` — ranked concept probabilities (machine-readable, rendered as labelled bars).
- `model` — provenance: exact model id, device, and method note.
- `preview` — the image the answer refers to.

## Tests
`backend/tests/test_vision_specialists.py`: question-kind classification, concept triggers,
vocabulary consistency, the counting refusal (including the no-digits assertion), and
real-inference tests asserting probabilities are in range, ordered, and non-degenerate.
Model-dependent tests skip when weights are not cached; tests never download.

## Known limits
- Zero-shot and uncalibrated; accuracy is not established (no benchmark evaluation yet —
  that is Phase 9).
- No counting, no spatial reasoning, no fine-grained attribute questions.
- Vocabulary is English-only and fixed; a concept outside it cannot be reported.
- Registry status stays `degraded` after Phase 5 adaptation: the adapted model measurably
  improved image↔text retrieval, but VQA accuracy itself is still unbenchmarked (Phase 9) and
  scores remain uncalibrated.
