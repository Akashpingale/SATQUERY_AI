# Spec 09 — Captioning & Grounding · Design (as built, Phase 4)

Both additional single-image capabilities are implemented, satisfying the master plan's
"at least one, preferably both".

## Captioning

**Method: retrieval-based composition, not generation.** The image is scored against the
24-concept remote-sensing vocabulary; concepts clearing a threshold are assembled into a
description carrying their measured probabilities.

```
top concept >= 15%  -> "The scene appears to be dominated by <label> (NN%)"
top concept <  15%  -> "No single feature dominates; the closest match is <label> (NN%)"
+ up to 3 secondary concepts >= 5%  -> ", with signs of <label> (NN%), ..."
```

Module: `backend/app/agent/specialists/captioning.py`.

### Why not generative
A generative captioner trained on natural photographs produces fluent but frequently wrong
descriptions of overhead imagery. Composing from a remote-sensing-adapted model's actual
scores yields a plainer description that is faithful to what was measured. Every result
carries a warning stating it is retrieval-composed, and the evidence records
`method: "retrieval-based composition from zero-shot concept scores"`. Calling it
"generated" would misrepresent it.

## Grounding

**Primary path: real open-vocabulary detection.** `google/owlvit-base-patch32` localizes a
free-text target phrase and returns boxes with scores.

```
query -> find_concepts() or intent target -> detection phrase
image -> load_display_image() -> decimated RGB + scale factor
      -> detect() -> boxes with scores
      -> draw_detections() -> overlay PNG (each box labelled with its score)
      -> box_to_geo() -> source pixels -> CRS -> lon/lat
```

Module: `backend/app/agent/specialists/grounding.py`.

### Target extraction and refusal
The target comes from a recognised remote-sensing concept (whose prompt phrasing is known
to work) or from the intent target extracted during query understanding. A filler/pronoun
stoplist rejects meaningless targets: "outline it for me" is **refused** rather than
searched for as the phrase "it for me".

### Coordinate mapping
Boxes are in decimated-image pixels. Dividing by the decimation `scale` recovers
full-resolution source pixels; the raster's affine transform maps those to CRS coordinates,
and pyproj converts to WGS84 lon/lat. Non-georeferenced images report coordinates as
unavailable rather than emitting fabricated numbers.

### Coarse fallback (`clip_patch_scoring`)
OWL-ViT is trained on natural photographs and frequently returns **nothing** for diffuse
overhead land cover such as water or farmland. Rather than reporting "not found" for a
concept the scene plainly contains, the image is split into a 4×4 grid, each tile is scored
against the target phrase with the remote-sensing CLIP model, and the best tiles are merged
into a region.

- **Contrast** = `(best_tile - median_tile) / (best - worst)`, a real measure of how much
  the winning tiles stand out. A normalised maximum would always be 1.0 and therefore
  meaningless; this varies with how localized the response actually is (measured: 74 % for
  water vs 38 % for buildings on the same scene).
- Regions below `MIN_PATCH_CONTRAST` (0.15) are **not** reported — a flat response means the
  phrase distinguishes nothing.
- The reported score is `contrast * 0.5`, so a tile estimate can never imply detector-grade
  certainty.
- Evidence kind is `region` (not `box`), the answer says "tile contrast" and "no distinct
  object boundary", and a warning states it is "not an object outline".

### Three honest outcomes
| Outcome | Evidence | Confidence |
|---|---|---|
| Detector boxes | `box` items + `overlay` | Model score |
| Coarse region | `region` item + `overlay` | Reduced (contrast × 0.5) |
| Nothing | preview only | `Unable to determine`, answer says "inconclusive" |

A null result is never presented as evidence of absence.

## Evidence serving
Overlays are written to `storage/evidence/overlays/<token>.png` and served by
`GET /api/v1/evidence/{token}`. Tokens are server-generated hex, validated against a strict
pattern, and resolved through the path-confinement helper — so the endpoint cannot read
arbitrary files. A test asserts traversal attempts return 404.

## Frontend
`EvidencePanel` renders overlays first (visual evidence leads), then geometry (score, pixel
box, lat/lon or an explicit "unavailable"), score bars with text percentages, and model
provenance. Percentages are always text as well as bar length, so nothing depends on colour
or size perception.

## Tests
Overlay rendering (including out-of-bounds and top-edge label clamping), pixel→world
coordinate mapping with decimation, target refusal, real-inference grounding across all
three outcomes, contrast non-degeneracy, and evidence-endpoint traversal rejection.

## Known limits
- Detection transfers poorly to diffuse land cover; the coarse fallback is tile-resolution.
- Captioning is not generative and cannot describe relationships or unusual content.
- No masks yet (boxes and tile regions only).
- No grounding IoU benchmark yet — Phase 9.
- Registry status stays `degraded` after Phase 5. Captioning now runs on the adapted model
  (`captioning_adapted`); **grounding was deliberately not adapted**, so no grounding
  improvement is claimed and a test asserts it has no adapted preference.
