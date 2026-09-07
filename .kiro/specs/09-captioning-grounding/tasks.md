# Spec 09 — Captioning & Grounding · Tasks

## Captioning
- [x] 1. Retrieval-based composition from real concept scores (primary + secondary thresholds)
- [x] 2. Explicitly labelled as retrieval-composed, not generated (answer, warning, evidence)
- [x] 3. Tentative wording plus a warning when no concept scores strongly
- [x] 4. Evidence: ranked scores with method, model provenance, preview
- [x] 5. Registry entry with real version/checkpoint/license, honest `degraded` status
- [x] 6. Tests including the "must not claim to be generative" assertion

## Grounding
- [x] 7. OWL-ViT open-vocabulary detection returning real boxes with scores
- [x] 8. Target extraction from concepts/intent, with filler-and-pronoun refusal
- [x] 9. Overlay rendering with per-box score labels; edge clamping
- [x] 10. Pixel -> source pixel -> CRS -> lon/lat mapping through the decimation factor
- [x] 11. Coarse tile-scoring fallback with a real contrast measure and a minimum threshold
- [x] 12. Three honest outcomes (boxes / coarse region / inconclusive)
- [x] 13. `GET /api/v1/evidence/{token}` with token validation and path confinement
- [x] 14. Tests: rendering, coordinates, refusal, all three outcomes, traversal rejection

## Frontend
- [x] 15. Overlay-first evidence ordering; geometry, score bars, provenance rendering
- [x] 16. Accessible presentation (text percentages, labelled bars, alt text)
- [x] 17. Component tests

## Remaining
- [ ] Segmentation masks in addition to boxes
- [ ] Grounding IoU / localization benchmark — Phase 9
- [ ] Caption quality metrics on a real captioning dataset — Phase 9
- [ ] Interactive overlay drawn on the main viewer canvas (currently a rendered PNG) — Phase 8
- [x] Captioning served by the Phase 5 adapted model (`captioning_adapted` registry entry)
- [ ] Re-evaluate the detector against a remote-sensing-trained alternative. **Not addressed in
      Phase 5**: the LoRA adaptation covered CLIP only, so grounding still uses the unchanged
      OWL-ViT baseline and no improvement is claimed for it.
