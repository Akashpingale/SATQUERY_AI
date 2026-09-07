# Phase 11 — Final Public Benchmark Validation

## 1. Executive Summary
Phase 11 validated SatQuery AI's capabilities against **public** remote-sensing benchmarks,
evidence-first. Two change-detection benchmarks were **actually evaluated on their full public
test splits** (LEVIR-CD n=2048, OSCD n=50); the VQA/captioning benchmarks were found
**architecturally incompatible** with the current specialists (documented, not faked); and
VRSBench grounding is **compatible but blocked** on a ~4 GB image download. No production model
was changed, no test set was tuned, no hidden ISRO/SAC data was used, and the external Phase 10
`change_model` remains **not integrated**.

Headline honest finding: the learned SECOND change detector is strong **in-domain** (SECOND
IoU 0.42) but **generalizes poorly** cross-dataset (LEVIR-CD IoU 0.077, OSCD IoU 0.105), and on
OSCD the deterministic method is actually better (IoU 0.145) — empirically supporting the policy
of keeping the deterministic detector as the robust production default.

## 2. Current System Before Phase 11
Unchanged from Phase 10: deterministic CVA temporal detector; learned `second_change_learned`
(SECOND in-domain IoU 0.4219 / F1 0.5934); `cdvqa_trained` (CDVQA Test 0.7114 / Test2 0.6632);
Phase 5 BigEarthNet LoRA adaptation; optical+SAR fusion; zero-shot VQA/grounding; retrieval
captioning; agent orchestration, evidence, confidence, trace, report generation; fixed image
preview. External `change_model` = VERIFIED_BUT_NOT_INTEGRATED.

## 3. RSVQA
- **Compatibility:** ❌ INCOMPATIBLE. The production VQA specialist is zero-shot CLIP
  concept-scoring over a fixed vocabulary; it cannot emit RSVQA's yes/no, counting, or
  area/comparison answers.
- **Dataset / split / model / evaluation:** not run (incompatibility is architectural, not a
  data-availability issue). Public RSVQA-LR/HR mirrors exist (e.g. `dmarsili/RSVQA-LR-2k`).
- **Result:** none (would be misleading). **Limitation:** requires a constrained/generative VQA
  head to produce valid answers — out of scope for this validation phase.

## 4. VRSBench — VQA
❌ INCOMPATIBLE, same reason as RSVQA (free-form/open answers vs fixed concept vocabulary).

## 5. VRSBench — Captioning
❌ INCOMPATIBLE. The captioner is retrieval/concept composition, not generative; it cannot
produce valid predictions for generative-caption metrics (BLEU/CIDEr/METEOR/ROUGE). Not
misrepresented as generative captioning.

## 6. VRSBench — Grounding
❌ BLOCKED (architecturally **compatible**, not run). OWL-ViT open-vocabulary detection maps to
VRSBench referring-expression grounding via box IoU@0.5. The referring annotations were
downloaded and the format confirmed (`referring.json`: `question` = referring expression,
`ground_truth` = normalized `<x1><y1><x2><y2>` box, `image_id`). Blocked only on the val image
set (`Images_val.zip` ≈ 3.98 GB) not provisioned here. Reproducible command recorded.

## 7. LEVIR-CD — ✅ FULL TEST EVALUATION (n=2048)
- **Dataset:** `ericyu/LEVIRCD_Cropped_256` public test split (256×256 crops, binary
  changed/unchanged; changed-pixel fraction 5.1%).
- **Models:** learned `second_change_learned` (frozen Phase 9 SECOND checkpoint — cross-dataset)
  and a deterministic Otsu radiometric-difference reference.
- **Result:** learned IoU **0.0773**, F1 **0.1435**, P 0.083, R 0.527; Otsu IoU 0.0466, F1 0.0891.
- **Limitations:** cross-dataset (never trained on LEVIR). LEVIR labels *building* change; the
  SECOND model flags *any* land-cover change → heavy false positives (low precision). Learned >
  deterministic here, but both are far below in-domain SECOND (IoU 0.42). `levir_cd_results.json`.

## 8. OSCD — ✅ FULL TEST EVALUATION (n=50)
- **Dataset:** `ericyu/OSCD_Cropped_256` public test split (Sentinel-2 RGB 256×256 crops,
  changed-pixel fraction 5.4%).
- **Result:** learned IoU **0.1053**, F1 **0.1905**; Otsu reference IoU **0.1453**, F1 **0.2538**,
  R 0.549. **The deterministic reference beats the learned model.**
- **Limitations:** Sentinel-2 sensor/resolution differs sharply from SECOND aerial imagery;
  learned model does not transfer. RGB-only (the RGB crop dataset was used; full 13-band MSI is a
  separate `blanchon/OSCD_MSI` not evaluated). `oscd_results.json`.

## 9. Data Leakage Audit
- LEVIR-CD & OSCD: **test split only**; no training, threshold tuning, or model selection on
  them. Learned model = frozen SECOND checkpoint; Otsu reference is unsupervised per-image →
  **zero test-set leakage**.
- No hidden ISRO/SAC data used. Public HF mirrors only; small test parquets downloaded (LEVIR
  69.5 MB, OSCD 1.49 MB) to `D:\dataset\...` outside the repo, not committed.

## 10. Model Comparison (like-for-like only)
Change-detection IoU/F1: SECOND in-domain 0.42/0.59 ≫ LEVIR-CD 0.077/0.144 ≈ OSCD 0.105/0.191.
Deterministic vs learned: learned wins on LEVIR, deterministic wins on OSCD. VQA-accuracy numbers
(CDVQA 0.71; external BLIP 0.55) are kept strictly separate from IoU/F1.

## 11. SIH26167 Capability Matrix
See `PHASE11_SIH_CAPABILITY_MATRIX.md`. PASS: CDVQA, SECOND in-domain, BigEarthNet adaptation,
agentic orchestration, evidence/trace. PARTIAL: cross-dataset change (LEVIR/OSCD), optical+SAR.
INCOMPATIBLE: single-image VQA, captioning. BLOCKED: VRSBench grounding (image data).

## 12. What Is Proven
- CDVQA change-VQA (0.71), SECOND in-domain change (IoU 0.42), BigEarthNet adaptation (retrieval
  6.9×) — all on real public data with held-out evaluation.
- Change detection **runs on external public benchmarks** (LEVIR-CD, OSCD) end-to-end.

## 13. What Is Partially Proven
- Cross-dataset change-detection generalization: measured but weak (LEVIR 0.077, OSCD 0.105).
- Optical+SAR fusion: validated on synthetic truth; no matching public benchmark exists.

## 14. What Is Not Yet Proven
- Single-image VQA / captioning on RSVQA/VRSBench (architecture incompatible).
- VRSBench grounding accuracy (compatible but image data not provisioned).

## 15. Blockers
- VRSBench grounding: `Images_val.zip` ≈ 3.98 GB (and train ≈ 8.3 GB) not provisioned.
- VQA/captioning: require new trained heads (a constrained/generative VQA head; a generative
  captioner) — deliberately out of scope for a *validation* phase.

## 16. Recommendations for Phase 12
1. If VQA benchmarking is required, train a constrained VQA head (RSVQA answer space) or adopt a
   generative VLM — then evaluate RSVQA/VRSBench-VQA properly.
2. Provision VRSBench val images and run the OWL-ViT grounding eval (adapter already scoped).
3. If cross-dataset change matters, fine-tune the change detector on LEVIR/OSCD train splits
   (kept separate from test) rather than relying on SECOND transfer.

## 17. Reproducibility Commands
```
# LEVIR-CD (downloads a 69.5 MB public test parquet to D:\dataset\levir_cd\)
python scripts/phase11/eval_levir_cd.py --parquet D:\dataset\levir_cd\test.parquet --out-file levir_cd_results.json
# OSCD (1.49 MB public test parquet)
python scripts/phase11/eval_levir_cd.py --parquet D:\dataset\oscd\test.parquet \
  --name "OSCD (public test split, RGB crops)" \
  --source "huggingface: ericyu/OSCD_Cropped_256" --out-file oscd_results.json
```
Test parquets from HF mirrors `ericyu/LEVIRCD_Cropped_256` and `ericyu/OSCD_Cropped_256`.
Requires the Phase 9 SECOND checkpoint at `models/phase9/second/second_best.pt`.

## 18. Regression Results
No production code changed in Phase 11 (only `scripts/phase11/` + `reports/phase11/` + `.gitignore`
added). Verified: backend pytest **447 passed**, ruff clean, `docker compose config` valid,
frontend `vite build` clean. Phase 9/10 baselines intact.

---
**PHASE 11 STATUS: COMPLETE** — every listed benchmark was either evaluated on real public data
(LEVIR-CD, OSCD) or documented as genuinely INCOMPATIBLE (RSVQA, VRSBench-VQA, VRSBench-caption)
or BLOCKED (VRSBench grounding), with no fabricated results, no test-set tuning, and all
production models preserved.
