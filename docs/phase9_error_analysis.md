# Phase 9 Error Analysis

Error analysis over the executed evaluations (synthetic fixtures + BigEarthNet retrieval) and the
documented incompatibilities. Real-benchmark error analysis (RSVQA/VRSBench/CDVQA/SECOND) is
pending dataset provisioning and is listed as remaining work — no errors are invented for
benchmarks that were not run.

## Change detection (synthetic, known masks)
- **Result:** IoU/F1 = 1.0, occurrence accuracy = 1.0 over 4 scenarios (incl. no-change).
- **Failure modes not exercised here:** these are clean, high-contrast synthetic blocks with
  perfect registration. Real datasets add: misregistration (the detector reports *inconclusive*
  rather than guessing — a safe failure), gradual/low-contrast change, seasonal radiometric
  differences misread as change (false positives), and semantic change the radiometric detector
  cannot classify. These are expected weaknesses documented against LEVIR-CD/OSCD/SECOND once
  provisioned.

## Change-occurrence VQA (synthetic)
- **Result:** 12/12 correct. All no-change pairs correctly answered "no"; all change pairs "yes".
- **Boundary risk:** a change just below `min_changed_fraction` would be answered "no" (a
  false negative); this is by design (avoids over-claiming) and is where confidence should drop.
  Our smallest block (~1.6% of scene) was still detected; a sub-threshold case would fail — to be
  probed on CDVQA.
- **Out of scope (correctly declined, not errors):** semantic-class change questions (building
  type, land-use category). These return a stated limitation.

## Cross-modal fusion (synthetic)
- **Passing:** SAR-only structured block raises SAR structure/`sar_only_information` above the
  flat-SAR baseline; water-agreement is available with real optical/SAR overlap (IoU 0.58).
- **Not passing (reported honestly):** the two ramp-based complementarity checks. Under a smooth
  left→right ramp, |correlation| is high for *both* the correlated and anticorrelated fixtures
  (≈0.96), so complementarity (= 1−|corr|) is low in both and the "correlated < anticorrelated"
  and "anticorrelated ≥ 0.5" checks fail. **Interpretation:** the metric behaves correctly
  (strong linear relationship ⇒ low complementarity regardless of sign); the *fixtures* were a
  poor probe for the intended property. This is a fixture-design limitation, not an algorithm
  bug, and is left visible rather than tuned to green. A better probe would use partially-shared
  structure with additive independent noise.

## Retrieval (BigEarthNet.txt)
- **Result:** adaptation improves mean recall ~6.9×, but absolute recall remains low
  (adapted i2t R@1 ≈ 3.5%). **Interpretation:** the adaptation clearly helps, but zero-shot RS
  retrieval on 2053 candidates is hard; this is honest evidence of adaptation benefit, not of
  strong absolute performance. It does not transfer to a VQA/caption accuracy claim.

## Confidence
- **Finding:** mildly under-confident on the synthetic set (accuracy 1.0 at mean confidence
  0.71; ECE 0.287). No incorrect high-confidence answers occurred on the fixture. Real
  mis-calibration (over-confident errors) can only be characterised on a real benchmark; the
  small synthetic N cannot establish it.

## Architectural incompatibilities (not "errors", but the reason some benchmarks can't score)
- VQA specialist emits fixed-vocabulary concept scores → cannot match RSVQA/VRSBench yes/no/count
  answers.
- Captioner is retrieval-composed → cannot produce generative-caption predictions.
Both require new trained components; documented in `docs/benchmarking.md` §4.
