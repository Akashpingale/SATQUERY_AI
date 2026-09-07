# Phase 9 Data-Leakage Protection

Measured checks that training never saw validation/test data.

## CDVQA
- Split by **image**: Train (1,600 images) / Val (400) / Test (968) are **disjoint**. Measured
  overlaps: Train∩Val = 0, Train∩Test = 0, Val∩Test = 0 (asserted in `scripts/train_cd_vqa.py`
  at run start and in `backend/tests/test_cdvqa.py::test_no_image_leakage_between_train_val_test`).
- **Test2** deliberately reuses Test's 968 images with a different (harder) question set — this
  is the CDVQA design and is documented, not hidden. Train∩Test2 = 0.
- Vocabularies (answers, words) are built from **Train only**. Test answers outside the closed
  train vocabulary are counted as incorrect, never added.
- Model selection used **Val** accuracy only; Test/Test2 were evaluated once, after freezing.
- Calibration temperature was fit on **Val**; ECE reported on Test. Test labels were never used
  for fitting or selection.

## SECOND
- The provided archive is the train set only; splits (train 2,077 / val 445 / test 446) are a
  deterministic seed-42 partition, **disjoint by construction** (`make_splits`), with the exact
  sample ids saved to `reports/benchmarks/second/splits.json` for audit.
- The held-out split is an **internal** test split (not the official SECOND test set) — stated
  everywhere it is reported.
- Best checkpoint selected on **val IoU**; the internal test split evaluated once. The
  deterministic baseline used no test-derived tuning (fixed Otsu per image).

## Duplicate handling
CDVQA image records repeat file_names within a split (each image carries multiple questions);
this is expected and does not cross split boundaries. No test image appears in train for either
dataset. No duplicates were silently deleted.

## Not used
Hidden ISRO/SAC evaluation data was never downloaded, extracted, or used for training,
selection, or calibration.
