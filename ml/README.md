# ml/ — Machine learning (datasets, training, inference, evaluation)

This package is intentionally **separable from the serving app** so GPU-heavy
work can run on a cloud GPU (compute Path A) without changing the backend.

```text
ml/
├── datasets/     Dataset acquisition, validation, preprocessing, splits (e.g. BigEarthNet)
├── training/     LoRA/QLoRA adaptation of a remote-sensing VLM; checkpoint + metadata
├── inference/    Uniform specialist interfaces (VQA, captioning, grounding, temporal, fusion)
├── evaluation/   Real benchmark evaluation (no fabricated metrics)
└── configs/      Reproducible training/eval configs (model, dataset, seed, hyperparams)
```

Populated in **Phase 4 (single-image inference)** and **Phase 5 (RS VLM adaptation)**.
Datasets and checkpoints are **not** committed to Git. See `docs/datasets.md` and
`docs/models.md` (created when those phases begin) for provenance and licensing.
