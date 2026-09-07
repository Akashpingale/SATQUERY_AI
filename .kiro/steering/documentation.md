---
inclusion: always
---

# Documentation Rules

Documentation must reflect the **actual** implementation. Never document a feature as
complete when it is not.

## Maintain
- `README.md` (overview, setup, run).
- `docs/architecture.md`, `docs/setup.md`, `docs/user-guide.md`, `docs/api.md`.
- `docs/training.md`, `docs/datasets.md`, `docs/models.md` (provenance + licensing).
- `docs/evaluation.md`, `docs/security.md`, `docs/troubleshooting.md`, `docs/demo.md`,
  `docs/limitations.md`.
- `docs/traceability.md` — the SIH requirement -> implementation -> test -> UI -> demo matrix.

## Rules
- Update docs in the same change as the behavior they describe.
- State real status. If a capability is partial or not yet evaluated, say so explicitly
  (e.g. `NOT EVALUATED — dependency unavailable`). No fabricated results or metrics.
- Record dataset/model provenance and licenses; never claim data is human-verified if
  it is not.
- Keep setup instructions runnable on a clean machine (verified via Docker).
- Document limitations honestly — important for hidden ISRO/SAC generalization.
