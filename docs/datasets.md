# Datasets — provenance, licensing, and preparation

Every dataset actually used by SatQuery AI is recorded here with its real source, real
measured size, and real counts. Numbers in this file come from the reports the pipeline
writes, not from dataset documentation:

| Number | Source of truth |
|---|---|
| Schema, row counts, distributions | `scripts/bigearthnet_txt_report.txt` |
| File sizes, disk space | `scripts/bigearthnet_sources_report.txt` |
| Retained/excluded counts, hashes | `data/manifests/bigearthnet_txt_manifest.json` |
| Caption token lengths | `scripts/caption_token_report.txt` |

Regenerate them with `scripts/inspect_bigearthnet_txt.py`,
`scripts/inspect_bigearthnet_sources.py`, `scripts/prepare_bigearthnet_txt.py`, and
`scripts/measure_caption_tokens.py`.

---

## BigEarthNet.txt — Phase 5 adaptation data

SIH26167 names BigEarthNet as a required data route for remote-sensing VLM adaptation.
Phase 5 uses **BigEarthNet.txt**, the vision-language annotation release built on
BigEarthNet v2.0 (reBEN).

Phase 5 needs two different things, and they come from two different places. This split is
the single most important provenance fact in this document.

| Part | Source | Official? |
|---|---|---|
| Image–text annotations (captions) | `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` | **Yes** |
| Sentinel-1/2 pixels | `hackelle/BigEarthNetV2-Lithuania-Summer-LMDB` | **No — unofficial mirror** |

### Annotations (official)

- Repository: [`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)
- File: `BigEarthNet.txt.parquet`, **445.2 MB** (466,819,745 bytes)
- SHA256: `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`
- License: **CDLA-Permissive-1.0**. Ungated; no token required.
- **9,553,962** rows × 13 columns, 78 row groups.

Columns: `ID, s1_name, patch_id, input, output, type, category, split, latitude, longitude,
country, season, climate_zone`.

Annotation types across the whole file:

| `type` | Rows |
|---|---|
| binary | 3,625,160 |
| mcq | 3,259,184 |
| bounding box | 2,205,686 |
| **captioning** | **463,932** |

Only the `captioning` rows are used (see *Why captioning rows only* below).

### Imagery (unofficial mirror) — read this before trusting results

> **Provenance warning.** The pixels were **not** downloaded from the official release. They
> come from `hackelle/BigEarthNetV2-Lithuania-Summer-LMDB`, which its own publisher declares
> to be an **unofficial pre-conversion** of the official BigEarthNet v2.0 release
> ([10.5281/zenodo.10891137](https://zenodo.org/records/10891137)), produced with `rico-hdl`
> — the converter the official project recommends. It is CDLA-Permissive-1.0 and ungated,
> but it is a third-party artifact and is treated as one.

- LMDB: `BENv2_lithuania_summer.lmdb/data.mdb`, **2,366.9 MB** (2,481,856,512 bytes)
- SHA256: `54c0d0cdf7d674528c2f436bb2f215e682b2f7d985ae323eb4b56311c8de95e4`
- 17,550 entries (Sentinel-2 and Sentinel-1 patches, keyed by patch id)
- Sentinel-2 bands stored: `B01–B09, B11, B12, B8A` as `uint16`; Sentinel-1: `VH, VV`
- Sidecar: `metadata_lithuania_summer.parquet` (8,775 patches), 0.2 MB
- Total downloaded: **2.75 GB**

**Why the official imagery was not used.** The official Zenodo release was measured, not
guessed: `BigEarthNet-S2.tar.zst` **58.91 GB** + `BigEarthNet-S1.tar.zst` **50.70 GB** +
reference maps 0.26 GB = **109.88 GB compressed**, roughly 170–220 GB once extracted.
Measured free space at the time: **C: 13.54 GB**, **D: 258.17 GB**. Downloading and
extracting the full release was not feasible on this machine within the phase, so a 2.31 GB
country/season subset was used instead. That is a real constraint, honestly recorded, not a
shortcut taken for convenience.

**How the risk is mitigated.** Trusting a third-party mirror blindly would be unacceptable,
so every patch is cross-checked against the **official** annotation file:

- Lithuania/Summer patches present in the official annotations: **8,363**
- Annotation patch ids missing from the mirror: **0**
- Mirror patch ids absent from the official annotations: 412 (unused)
- Mirror coverage explained by official annotations: **95.30 %**

Zero annotation-side mismatches means every trained sample's caption is officially sourced
and its image key is confirmed present. Pixel *values* are still only as trustworthy as the
converter; that residual risk is stated rather than dismissed.

### Filters and the prepared subset

Filters: `country = Lithuania`, `season = Summer`, `type = captioning` — the Lithuania/Summer
slice is what the imagery mirror covers.

Measured funnel (`scripts/bigearthnet_prepare_report.txt`):

| Stage | Count |
|---|---|
| Annotation rows scanned | 9,553,962 |
| Matched country + season | 185,443 |
| Of those, captioning rows | 8,362 |
| **Retained for Phase 5** | **8,362** |
| Excluded — image key absent from LMDB | 0 |
| Excluded — image did not decode | 0 |
| Excluded — empty caption | 0 |
| Excluded — duplicate (patch, caption) | 0 |
| Unique patches | 8,362 |

Nothing was dropped. The exclusion counters are real code paths that recorded zero, not
absent checks — `prepare_caption_dataset` verifies every patch id against the LMDB, decodes
a deterministic 64-patch probe, and records a reason for anything it removes.

### Splits — official, not custom

| Split | Samples | Used for |
|---|---|---|
| `train` | 4,008 | LoRA training |
| `validation` | 2,291 | Per-epoch validation loss |
| `test` | 2,053 | **Held-out** baseline-vs-adapted evaluation |
| `bench` | 10 | Reserved for qualitative checks |

Splits come from the dataset's own `split` column. No custom splitting was performed, so
there is no chance of a patch leaking between train and test through a re-shuffle. The test
split was never used for training, validation, model selection, or hyperparameter choice.

### Preprocessing

reBEN patches are 120×120 at 10 m (~1.2 km across). Preparation
(`ml/datasets/bigearthnet.py`):

1. Read `B04` (red), `B03` (green), `B02` (blue) — the three 10 m bands, so no inter-band
   resampling is needed.
2. Apply a **joint 2–98 percentile stretch across the three bands** to `uint8`, preserving
   relative colour balance.
3. Resize 120×120 → 224×224 bicubic for CLIP.

**Why a percentile stretch and not a fixed reflectance ceiling.** A fixed ceiling was tried
first and rejected on measurement. Raw `uint16` RGB values in this subset are low:
median **277**, 99th percentile **528**, max **1148**. Dividing by a conventional ceiling of
3000 produced near-black images (mean pixel **24/255**), which would have crippled training.
The per-image stretch is also exactly what `geospatial.render` applies at inference time, so
training and serving see the same kind of image — a fixed dataset-specific constant would
have introduced a train/serve mismatch. Recorded in the manifest under `imagery.stretch`.

Preprocessing version: `phase5-rgb-v1`. Seed: `42`.

### Caption truncation — a hard, measured ceiling

CLIP's text encoder has a fixed **77-token** context (75 content tokens plus BOS/EOS).
BigEarthNet.txt captions are long structured descriptions, so most of them do not fit.
Measured with the real tokenizer over all 8,362 prepared samples:

| Split | n | Median tokens | Mean | Max | Over 77 tokens | Mean % of caption kept |
|---|---|---|---|---|---|---|
| train | 4,008 | 190 | 178.7 | 445 | 80.6 % | 52.2 % |
| validation | 2,291 | 202 | 185.1 | 425 | 81.4 % | 50.4 % |
| test | 2,053 | 215 | 211.3 | 449 | 93.3 % | 41.5 % |
| bench | 10 | 248 | 214.3 | 386 | 90.0 % | 45.1 % |
| **all** | **8,362** | **200** | **188.5** | **449** | **84.0 %** | **49.1 %** |

So **84 % of captions are truncated, and on average less than half of each caption reaches
the model.** What survives is the opening scene summary and the dominant land cover — the
part that describes the image as a whole. What is discarded is per-parcel area figures and
adjacency detail.

This matters when reading the Phase 5 results in `docs/evaluation.md`. The adapted model
improved substantially over the baseline, but it was trained against roughly half of each
caption, and every caption begins with near-identical boilerplate (*"This satellite image,
captured during the summer in Lithuania, showcases…"*). Both facts limit how much
discriminative text signal exists, and they help explain why absolute retrieval numbers stay
low even after a 6.9× improvement. Lifting this ceiling needs either caption segmentation or
a text encoder with a longer context — neither was in Phase 5's scope.

### Why captioning rows only

BigEarthNet.txt also contains 3.6 M binary, 3.3 M multiple-choice, and 2.2 M bounding-box
rows. They were deliberately not used. The Phase 5 objective is symmetric contrastive
(InfoNCE) training of a **dual encoder**, which needs matched image–description pairs.
Feeding it `"yes"`/`"no"` answers or coordinate strings would train the text tower on
targets that carry no scene description, which is the wrong objective for this architecture.
Those row types are the natural data for a generative VQA head — a later phase, if one is
added.

### Sentinel-1 is present but was not trained on

The LMDB contains Sentinel-1 `VV`/`VH` for every patch, and the manifest records its
availability (`imagery.sentinel1_used_for_training: false`). Phase 5 trained on
**Sentinel-2 RGB only**. No optical–SAR fusion was learned here — that is Phase 7. Any claim
that Phase 5 produced a fusion-capable model would be false.

### Licensing summary

| Artifact | License | Notes |
|---|---|---|
| BigEarthNet.txt annotations | CDLA-Permissive-1.0 | Official, ungated |
| BigEarthNet v2.0 (reBEN) imagery | CDLA-Permissive-1.0 | Official release on Zenodo |
| Lithuania/Summer LMDB mirror | CDLA-Permissive-1.0 | Unofficial third-party conversion |
| Underlying Sentinel-1/2 data | Copernicus Sentinel data terms | Attribution to ESA/Copernicus |

CDLA-Permissive-1.0 permits use and redistribution of derived results. Attribution to the
BigEarthNet authors and to the Copernicus programme is required. The **adapter** trained
from this data is a derived work of both the dataset and the base model — see
`docs/models.md` for the base-model licensing caveat, which is the weaker link.

### Human verification

**None of these annotations are human-verified.** BigEarthNet.txt captions are generated
from CORINE Land Cover reference labels and patch statistics. The land-cover labels
themselves derive from CORINE, not from per-patch human inspection. So the adapted model
learned to align imagery with *automatically generated descriptions of automatically derived
labels*. This is legitimate supervision and it is what the dataset provides, but it must not
be described as human-labelled ground truth.

### What this data does *not* cover

The hidden ISRO/SAC evaluation set will not look like this. Honest scope of the Phase 5
training data:

- **One country** (Lithuania) and **one season** (Summer) — no seasonal or geographic variety.
- **One sensor family** at one resolution (Sentinel-2, 10 m), one patch size (~1.2 km).
- **European CORINE land-cover semantics** — no Indian land-cover classes, no urban
  morphology outside Europe, no arid or tropical regions.
- **No SAR training**, no temporal pairs, no object-level annotation used.

Adaptation on this subset should be expected to improve remote-sensing land-cover language
grounding in general, while being unable to guarantee transfer to unseen geographies. That
expectation is why the adapted registry entries remain `degraded` rather than `available`.

### Reproduce

All commands need the weight/dataset cache pointed at a drive with space:

```powershell
$env:HF_HOME="D:\SatQuery-AI\models\hf-cache"

# 1. Measure real source sizes and free disk before downloading anything
.\.venv\Scripts\python.exe scripts\inspect_bigearthnet_sources.py

# 2. Download annotations + the Lithuania/Summer imagery subset (~2.75 GB)
.\.venv\Scripts\python.exe scripts\download_bigearthnet.py

# 3. Inspect real schema, distributions, and image/text pairing
.\.venv\Scripts\python.exe scripts\inspect_bigearthnet_txt.py

# 4. Build the prepared splits + manifest with SHA256 hashes
.\.venv\Scripts\python.exe scripts\prepare_bigearthnet_txt.py

# 5. Measure caption truncation against CLIP's 77-token context
.\.venv\Scripts\python.exe scripts\measure_caption_tokens.py
```

Outputs: `data/bigearthnet/prepared/{train,validation,test,bench}.jsonl` and
`data/manifests/bigearthnet_txt_manifest.json`. The downloaded data and the prepared splits
are gitignored (2.8 GB, fully re-derivable); the manifest is committed so any claim here can
be checked against the hashes.

---

## Other datasets

| Dataset | Purpose | Status |
|---|---|---|
| RSICD | Training data for the base model `clip-rsicd-v2` | Used upstream by the base model, **not** downloaded or trained on here |
| Benchmark sets (RSVQA, LEVIR-CC, …) | Phase 9 benchmark evaluation | **NOT DOWNLOADED** — Phase 9 |
| Optical–SAR pairs | Phase 7 fusion | **NOT DOWNLOADED** — Phase 7 |
| Hidden ISRO/SAC set | Final evaluation | Not available to us, and by rule **never** used for training |

Test fixtures in `backend/tests/raster_fixtures.py` are synthetic rasters generated in code.
They verify pipeline behaviour, not model accuracy: no accuracy claim is ever made from them.
