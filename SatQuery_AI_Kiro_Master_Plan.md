# SatQuery AI — Master Engineering & Kiro Implementation Plan

## 0. Purpose

This is the **single source of truth** for building the complete SatQuery AI solution for:

- **SIH26167 — SatQuery AI: An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**
- Organization: ISRO / Department of Space
- Category: Software
- Theme: Space Technology

Kiro must read this entire document before implementation and treat it as the master engineering specification.

**Critical rule:** Do not build only a frontend prototype. Every mandatory capability must be implemented as a real, executable, tested, integrated feature.

If Kiro needs a credential, dataset access, model choice, cloud resource, hardware decision, private evaluation data, deployment decision, or any other input that cannot safely be inferred, **Kiro MUST ask the user instead of guessing**.

---

# 1. Kiro Operating Rules

1. Inspect the repository before making architectural changes.
2. Read this master plan before coding.
3. Create a structured implementation plan/specification before large-scale coding.
4. Work phase-by-phase.
5. Keep the application runnable after meaningful phases.
6. Prefer real implementations over placeholders.
7. Never replace mandatory remote-sensing AI with a generic chatbot response.
8. Never claim a feature is complete until it is implemented, integrated, and tested.
9. Never invent credentials, benchmark results, model availability, dataset access, or user decisions.
10. Never hard-code secrets.
11. Add tests with every major capability.
12. Maintain a requirement traceability matrix.
13. Do not expose chain-of-thought. Only expose a concise observable execution trace.
14. Preserve useful existing work.
15. If a decision materially affects correctness and cannot be inferred, stop and ask the user.
16. If a normal engineering choice can safely be made, make it, document it, and continue.
17. At the end of every phase report implemented, tested, remaining, blockers, user input required, and next phase.

---

# 2. Things the User Must Have Before Starting Kiro

## Required

- Kiro Pro installed and authenticated.
- Git installed.
- Python 3.11+.
- Node.js 20+ and npm.
- Docker Desktop installed and running.
- GitHub repository recommended.
- At least 20–50 GB free disk space recommended.
- Internet access for packages, datasets, and open-source model downloads.

Verify with:

```powershell
python --version
node --version
npm --version
git --version
docker --version
docker compose version
```

## Strongly recommended

- NVIDIA GPU with CUDA.
- 12 GB+ VRAM; 16–24 GB+ is better for VLM work.
- If no GPU is available, Kiro must support CPU development and keep training separable so cloud GPU can be used later.

## Not required before the first Kiro session

The user does NOT need to already have:

- trained VLM
- LoRA adapter
- VQA model
- grounding model
- captioning model
- change model
- optical-SAR fusion model
- all datasets
- PostgreSQL database
- finished UI
- Docker configuration
- final model registry

Kiro should build these progressively.

---

# 3. User Input Required Protocol

Whenever Kiro needs something from the user, it MUST ask instead of guessing.

Examples:

- Hugging Face token for a gated model.
- OpenAI/Anthropic API key for an optional external model.
- AWS/Bedrock credentials.
- Dataset registration/download approval.
- User choice of local GPU vs cloud GPU.
- Private SIH/ISRO evaluation data.
- Deployment target.
- Cloud storage credentials.
- Domain/HTTPS information.

Use this format:

> USER INPUT REQUIRED  
> What is needed: ...  
> Why: ...  
> Where/how to obtain it: ...  
> Expected value/format: ...  
> What will happen after it is provided: ...

For secrets, ask the user to put them into `.env` or a secret manager. Do not ask them to commit or hard-code credentials.

Kiro must continue all independent work while waiting for a blocking dependency.

---

# 4. Problem Understanding

SatQuery AI is an interactive **agentic vision-language assistant** for remote-sensing image analysis through natural-language queries.

It must accept:

1. Single optical/multispectral imagery.
2. Single SAR imagery.
3. Co-registered optical + SAR pairs.
4. Bi-temporal imagery of the same area.
5. GeoTIFF/TIFF.
6. PNG/JPEG where allowed for public/benchmark datasets.

The user asks a natural-language question. The system interprets the query, validates the input configuration, selects specialist models/tools, executes them, combines evidence, estimates confidence, and returns an answer plus an auditable execution summary.

---

# 5. Mandatory SIH Requirement Matrix

| Requirement | Priority |
|---|---|
| Natural-language queries | MANDATORY |
| Interactive GUI/web app | MANDATORY |
| Optical/multispectral support | MANDATORY |
| SAR support | MANDATORY |
| GeoTIFF/TIFF | MANDATORY |
| Input/metadata validation | MANDATORY |
| Single-image VQA | MANDATORY |
| At least one additional single-image capability | MANDATORY |
| Remote-sensing VLM fine-tuning/adaptation | MANDATORY |
| BigEarthNet.txt or open-source remote-sensing training data | MANDATORY route |
| Bi-temporal analysis | MANDATORY |
| Change description OR change-based VQA | MANDATORY |
| Optical + SAR joint analysis | MANDATORY |
| Agentic orchestration | MANDATORY |
| Automatic task/model/tool selection | MANDATORY |
| Workflow sequencing | MANDATORY |
| Controlled parameters | REQUIRED |
| Text/spatial output combination | REQUIRED |
| Visual evidence | REQUIRED |
| Confidence | REQUIRED |
| Observable execution summary | REQUIRED |
| Benchmark evaluation | REQUIRED for serious validation |
| Hidden ISRO/SAC generalization readiness | CRITICAL |
| Change map | OPTIONAL / HIGH-VALUE |
| Downloadable report | HIGH-VALUE |
| Docker clean-machine run | HIGH-VALUE |
| Automated tests | REQUIRED ENGINEERING QUALITY |
| Accessibility | REQUIRED ENGINEERING QUALITY |

---

# 6. Product Vision

The product must feel like a professional remote-sensing analysis assistant, not a generic chatbot.

Example:

User uploads optical and SAR imagery and asks:

> Use the optical and SAR images together to identify built-up and water-covered regions.

The system should:

1. Inspect inputs.
2. Detect modalities.
3. Validate compatibility.
4. Classify the task.
5. Select the appropriate tools/models.
6. Execute preprocessing.
7. Perform genuine optical-SAR analysis.
8. Produce the answer.
9. Show visual evidence.
10. Report confidence.
11. Show a concise execution trace.
12. Allow the user to inspect/download the result.

---

# 7. User-Friendly Frontend

A polished, easy-to-use frontend is a core product requirement.

## Recommended stack

- React
- TypeScript
- Vite
- Tailwind CSS
- shadcn/ui
- React Router
- TanStack Query
- React Hook Form
- Zod
- Lucide icons
- Vitest
- React Testing Library
- Playwright

Use reusable components and a consistent design system.

## Main layout

### Header

- SatQuery AI branding
- system status
- help/about
- settings

### Input panel

- drag-and-drop upload
- optical upload
- SAR upload
- T1 upload
- T2 upload
- supported-format indicators
- validation status
- metadata preview
- replace/remove controls

### Main visualization

- large image viewer
- zoom
- pan
- fit-to-screen
- layer controls
- optical/SAR switching
- T1/T2 comparison
- bounding boxes
- masks
- evidence highlights
- change overlay
- legend
- coordinates when available

### Assistant panel

- natural-language query box
- example questions
- Analyze button
- answer
- confidence
- evidence
- warnings
- selected task
- selected model/tool
- execution summary

### History

- previous analyses
- timestamps
- task
- status
- reopen result

## UX requirements

- Responsive.
- Clean and modern.
- Beginner-friendly.
- Clear empty/loading/error states.
- No raw stack traces in normal UI.
- Explain unsupported formats.
- Explain incompatible image pairs.
- Expandable advanced metadata.
- Keyboard accessible.
- Screen-reader-friendly.
- Good contrast.
- Do not rely on color alone.
- Avoid unnecessary animations.
- Keep important evidence visible.

---

# 8. Frontend Screens

Implement:

1. Home/dashboard.
2. Analysis workspace.
3. Upload/input area.
4. Results/evidence view.
5. Execution trace.
6. History.
7. Demo examples.
8. Settings.
9. Help/about.
10. Error/fallback states.

---

# 9. Backend Stack

Recommended:

- Python 3.11+
- FastAPI
- Pydantic
- Uvicorn
- SQLAlchemy
- Alembic
- PostgreSQL
- psycopg
- HTTPX
- structured logging

Optional for expensive asynchronous jobs:

- Redis
- Celery/RQ/Arq or equivalent.

---

# 10. Geospatial Stack

Use:

- Rasterio
- GDAL
- NumPy
- OpenCV
- scikit-image
- pyproj
- Shapely
- GeoPandas

Support:

- GeoTIFF metadata.
- CRS.
- bounds.
- transform.
- resolution.
- band information.
- windowed reads.
- tiling/chunking.
- resampling.
- spatial alignment.
- coordinate conversion.
- large-raster processing.

Never load a huge GeoTIFF fully into RAM unnecessarily.

---

# 11. ML Stack

Recommended:

- PyTorch
- Transformers
- Hugging Face Datasets
- PEFT
- LoRA / QLoRA
- Accelerate
- safetensors
- scikit-learn
- SciPy

Do not lock a specific VLM before checking:

1. Hardware.
2. VRAM.
3. License.
4. Remote-sensing suitability.
5. Inference feasibility.
6. Training/adaptation feasibility.

The architecture must allow model replacement.

---

# 12. Remote-Sensing VLM Adaptation

This is a critical SIH requirement.

A generic VLM plus a remote-sensing prompt does NOT count as genuine adaptation.

Implement:

- open-source VLM selection
- remote-sensing training data
- preprocessing
- fine-tuning/adaptation
- LoRA/QLoRA where appropriate
- checkpoint saving
- model metadata
- evaluation
- production integration

Track:

- base model
- model version
- dataset/version
- preprocessing
- configuration
- hyperparameters
- seed
- adapter/checkpoint
- hardware
- evaluation metrics

The adapted model must actually be connected to the intended production workflow.

---

# 13. BigEarthNet.txt

Support BigEarthNet.txt as a major training/adaptation source.

Implement:

- acquisition instructions
- version/configuration
- validation
- preprocessing
- train/validation/test handling
- task conversion
- training configuration
- LoRA/QLoRA
- checkpoint saving
- evaluation
- reproducibility
- licensing/provenance documentation

Do not commit the dataset to Git.

If manual registration/download is needed, Kiro must ask the user at that point.

---

# 14. Other Dataset Support

Evaluate where appropriate:

- VRSBench
- RSVQA
- CDVQA
- BigEarthNet.txt
- other compatible open-source remote-sensing datasets

Create a dataset registry documenting:

- name
- source
- license
- task
- modalities
- format
- preprocessing
- split
- intended use
- limitations
- annotation quality

Never claim a dataset is human verified if it is not.

---

# 15. Model/Tool Registry

Create a central registry.

Each entry should contain:

- name
- version
- task
- modalities
- input contract
- output contract
- allowed parameters
- device requirements
- checkpoint
- license
- health status

Logical tools:

```text
VQA
Captioning
Grounding
TemporalChange
ChangeVQA
OpticalSARFusion
RasterPreprocessor
RegistrationValidator
EvidenceAggregator
ConfidenceCalibrator
```

The agent may only select from this registry.

---

# 16. Agentic Orchestration

Implement a real orchestration layer.

```text
User Query
    ↓
Input Inspection
    ↓
Query Understanding
    ↓
Task Classification
    ↓
Compatibility Validation
    ↓
Model/Tool Selection
    ↓
Workflow Planning
    ↓
Specialist Execution
    ↓
Evidence Extraction
    ↓
Confidence
    ↓
Answer Synthesis
    ↓
Execution Trace
    ↓
UI
```

Task types should include:

- VQA
- captioning
- scene description
- grounding
- temporal change
- change VQA
- optical-SAR fusion
- metadata inspection
- unsupported task

Do not implement the "agent" as a hard-coded demo switch.

---

# 17. Agent Safety and Parameter Control

The agent must not invent arbitrary parameters.

Use allowlisted schemas.

Validate:

- task
- model
- version
- input IDs
- modality
- image count
- dimensions
- CRS
- parameters
- timeout
- memory constraints

Reject invalid plans.

---

# 18. Input Validation

## Files

Validate:

- extension
- MIME
- size
- corruption
- raster readability
- supported format

## Remote sensing

Inspect:

- dimensions
- band count
- data type
- CRS
- transform
- resolution
- bounds
- metadata

## Optical + SAR

Check:

- overlap
- CRS
- extent
- resolution
- alignment
- resampling needs

## Temporal

Check:

- geographic compatibility
- CRS
- extent
- alignment
- resolution
- temporal metadata where available

Do not run unreliable change analysis on clearly incompatible images.

---

# 19. Single-Image VQA

Mandatory.

Examples:

- What land cover dominates?
- Is water present?
- Are buildings present?
- What major objects are visible?

Return:

- answer
- confidence
- evidence where available
- model/tool information
- execution trace

---

# 20. Additional Single-Image Capability

Implement at least one, preferably both:

### Grounding

Example:

> Highlight the water body in this image.

Return:

- bounding box and/or mask
- overlay
- confidence

### Captioning

Generate a remote-sensing-aware scene description.

---

# 21. Temporal Change

Mandatory.

Inputs:

- T1
- T2

Process:

1. Validate.
2. Align/register.
3. Resample/normalize where appropriate.
4. Detect/understand changes.
5. Generate change description and/or change VQA.
6. Generate evidence.
7. Estimate confidence.

Examples:

> What changed between these two dates?

> Has the built-up area increased, decreased, or remained unchanged?

---

# 22. Change Map

Optional but strongly recommended.

When reliable detection/masks are available:

- generate map
- overlay on imagery
- legend
- area/percentage when geospatially valid
- distinguish detection from interpretation
- never present uncertain output as ground truth

---

# 23. Optical + SAR Analysis

Mandatory.

The system must use both modalities together.

It must demonstrate complementary information, not merely:

```text
Optical result + SAR result
```

and call that fusion.

Examples:

- Optical contributes spectral/contextual information.
- SAR contributes structural/backscatter information.
- Fusion combines them into a joint result.

Implement a dedicated fusion interface.

---

# 24. Fusion Architecture

Possible approaches:

- feature-level fusion
- cross-attention
- embedding fusion
- learned/calibrated late fusion
- multimodal VLM input

Choose after hardware/model evaluation.

Document the selected architecture and why it is genuine multimodal fusion.

---

# 25. Evidence System

Provide machine-readable evidence internally and visual evidence in the UI.

Possible evidence:

- region
- bounding box
- mask
- highlighted pixels
- T1/T2 comparison
- optical evidence
- SAR evidence
- metadata
- specialist outputs

---

# 26. Confidence

Implement a meaningful confidence layer.

Potential inputs:

- model confidence
- input compatibility
- evidence agreement
- modality agreement
- calibration data

Use clear levels:

- High
- Medium
- Low
- Unable to determine

Never invent confidence values.

---

# 27. Execution Trace

Show a concise auditable trace, for example:

```text
Task: Temporal Change VQA
Inputs: Optical T1 + Optical T2
Validation: Passed
Alignment: Passed
Selected Tool: TemporalChangeModel
Model: <version>
Parameters: <safe summary>
Evidence: Change region detected
Confidence: 0.84
Result: Built-up area increased
```

Do not expose private chain-of-thought.

---

# 28. API

Use versioned APIs.

Suggested:

```text
GET  /api/v1/health
GET  /api/v1/models
GET  /api/v1/tasks
POST /api/v1/uploads
POST /api/v1/analysis
GET  /api/v1/analysis/{id}
GET  /api/v1/analysis/{id}/trace
GET  /api/v1/analysis/{id}/evidence
GET  /api/v1/analysis/{id}/report
GET  /api/v1/history
```

Use Pydantic schemas and OpenAPI documentation.

---

# 29. Database

Use PostgreSQL for metadata.

Store:

- analysis ID
- timestamps
- input metadata
- task
- selected model/tool
- version
- status
- confidence
- evidence references
- execution trace
- error information
- report reference

Keep huge raster binaries in filesystem/object storage rather than PostgreSQL unless deliberately justified.

---

# 30. Storage

Separate:

```text
uploads/
processed/
models/
evidence/
reports/
temp/
```

Use safe unique IDs and filenames.

Clean temporary artifacts.

---

# 31. Performance

Implement:

- windowed raster reads
- tiling
- caching
- preprocessing cache
- asynchronous expensive jobs
- model reuse
- GPU memory management
- timeouts
- cancellation where practical

---

# 32. Security

Implement:

- upload limits
- MIME + extension checks
- safe filenames
- path traversal protection
- isolated temporary files
- resource/time limits
- model input validation
- prompt injection resistance
- no secret logging
- sanitized errors

Treat uploads and user queries as untrusted input.

---

# 33. Prompt Injection Protection

The system must:

1. Separate user task from system/tool instructions.
2. Treat text extracted from images as untrusted.
3. Restrict available tools.
4. Validate tool parameters.
5. Never execute arbitrary commands from user input.
6. Keep secrets out of model context.
7. Log blocked unsafe tool attempts.

---

# 34. Reports

Generate a downloadable report containing:

- query
- input summary
- task
- model/tool
- concise trace
- answer
- confidence
- evidence
- warnings
- timestamps
- provenance

Never include hidden chain-of-thought.

---

# 35. Accessibility

Implement:

- semantic HTML
- keyboard navigation
- focus states
- accessible labels
- screen-reader support
- contrast
- meaningful alt text
- non-color-only indicators
- accessible errors

Before finalization, perform an accessibility review. If a human NVDA/screen-reader check is required, Kiro must ask the user to perform it and record the result.

---

# 36. Demo Mode

Provide curated demo cases.

1. Single-image VQA.
2. Grounding/captioning.
3. Temporal change.
4. Change VQA.
5. Optical + SAR.
6. Agent routing.

The demos must use the same real production workflow, not fake frontend-only responses.

---

# 37. Evaluation

Implement evaluation for applicable tasks:

- VQA accuracy
- caption quality
- grounding IoU/localization
- change detection quality
- change-answer accuracy
- multimodal fusion quality
- confidence calibration
- routing accuracy
- latency
- memory
- failure rate

Never fabricate results.

Unavailable evaluation:

```text
NOT EVALUATED — dependency unavailable
```

---

# 38. Hidden ISRO/SAC Readiness

Avoid demo-specific hacks.

Design for:

- unseen geographic areas
- unseen examples
- sensor variation
- resolution variation
- different modality combinations
- georeferenced data
- pre-registered/co-registered optical and SAR
- no hard-coded answers
- no hard-coded coordinates
- no demo-only routing
- configurable models
- reproducible evaluation

---

# 39. Testing

## Unit

- raster validation
- metadata
- CRS
- alignment
- query classification
- routing
- parameter validation
- evidence
- confidence
- API schemas

## Integration

- upload
- validation
- VQA
- grounding
- temporal
- optical-SAR
- model registry
- database
- evidence
- reports

## E2E

Use Playwright for:

- upload
- query
- analysis
- result
- evidence
- trace
- history
- report
- errors

## Negative

Test:

- invalid files
- corrupted TIFF
- unsupported format
- incompatible pairs
- mismatched CRS
- missing modality
- missing model
- GPU unavailable
- timeout
- invalid agent plan
- malicious filename
- prompt injection

---

# 40. Docker

Provide:

- Dockerfile(s)
- docker-compose.yml
- frontend
- backend
- PostgreSQL
- optional Redis/worker
- health checks
- environment configuration
- persistent volumes where needed

Document clean-machine setup.

---

# 41. CI/CD

GitHub Actions should cover:

- backend tests
- frontend tests
- linting
- type checking
- build
- Docker build
- smoke tests

Do not run expensive training on every CI build.

---

# 42. Code Quality

Backend:

- type hints
- clear modules
- Pydantic
- logging
- error handling

Frontend:

- strict TypeScript
- reusable components
- API abstraction
- validation
- accessibility

ML:

- configs
- reproducibility
- model interfaces
- checkpoint metadata
- seeds where practical

---

# 43. Configuration

Use:

```text
.env.example
config/
configs/
```

Never commit:

```text
.env
API keys
tokens
credentials
large datasets
private model secrets
```

---

# 44. Repository Structure

Recommended:

```text
satquery-ai/
├── .github/
├── .kiro/
│   ├── steering/
│   ├── specs/
│   └── hooks/
├── backend/
│   ├── app/
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   ├── tests/
│   └── package.json
├── ml/
│   ├── datasets/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── configs/
├── geospatial/
├── models/
├── data/
│   ├── raw/
│   ├── processed/
│   └── demo/
├── reports/
├── scripts/
├── docs/
├── docker/
├── docker-compose.yml
├── .env.example
├── README.md
└── LICENSE
```

---

# 45. Kiro Steering Rules

Create:

```text
.kiro/steering/product.md
.kiro/steering/architecture.md
.kiro/steering/coding-standards.md
.kiro/steering/ai-ml-rules.md
.kiro/steering/frontend-ux.md
.kiro/steering/geospatial-rules.md
.kiro/steering/testing.md
.kiro/steering/security.md
.kiro/steering/documentation.md
```

These rules should enforce:

- SIH traceability
- real AI implementation
- user-friendly UI
- modular architecture
- safe agent behavior
- geospatial correctness
- testing
- documentation
- security
- reproducibility

Do not copy the entire master plan into every steering file.

---

# 46. Kiro Specs

Create specs for:

1. Foundation.
2. Frontend UX.
3. Raster ingestion.
4. Query understanding.
5. Agent orchestration.
6. Model registry.
7. Remote-sensing VLM adaptation.
8. VQA.
9. Captioning/grounding.
10. Temporal change.
11. Optical-SAR fusion.
12. Evidence/confidence.
13. API/database.
14. Evaluation.
15. Security.
16. Docker/deployment.
17. Final demo.

Each spec must contain:

- requirements
- design
- tasks
- acceptance criteria
- tests
- dependencies
- user inputs required

---

# 47. Development Sequence

## Phase 0 — Environment

Check:

- Python
- Node
- Git
- Docker
- Docker Compose
- GPU/CUDA
- disk
- repository

## Phase 1 — Foundation

Build:

- repository structure
- backend
- frontend
- database
- Docker
- config
- logging
- health checks

## Phase 2 — Raster Pipeline

Build:

- upload
- validation
- metadata
- GeoTIFF reading
- visualization
- tiling/windowing
- compatibility

## Phase 3 — Agent

Build:

- query classification
- task registry
- model registry
- routing
- parameter validation
- execution trace

## Phase 4 — Single Image

Build:

- VQA
- grounding and/or captioning
- evidence
- confidence

## Phase 5 — Remote-Sensing Adaptation

Build:

- dataset pipeline
- BigEarthNet.txt
- additional datasets
- training
- LoRA/QLoRA
- evaluation
- production checkpoint integration

## Phase 6 — Temporal

Build:

- T1/T2 validation
- alignment
- change understanding
- change description/change VQA
- evidence
- optional change map

## Phase 7 — Optical-SAR

Build:

- paired input
- compatibility
- genuine fusion
- complementary evidence

## Phase 8 — UX

Polish:

- responsive UI
- visualization
- history
- reports
- accessibility
- errors

## Phase 9 — Evaluation

Run:

- benchmark tests
- task metrics
- routing
- robustness
- performance

## Phase 10 — Finalization

Complete:

- Docker clean-machine test
- CI
- docs
- security
- demos
- judge workflow
- README
- final checklist

---

# 48. Definition of Done

## Functional

- [ ] User-friendly web application.
- [ ] Optical upload.
- [ ] SAR upload.
- [ ] GeoTIFF/TIFF.
- [ ] Metadata.
- [ ] Validation.
- [ ] Single-image VQA.
- [ ] Additional single-image task.
- [ ] Temporal analysis.
- [ ] Change description/change VQA.
- [ ] Optical-SAR joint analysis.
- [ ] Genuine remote-sensing VLM adaptation.
- [ ] Adapted model integrated.
- [ ] Automatic agent routing.
- [ ] Specialist model/tool selection.
- [ ] Evidence.
- [ ] Confidence.
- [ ] Execution trace.
- [ ] Report.

## Engineering

- [ ] Tests pass.
- [ ] Negative tests pass.
- [ ] Docker clean-machine test.
- [ ] CI passes.
- [ ] Secrets protected.
- [ ] Security review.
- [ ] Accessibility review.
- [ ] README complete.
- [ ] Reproducible setup.
- [ ] Model/dataset provenance.
- [ ] No fake metrics.
- [ ] No fake final AI responses.

## Judge demos

- [ ] VQA.
- [ ] Grounding/captioning.
- [ ] Temporal change.
- [ ] Change VQA.
- [ ] Optical-SAR.
- [ ] Agent routing.
- [ ] Evidence.
- [ ] Confidence.
- [ ] Execution trace.

---

# 49. First Kiro Session — EXACT BEHAVIOR

Kiro must NOT immediately generate the entire application.

First:

### Step 1 — Inspect environment

Check:

```text
Python
Node/npm
Git
Docker
Docker Compose
GPU/CUDA
disk
repository
```

### Step 2 — Inspect repository

Identify existing files and avoid unnecessary overwrites.

### Step 3 — Create Kiro configuration

Create:

```text
.kiro/steering/
.kiro/specs/
.kiro/hooks/
```

### Step 4 — Create steering rules

Apply the rules from this master plan.

### Step 5 — Create initial specs

Start with foundation and architecture.

### Step 6 — Create requirement traceability matrix

Every SIH requirement must map to:

```text
Requirement → Implementation → Test → UI → Demo evidence
```

### Step 7 — Identify blockers

Tell the user exactly what is needed.

### Step 8 — Ask user only for actual missing inputs

Examples:

```text
USER INPUT REQUIRED:
GPU unavailable — choose:
A) CPU development + cloud GPU later
B) provide cloud GPU
```

or:

```text
USER INPUT REQUIRED:
This model requires a Hugging Face token.
Please configure it in .env.
```

### Step 9 — Begin Phase 0/1

Do not wait for non-blocking dependencies.

### Step 10 — Report progress

At every phase:

```text
IMPLEMENTED
TESTED
REMAINING
BLOCKERS
USER INPUT REQUIRED
NEXT PHASE
```

---

# 50. Final Success Standard

A judge must be able to upload remote-sensing imagery, ask a natural-language question, and observe:

- query understanding
- input validation
- automatic task selection
- automatic model/tool selection
- genuine remote-sensing AI
- single-image VQA
- additional single-image analysis
- temporal change understanding
- optical-SAR joint analysis
- visual evidence
- confidence
- concise execution trace
- polished user-friendly UI

The final product must be a real end-to-end agentic remote-sensing AI system, not a chatbot demo or frontend mockup.

---

# 51. Master Checklist

## User prerequisites

- [ ] Kiro Pro
- [ ] Python 3.11+
- [ ] Node 20+
- [ ] npm
- [ ] Git
- [ ] Docker Desktop
- [ ] Docker Compose
- [ ] Internet
- [ ] Disk space
- [ ] GitHub repository
- [ ] GPU checked

## Product

- [ ] User-friendly UI
- [ ] Optical
- [ ] SAR
- [ ] Temporal
- [ ] GeoTIFF/TIFF
- [ ] Natural-language query

## AI

- [ ] VQA
- [ ] Grounding/captioning
- [ ] Remote-sensing adaptation
- [ ] BigEarthNet.txt
- [ ] Temporal change
- [ ] Change VQA/description
- [ ] Optical-SAR fusion
- [ ] Agent routing
- [ ] Model registry
- [ ] Evidence
- [ ] Confidence
- [ ] Execution trace

## Engineering

- [ ] React/TypeScript
- [ ] FastAPI
- [ ] PostgreSQL
- [ ] Rasterio/GDAL
- [ ] PyTorch/Transformers
- [ ] LoRA/QLoRA
- [ ] Docker
- [ ] CI
- [ ] Tests
- [ ] Security
- [ ] Accessibility
- [ ] Reports
- [ ] Documentation

## Validation

- [ ] VQA demo
- [ ] Grounding/caption demo
- [ ] Temporal demo
- [ ] Change VQA demo
- [ ] Optical-SAR demo
- [ ] Agent-routing demo
- [ ] Benchmark evaluation
- [ ] Hidden-set readiness
- [ ] Clean-machine Docker test
- [ ] Final judge walkthrough

# END OF MASTER PLAN
