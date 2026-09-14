# Drishti — Explainable DR Screening & District Triage

Prototype for **SIH 2026 · PS 26038 (MathWorks)** — *"Design a retinal image
analysis pipeline for diabetic retinopathy screening and triage, complete with
explainability and district-scale telemedicine simulation."*

> **The problem in one line:** 77M+ Indian adults live with diabetes and about
> one in three will develop diabetic retinopathy, but rural India has roughly
> one ophthalmologist per 100,000 people. The bottleneck is not diagnosis — it
> is *specialist attention*. This system is built to spend that attention well.

---

## What it does

An ASHA worker photographs a patient's retina on a low-cost fundus camera. The
system then:

1. **Gates image quality** and says exactly what to fix, while the patient is
   still in the chair. This runs first and short-circuits, so an unusable
   capture costs milliseconds instead of a GPU slot.
2. **Segments lesions** — microaneurysms, haemorrhages, hard exudates,
   cotton-wool spots — with sizes in microns and distance to the fovea.
3. **Grades severity** on the ICDR 0–4 scale, combining a deterministic
   clinical rule grader with an ordinal CNN.
4. **Explains itself** with a pixel-exact lesion overlay and a narrative that
   cites only measured numbers.
5. **Triages** into a referral window, escalating on evidence the grade alone
   does not capture (exudates at the fovea are sight-threatening at any grade).
6. **Routes uncertain cases to a human** and tracks them through a real
   ophthalmologist queue until sign-off.
7. **Simulates the district programme**, so value is measured in patients seen
   on time rather than in accuracy points.

## The core design decision

Most DR systems are a single CNN emitting a grade. That is a black box, and a
black box is what a rural clinician will not act on.

This pairs a **learned ordinal grader** with a **deterministic morphological
segmenter** that measures physical evidence, then **checks them against each
other**. Disagreement, low confidence, or a failed quality gate routes the case
to a human rather than issuing a confident answer. The system is built to be
able to say *"do not trust me on this one"* — which is what makes it
trustworthy on the rest.

Two further choices follow from the clinical setting:

**Ordinal, not 5-way softmax.** A softmax head treats grade 0 and grade 4 as
merely different, so confusing them costs what confusing 3 and 4 costs. The
CORAL head learns cumulative units ("is the grade > k?") sharing one weight
vector, which makes the predicted probabilities monotonic by construction and
yields a calibrated `P(grade ≥ 2)` — exactly the referral decision.

**Fusion escalates rather than averages.** A confident severe vote from any
grader carries forward. The system over-refers rather than missing disease, and
the district simulation is where that trade is priced — its false referrals are
counted against clinic capacity, not hidden.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
export DRISHTI_SECRET_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")
```

```bash
export DRISHTI_BOOTSTRAP_ADMIN_EMAIL=admin@district.gov.in DRISHTI_BOOTSTRAP_ADMIN_PASSWORD=change-me-now
```

```bash
uvicorn server.app:app --port 8000
```

Open <http://localhost:8000> and sign in with the bootstrap admin. Capture,
quality gating, records, the review queue and the district simulation all work
immediately; **grading returns a clear 503 until a model is trained**, because
this repo ships no weights and invents no data.

---

## Training on real data

The corpora are licence-restricted and are never committed. Point
`DR_DATA_ROOT` at them (on Kaggle they mount under `/kaggle/input`
automatically).

| Corpus | Size | Purpose |
|---|---|---|
| **IDRiD** | 516 graded, 81 with pixel masks | The only public set with per-lesion masks — the only way to score the segmenter's "where" channel |
| **APTOS 2019** | 3,662 | Indian population (Aravind); closest public proxy to the deployment setting |
| **EyePACS** | 88,702 | Scale, needed for a CNN that generalises |
| **Messidor-2** | 1,748 | Held out entirely as external validation; never trained on |

```bash
python -m dr.train --datasets aptos idrid --external messidor2 --size 512 --backbone tf_efficientnet_b3_ns
```

On Kaggle, open [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb):
it attaches the corpora, trains, prints the metrics, and verifies the exported
ONNX graph still agrees with the checkpoint that was validated.

Cache the resized images once first — full-resolution JPEG decoding, not the
GPU, is what makes an EyePACS epoch slow:

```bash
python -m dr.train --datasets eyepacs aptos --build-cache --cache-dir /kaggle/working/cache
```

Training writes `artifacts/grader.onnx` plus `grader.json` (thresholds,
backbone, input size). The server loads those; it never imports the training
code.

Score the segmenter against real lesion masks:

```bash
python -m dr.eval_lesions --data-root /path/to/datasets
```

### What the training code guards against

* **Patient-grouped splits.** EyePACS carries both eyes per patient and the two
  are highly correlated; a random image split validates partly on memorisation.
  `assert_no_patient_leakage` fails the run rather than reporting an inflated
  score.
* **Grade-stratified folds.** Grades 3–4 are a few percent of these corpora, so
  an unstratified fold can contain almost no severe disease and produce a
  meaningless sensitivity estimate.
* **QWK for model selection, not loss.** Loss keeps improving on the majority
  grade long after the clinically useful ranking has stopped.
* **Thresholds fitted on validation only**, saved with the weights, and forced
  apart by a minimum separation — an unconstrained fit collapses the cut-points
  into a cluster that maximises kappa on one split while making grades 1–3
  unreachable for every future patient.
* **Collapsed-model refusal.** Trained from scratch on a small cohort the CNN
  converges to predicting one class for everything. That is worse than no CNN:
  fusion drags every grade toward it while Grad-CAM produces convincing-looking
  saliency that means nothing. `train.py` refuses to save it. Use
  `--pretrained`.

---

## Deployment

```bash
cd deploy && cp .env.example .env    # fill in the secrets, then:
docker compose up -d
```

One image serves both roles, so an edge node cannot drift to a different
pipeline version than the district it syncs into.

| | District | PHC edge |
|---|---|---|
| Database | Postgres | SQLite, single file |
| Inference | GPU, batched | CPU, batch size 1 |
| Network | inbound | none required |

```bash
docker compose -f edge-compose.yml up -d      # at each PHC
```

```bash
python scripts/edge_sync.py --district https://district.example.in --email edge-7@svc.gov.in --password ...
```

The edge keeps screening while the link is down. `client_uuid` is the
idempotency key end to end, so a batch retried over a bad link is deduplicated
rather than creating a second clinical record, and records are marked synced
only after the district confirms.

---

## Architecture

```
 fundus image
      |
      v
 [core/quality.py]  focus / illumination / FOV / exposure  --fail--> recapture guidance
      |
      v
 [core/preprocess.py]  FOV crop, CLAHE, illumination flattening
      |                (records its crop transform so annotations can be replayed)
      v
 [core/lesions.py]  multi-scale top-hat, vessel- and disc-aware
      |             -> per-lesion class, area in um2, distance to fovea
      v
 [core/features.py]  23 clinically-named features
      |
      +--> [core/grade.rule_grade]  ICDR rules, no training
      +--> [dr/model.py CORAL head] ordinal CNN, P(grade >= 2)
      |
      v
 [core/grade.fuse]  average, then escalate to the most severe confident vote
      |
      v
 [core/explain.py] overlay + narrative      [core/triage.py] window + escalation
      |
      v
 [server/] records, review queue, audit     [sim/district.py] programme impact
```

| Path | Purpose |
|---|---|
| `core/` | Quality gate, segmentation, features, rule grader, explanation, triage |
| `dr/` | Datasets, splits, transforms, CORAL model, training, metrics, lesion eval |
| `server/` | API: auth, records, review queue, batched inference, audit, sync |
| `sim/` | District telemedicine simulation |
| `web/` | Dashboard |
| `deploy/`, `scripts/` | Docker, edge compose, edge sync client |
| `docs/` | MATLAB port guide, engineering notes |

### Clinical safety properties the backend enforces

* A machine grade is **written once and never mutated**. An ophthalmologist's
  correction is appended as a Review that supersedes it, so both survive and
  disagreement stays measurable.
* Every decision records **model version and thresholds**, so a past grade can
  be reproduced after an update.
* **Sign-off is ophthalmologist-only.** ASHA workers capture but cannot sign
  off; district admins are excluded too, since administrative seniority is not
  a clinical qualification.
* The queue orders by **clinical urgency**, not arrival. Strict FIFO would put
  a proliferative case behind last month's routine one.
* An **append-only audit log** covers every grade, review and failed login.

---

## Tests

```bash
python -m pytest tests -q
```

73 pass, 2 skip without IDRiD. They need no GPU, no trained model and no
Postgres. The suite asserts properties rather than numbers: that a featureless
retina yields zero lesions (guarding against a percentile threshold, which
makes "healthy" unrepresentable), that the crop transform round-trips, that
splits never leak a patient, that an ASHA worker cannot sign off, that a
retried sync batch deduplicates, and that a missing model degrades to a clear
503 while still persisting the capture.

`docs/engineering_notes.md` records nine defects found only under measurement,
several of which would have passed code review — including a 6% coordinate
misalignment in the *evaluator* that held lesion recall at 0.147 and made
tightening the threshold look like an improvement.

## MATLAB

PS 26038 is a MathWorks problem statement. Every operator in `core/` was chosen
to have a direct Image Processing Toolbox equivalent — `imbothat`,
`adapthisteq`, `fibermetric`, `regionprops`, `gradCAM`.
[`docs/matlab_mapping.md`](docs/matlab_mapping.md) gives the function-by-function
correspondence and the three places the port needs care.

## Status

Working: quality gate, segmentation, rule grader, explanation, triage, records,
auth, review queue, audit, offline sync, district simulation, deployment.

Requires training before clinical numbers exist: the CNN grader. No accuracy
figure is quoted in this repo because none has been measured on real data yet —
the synthetic generator that earlier stood in for a dataset has been removed
precisely so that no synthetic number can be mistaken for a clinical one.
