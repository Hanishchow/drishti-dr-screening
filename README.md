# Drishti — Explainable DR Screening & District Triage

Prototype for **SIH 2026 · PS 26038 (MathWorks)** — *"Design a retinal image
analysis pipeline for diabetic retinopathy screening and triage, complete with
explainability and district-scale telemedicine simulation."*

> **India's DR problem in one line:** 77M+ adults live with diabetes and about
> one in three will develop diabetic retinopathy, but rural India has roughly
> one ophthalmologist per 100,000 people. The bottleneck is not diagnosis — it
> is *specialist attention*. This system is built to spend that attention well.

---

## What it does

An ASHA worker photographs a patient's retina on a low-cost fundus camera. In
well under a second the system:

1. **Gates image quality** and tells the worker exactly what to fix, while the
   patient is still in the chair.
2. **Segments lesions** — microaneurysms, haemorrhages, hard exudates,
   cotton-wool spots — with sizes in microns and distance to the fovea.
3. **Grades severity** on the ICDR 0–4 scale using three independent graders.
4. **Explains itself** three ways: a pixel-exact lesion overlay, a Grad-CAM
   saliency map, and a narrative that cites measured numbers only.
5. **Triages** the patient into a referral window, escalating on evidence the
   grade alone does not capture.
6. **Simulates the district programme** so the value is measured in patients
   seen on time, not in accuracy points.

## The core design decision

Most DR systems are a single CNN that outputs a grade. That is a black box, and
a black box is exactly what a rural clinician will not act on.

This system pairs a **learned grader** with a **deterministic morphological
segmenter** that measures physical evidence, and then **checks them against each
other**. If the CNN produces a severe grade while its attention sits somewhere
the segmenter found no lesion, the case is flagged as *attention unexplained*
and routed to a human. The machine is built to be able to say "do not trust me
on this one" — which is what makes it trustworthy on the rest.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python train.py --cnn --pretrained
```

Omit `--cnn` to train only the feature grader in about a minute; the pipeline
runs correctly without a CNN, it simply omits Grad-CAM from the report.

```bash
uvicorn api.server:app --port 8000
```

Then open <http://localhost:8000>. No dataset needed — the **Synthetic case**
panel generates labelled retinas on demand, with sliders for blur, uneven
illumination, glare and exposure so the quality gate can be exercised live.

Other entry points:

```bash
python -m sim.district
```

```bash
python eval_lesions.py --n 8
```

```bash
python -m pytest tests -q
```

## Measured results

All figures are on held-out synthetic data (disjoint generator seeds).

| Stage | Metric | Result |
|---|---|---|
| Quality gate | clean images falsely rejected | 0 / 40 |
| Quality gate | blur, shadow, glare, exposure faults | all detected, correct guidance |
| Anatomy | optic disc localisation, median error | **4.5 px** (~115 µm) |
| Anatomy | fovea localisation, median error | **10.2 px** (~260 µm) |
| Lesion detection | dark lesions (MA + haemorrhage) F1 | **0.77** (P 0.79 / R 0.75) |
| Lesion detection | bright lesions (exudate + CWS) F1 | **0.76** (P 0.80 / R 0.72) |
| Grading | exact ICDR grade (all three graders fused) | **78.3%** |
| Grading | within ±1 grade | **100%** |
| Grading | **referable DR (≥2) sensitivity** | **1.000** |
| Grading | referable DR (≥2) specificity | **0.833** |
| Explainability | attention/lesion agreement, mean lift | **2.62×** chance |
| Explainability | cases flagged "attention unexplained" | 10 / 60 |
| Latency | rules + features (no CNN) | ~680 ms CPU |
| Latency | full pipeline incl. CNN + Grad-CAM | ~1.3 s CPU |

Sensitivity is the number that matters for screening, and it is deliberately
bought at the cost of specificity: fusion escalates to the most severe grade any
grader asserts confidently, so the system over-refers rather than miss disease.
The district simulation is where that trade is priced — the false referrals it
generates are counted against clinic capacity, not hidden.

For reference, the individual graders on the same held-out set:

| Grader | Exact | Referable sens / spec |
|---|---|---|
| ICDR rules (no training at all) | 42.7% | 0.956 / 0.833 |
| Feature GBM | 86.7% | 1.000 / 0.900 |
| EfficientNet-B0 | 73.3% | 1.000 / 1.000 |
| **Fused** | 78.3% | **1.000** / 0.833 |

Fused exact accuracy is *lower* than the feature grader alone, and that is the
design working rather than failing. The CNN confuses grades 3 and 4, and
escalation carries the more severe vote forward — so the fused system
over-grades some severe cases while never under-grading one. Within ±1 grade it
is 100%, and referable sensitivity is perfect.

The three graders also fail in usefully different places. The CNN is weakest at
grade 0 vs 1, which is a microaneurysm-counting problem the morphological
channel is built for; the morphological channel cannot see neovascularisation at
all, which the CNN can learn. The rule grader never predicts grade 4, and that
is correct rather than a defect — proliferative DR is defined by
neovascularisation, and the rule grader declares that limit explicitly in
`cannot_assess` instead of guessing.

**The CNN will not train from scratch on a cohort this small.** Without
ImageNet initialisation it collapses to predicting a single class for every
image — 20% accuracy, and far worse than having no CNN at all, since fusion
would drag every grade toward that class while Grad-CAM produced
convincing-looking saliency that meant nothing. `train.py` now refuses to save a
collapsed model. Use `--pretrained`.

District simulation, 25 PHCs and **2 ophthalmologists** over 180 days
(46,222 patients screened):

| | AI triage | Manual reading |
|---|---|---|
| Specialist image reads needed | 3,857 | 27,900 |
| Emergency cases seen within 7 days | **100%** | 9.9% |
| Urgent cases seen within 28 days | **100%** | 37.6% |
| Mean wait, urgent + emergency | **0.0 days** | 36.6 days |
| Backlog at end | 5,941 (routine) | 18,600 (all grades) |

The AI arm still has a backlog. The difference is *what is in it*: routine
patients who can safely wait, rather than the emergency cases stuck behind them
in the manual arm.

> **Scope note.** These numbers validate that the pipeline works end to end;
> they are **not clinical accuracy claims**. The generator is an approximation
> of retinal appearance, not a substitute for EyePACS/APTOS/IDRiD/Messidor.
> `data/synth.py` explains why it ships anyway: it provides pixel-level lesion
> ground truth that grade-only public datasets do not, and it makes the repo
> runnable with zero download. Retrain on real data before quoting any figure
> as clinical.

## Architecture

```
 fundus image
      |
      v
 [quality.py]  focus / illumination / FOV / exposure  --fail--> recapture guidance
      |
      v
 [preprocess.py]  FOV crop, CLAHE, illumination flattening
      |            (records its crop transform so annotations can be replayed)
      v
 [lesions.py]  multi-scale top-hat, vessel & disc suppression
      |         -> per-lesion class, area in um2, axes, distance to fovea
      v
 [features.py]  23 clinically-named features
      |
      +--> [grade.rule_grade]   ICDR rules, no training
      +--> [grade.FeatureGrader] gradient boosting + local attributions
      +--> [grade.CnnGrader]     EfficientNet-B0 + Grad-CAM
      |
      v
 [grade.fuse]  average, then escalate to the most severe confident vote
      |
      v
 [explain.py]  overlay + saliency + narrative + attention/lesion agreement
      |
      v
 [triage.py]  referral window, escalations, human-review flags
      |
      v
 [sim/district.py]  what this does to a district, at scale
```

## Layout

| Path | Purpose |
|---|---|
| `core/quality.py` | Capture-time quality gate and recapture guidance |
| `core/preprocess.py` | Normalisation and the replayable crop transform |
| `core/lesions.py` | Multi-scale morphological lesion segmentation |
| `core/features.py` | Lesion inventory → clinical feature vector |
| `core/grade.py` | Rule, feature and CNN graders, plus fusion |
| `core/explain.py` | Overlays, Grad-CAM, narrative, attention agreement |
| `core/triage.py` | Referral pathway and escalation logic |
| `core/pipeline.py` | End-to-end orchestration |
| `data/synth.py` | Synthetic fundus generator with pixel ground truth |
| `sim/district.py` | Discrete-event district telemedicine simulation |
| `api/server.py`, `web/index.html` | FastAPI service and dashboard |
| `train.py`, `eval_lesions.py`, `tune_lesions.py` | Training, evaluation, threshold sweeps |
| `docs/matlab_mapping.md` | Stage-by-stage MATLAB port guide |
| `docs/engineering_notes.md` | Seven measured defects, their causes and fixes |

## MATLAB

PS 26038 is a MathWorks problem statement. Every operation in the pipeline was
chosen to have a direct Image Processing Toolbox / Deep Learning Toolbox
equivalent — `imbothat`, `adapthisteq`, `fibermetric`, `regionprops`,
`gradCAM`, `efficientnetb0`. See [`docs/matlab_mapping.md`](docs/matlab_mapping.md)
for the function-by-function correspondence and the three places where the port
needs care.

## Using real data

Replace the generator with a real loader; nothing downstream changes:

```python
X, y = build_feature_dataset(...)   # swap data.synth.cohort for your loader
```

Each image needs a BGR array and an ICDR grade 0–4. For lesion-level evaluation
(`eval_lesions.py`) you also need per-class masks — IDRiD provides these.
