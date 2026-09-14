# Engineering notes

Nine defects found during development that were invisible to inspection and only
surfaced under measurement. Each is recorded with its symptom, cause, fix and
the measured effect, because each is a trap the next person to touch this
pipeline — or to port it to MATLAB — will otherwise walk straight into. The
regression test guarding each one is named.

---

## 1. A percentile threshold cannot express "healthy"

**Symptom.** A grade-0 retina returned 25 lesions. Grade 0 and grade 4 images
produced *identical* top-hat response statistics: median 10, MAD 5, max ~188.

**Cause.** The detector thresholded at the 99.3rd percentile of the response.
A percentile threshold declares a fixed fraction of pixels to be lesions
regardless of content, so "no disease" was not a representable outcome.

**Fix.** Threshold at `median + k·MAD`, anchored to the noise floor, with an
absolute grey-level floor.

**Effect.** A featureless retina now correctly returns zero lesions. This is the
single most important detail to preserve in any port.

**Guarded by** `test_featureless_retina_returns_no_lesions` (a smooth synthetic
retina must yield exactly zero) and
`test_healthy_eye_stays_within_the_graders_noise_floor`.

---

## 2. Morphology against a black surround measures the bezel

**Symptom.** Top-hat responses were dominated by something invisible in the
lesion maps, saturating the detector's dynamic range.

**Cause.** The retinal field of view is a bright circle on black. That rim is a
~150 grey-level step edge — larger than any lesion — and every structuring
element saw it.

**Fix.** Fill outside the FOV with the median retinal intensity before
morphology (`preprocess.fill_outside_fov`), and erode the working mask by more
than the largest structuring element.

**Effect.** Prerequisite for every subsequent detection improvement.

---

## 3. Ground truth compared in the wrong coordinate frame

**Symptom.** Dark-lesion recall sat at 0.147 and would not respond to *any*
threshold. Tightening the threshold improved F1, which is backwards.

**Cause.** An evaluation bug, not a detector bug. `prepare()` crops to the FOV
bounding box (~602 px) and resizes to 512, but the evaluator resized ground
truth from the full 640 px frame. Everything was ~6% out of register —
undetectable by eye, fatal at a 4 px matching tolerance.

**Fix.** `Prepared` now records `crop_origin` / `crop_side` and exposes
`map_from_original()`. Ground truth replays the exact transform.

**Effect.** Dark-lesion F1 **0.23 → 0.54** with no change to the detector.

**Guarded by** `test_crop_transform_round_trips` (FOV IoU > 0.95).

**Lesson.** When a metric is flat against a parameter it should respond to,
suspect the measurement before tuning the thing being measured.

---

## 4. Vessel suppression deleted the lesions it was meant to disambiguate

**Symptom.** After fixing (3), recall plateaued at 0.40 regardless of threshold.

**Cause.** Measured directly: of 752 true dark lesions, the FOV/disc mask
removed 11.4% and **vessel suppression removed 42.3%**, leaving a hard recall
ceiling of 46%. Microaneurysms are outpouchings of the capillary bed — they sit
*on* vessels by definition, so deleting vessel pixels deletes them.

**Fix.** Use the vessel map to *judge* rather than to erase. A component is
discarded only if it is both ≥70% covered by vessel AND not compact. Threshold
statistics are still computed off-vessel so the vessel population does not
inflate the noise estimate.

**Effect.** Dark-lesion F1 **0.54 → 0.78**.

---

## 5. Circularity was unbounded at the scale that matters most

**Symptom.** A test asserting `0 ≤ circularity ≤ 1` failed with the value π.

**Cause.** The textbook `4πA/P²` is unstable for tiny components. A 2-pixel blob
has a discrete perimeter so short that the ratio explodes. Because circularity
is exactly what separates a microaneurysm from a haemorrhage (threshold 0.55),
**every degenerate speck was being classified as a confident microaneurysm** —
a silent misclassification, not just a failing assertion.

**Fix.** Circularity is now the fill fraction of the minimum enclosing circle,
bounded in [0, 1] and stable at any size.

**Effect.** Correct MA/haemorrhage labelling. Note that this changed the
*scale* of the metric, so `VESSEL_CIRCULARITY_KEEP` had to be recalibrated
(0.45 → 0.85) — a threshold tuned against one definition is meaningless under
another.

**Guarded by** `test_lesion_measurements_are_physical`.

---

## 6. Anatomy localisation failed in two opposite ways

**Symptom.** The rendered overlay showed the fovea marker sitting on the optic
disc.

**Cause — fovea.** `locate_macula` took the darkest point of an annulus around
the disc. The vignetted retinal periphery is far darker than the fovea, so it
won every time. Mean error **263 px**. Compounding it, the search band was
expressed in *estimated* disc radii; since that estimate was ~50% too large,
the band (184–322 px) excluded the true fovea at ~150 px entirely.

**Cause — disc.** `locate_optic_disc` took the globally brightest smoothed
pixel. A healthy retina is brightest at the posterior pole, so the detector
found the centre of the frame, not the disc — and it failed *worst on healthy
eyes* (6/10) because there was nothing pathological to accidentally anchor it.

**Fix.** The fovea search is constrained to a wedge within ~40° of the temporal
direction, with the distance band expressed as a fraction of the FOV rather than
of an estimated radius. The disc detector subtracts a large-scale background so
it responds to *locally* bright structure, and uses normalised convolution so
the black surround does not dim a disc near the rim.

**Effect.** Fovea median error **263 px → 10 px**; disc median error
**22 px → 4.5 px**. Both now well inside the 59 px fovea radius that macular
escalation depends on, so the triage rule is measuring what it claims to.

**Guarded by** `test_optic_disc_and_macula_are_localised`.

---

## 7. The synthetic generator was wrong, not the detector

Found while diagnosing (6): raw red at the true disc centre was 151, though the
disc is drawn at 245. All eight vessel branches started at the exact disc centre
and painted dark vessel straight through it (245 × 0.62 = 152 — an exact match).

Vessels emerge from the disc *rim* in a real fundus. The generator now starts
each branch at 0.75·radius from the centre.

**Lesson.** When a detector fails hardest on the easiest cases, question the
data before the algorithm.

---

## 8. A report that contradicted itself

**Symptom.** A single report stated "Microaneurysms predominate, with no
significant haemorrhage or exudate" and, three lines later, "2 exudates within
1500 um of the fovea ... sight-threatening".

**Cause.** The rule grader's noise floor suppressed low exudate counts
everywhere on the retina, while triage correctly escalated on the same exudates
because of where they were.

**Fix.** The noise floor no longer applies when any exudate lies within the
fovea-centred circle. Two exudates are within the detector's error bar in the
periphery; at the macula they are sight-threatening at any count.

**Effect.** Zero self-contradictory reports across the held-out set, with fused
metrics unchanged (85.3% exact, sensitivity 1.000).

**Lesson.** Internal consistency is a correctness property of an explainable
system, not a presentation detail. A clinician who catches the system
contradicting itself once will not trust any of its output again.

**Guarded by** `test_report_never_contradicts_itself_about_macular_exudate`.

---

## 9. A collapsed CNN is worse than no CNN

**Symptom.** EfficientNet-B0 trained from random init reached 20% accuracy --
exactly chance -- by predicting grade 1 for all 60 test images.

**Cause.** 150 training images is nowhere near enough to train a CNN from
scratch.

**Why it matters more than the accuracy number.** A collapsed model is actively
harmful, not merely useless. Fusion would drag every grade toward the collapsed
class, and Grad-CAM over a degenerate model produces saliency maps that look
entirely convincing while meaning nothing -- the precise failure this system's
attention-agreement check exists to catch.

**Fix.** ImageNet initialisation (`--pretrained`): 73.3% exact, referable
sensitivity and specificity both 1.000. `train.py` now refuses to save a model
that predicts fewer than three distinct grades or scores under 40%.

---

## Two measurement traps worth repeating

**Survivorship bias in the simulation.** `within_window` was computed only over
patients who were actually *seen*. The AI arm scored 100% on every urgency band
while leaving 5,941 people in the queue — they simply were not counted. Patients
still queued at end-of-run are now charged against their window. The honest
comparison (emergency cases seen in time: 100% vs 9.9%) is more favourable to
the AI arm than the flattering one was, because the manual arm's failure is
precisely that it never reaches people.

**Stale calibration constants.** `NOISE_FLOOR` in `grade.py` encodes the
segmenter's false-positive rate. After the detector improved, these were silently
stale and the rule grader's specificity fell from 0.90 to 0.63 with no other
symptom. They are a property of the detector and must be recalibrated with it —
the procedure is documented at the constant.

---

## Metric design notes

The focus metric went through three implementations, and the two that failed
are instructive because both are the obvious choice:

1. **Laplacian variance** — not monotonic in blur.
2. **Contrast-normalised Laplacian variance** — dividing edge energy by image
   variance divides out the signal being measured. An underexposed frame scored
   **869** against a clean frame's 85.
3. **High-band / mid-band energy ratio** — *inverted*. Sensor noise is added
   after optical blur in a real camera, so a blurred frame keeps its full noise
   floor in the high band and scores as sharper than a crisp one.
4. **Mid-band structure vs overall contrast** (shipped) — monotonic in blur
   (226 → 84) and invariant to exposure (dark 228 vs clean 226).

Similarly, illumination uniformity must measure *directional* imbalance. Every
fundus image has strong radial falloff; a decile or min/max ratio flags healthy
images for it. Fitting a plane to the low-frequency luminance isolates the
one-sided shadow that actually warrants a recapture (0.889 clean → 0.203 under
heavy vignetting).

**Guarded by** `test_sharpness_decreases_monotonically_with_blur` and
`test_sharpness_is_exposure_invariant`.
