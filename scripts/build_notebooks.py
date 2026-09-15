"""Generate the self-contained training notebooks for Kaggle and Colab.

Both embed the project's real module source verbatim and write it to disk at
runtime. That matters for two reasons: the repository is private, so neither
platform can clone it, and embedding the ACTUAL source means a notebook can
never drift from the code that was tested.

Regenerate after changing anything under dr/ or core/:

    python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

# Training needs only dr/. core/ is included so the IDRiD lesion benchmark can
# run in the same session.
MODULES = [
    "dr/__init__.py", "dr/metrics.py", "dr/datasets.py", "dr/splits.py",
    "dr/transforms.py", "dr/model.py", "dr/torchdata.py", "dr/train.py",
    "dr/eval_lesions.py",
    "core/__init__.py", "core/quality.py", "core/preprocess.py",
    "core/lesions.py", "core/features.py",
]


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip().splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip().splitlines(True)}


def source_cell(project_dir: str):
    files = {}
    for rel in MODULES:
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"missing module: {rel}")
        files[rel] = p.read_text(encoding="utf-8")
    payload = json.dumps(files, ensure_ascii=False, indent=0)

    return code(f'''
# The project source, embedded verbatim and written to disk. Not cloned: the
# repository is private, and embedding the real source means this notebook
# cannot drift from the tested code.
import json, sys, os
from pathlib import Path

PROJECT = Path("{project_dir}")
_SOURCES = json.loads(r"""{payload}""")

for rel, text in _SOURCES.items():
    dest = PROJECT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

print(f"wrote {{len(_SOURCES)}} modules to {{PROJECT}}")
''')


# --------------------------------------------------------------- shared body
def shared_cells(artifacts_expr: str, resume_hint: str):
    """Cells identical between platforms: data, split, train, results, export."""
    return [
        md("""
## What the data looks like

Check the corpora resolve, and look hard at the grade distribution. These
datasets run roughly 73% grade 0 and under 3% grade 4, and that imbalance — not
model capacity — is the main reason DR models miss sight-threatening disease.
"""),
        code("""
import dr.datasets as D

DATASETS = ["aptos"]          # add "idrid", "eyepacs" once available

records = D.load(DATASETS)
if not records:
    raise SystemExit("No images found — see the data cell above.")
print()
print(D.describe(records))
"""),

        md("""
## Split

Grouped by patient, stratified by grade. Both matter:

* **Grouping** — EyePACS carries both eyes of a patient and the two are highly
  correlated. A random *image* split puts one eye in train and the other in
  validation, so the score is partly memorisation. The assertion below fails the
  run rather than reporting an inflated number.
* **Stratification** — grades 3 and 4 are a few percent of these corpora. An
  unstratified fold can contain almost no severe disease, which makes its
  sensitivity estimate meaningless.
"""),
        code("""
import dr.splits as S

train_idx, val_idx = S.train_val_split(records, val_fraction=0.2, seed=0)
S.assert_no_patient_leakage(records, train_idx, val_idx)
print("no patient appears on both sides\\n")
print(S.summarise_split(records, train_idx, val_idx))
"""),

        md("""
## Cache the resized images  *(optional, once)*

Decoding multi-megapixel JPEGs — not the GPU — is what makes an epoch slow. On
EyePACS this is the difference between hours and minutes per epoch. Skip it for
a small APTOS-only run, where it costs more than it saves.
"""),
        code("""
# from dr.torchdata import build_cache
# build_cache(records, size=512, cache_dir=str(CACHE_DIR), workers=4)
"""),

        md(f"""
## Train

**`--pretrained` is not optional.** From random initialisation on a cohort this
size the network collapses to predicting one class for every image: 20%
accuracy, and *worse than having no CNN at all*, because fusion would drag every
grade toward that class while Grad-CAM produced convincing-looking saliency that
meant nothing. The trainer refuses to save such a model.

{resume_hint}

Rough timings on a T4 with APTOS (~3,660 images):

| Backbone | Size | Batch | Per epoch |
|---|---|---|---|
| `tf_efficientnet_b0_ns` | 384 | 24 | ~5 min |
| `tf_efficientnet_b3_ns` | 512 | 12 | ~10 min |
| `tf_efficientnet_b4_ns` | 640 | 8 | ~18 min |

Resolution matters more than depth here — microaneurysms are only a few pixels
across, so dropping below 384 px removes the earliest sign of disease from the
image entirely.
"""),
        code(f"""
from dr.train import main as train_main

ARTIFACTS = {artifacts_expr}

train_main([
    "--datasets", *DATASETS,
    "--size", "512",
    "--backbone", "tf_efficientnet_b3_ns",
    "--epochs", "12",
    "--batch-size", "12",
    "--lr", "3e-4",
    "--workers", "2",
    "--out", str(ARTIFACTS),
    # "--cache-dir", str(CACHE_DIR),
    # "--external", "messidor2",                   # never trained on
    # "--resume", str(ARTIFACTS / "last.pt"),      # continue a killed session
])
"""),

        md("""
## Results

Read **QWK first**. It is what this task is scored on, and the only common
metric that understands the grades are ordinal — confusing grade 0 with grade 4
is far worse than confusing 3 with 4, and plain accuracy scores those
identically.

Then read **referable sensitivity**: that is what a screening programme is
accountable for. The NHS DR screening standard is ≥85% sensitivity and ≥80%
specificity for referable disease.
"""),
        code("""
import json
from pathlib import Path
import dr.metrics as M

art = Path(ARTIFACTS)
metrics = json.loads((art / "metrics.json").read_text())

print(M.format_report(metrics["val"], "validation"))
print()
print("grade cut-points:", [round(t, 3) for t in metrics["thresholds"]])
print("best epoch      :", metrics["epoch"])
"""),
        code("""
history = json.loads((art / "history.json").read_text())
print(f"{'epoch':>5} {'loss':>9} {'QWK':>8} {'acc':>7} {'ref sens':>9} {'ref spec':>9}")
for h in history:
    print(f"{h['epoch']:>5} {h['train_loss']:>9.4f} {h['qwk']:>8.4f} "
          f"{h['accuracy']:>7.3f} {h['referable_sensitivity']:>9.3f} "
          f"{h['referable_specificity']:>9.3f}")
"""),

        md("""
## Lesion benchmark  *(IDRiD only)*

IDRiD is the only public corpus with **pixel-level lesion masks**, which makes
it the only way to score the morphological segmenter's "where" channel honestly.
Grade-only corpora can assess it indirectly at best.

Skip this cell if IDRiD is not available.
"""),
        code("""
# from dr.eval_lesions import main as eval_lesions
# eval_lesions([])
"""),
    ]


def parity_cell():
    return code("""
# Parity check: the exported graph must agree with the checkpoint that was
# validated. If these diverge, the model you serve is not the model you
# measured — which would make every number above meaningless.
import numpy as np, torch
import dr.model as MD, dr.metrics as M

ck = torch.load(art / "best.pt", map_location="cpu")
net = MD.build(ck["backbone"], pretrained=False)
net.load_state_dict(ck["model"])
net.eval()

x = np.random.randn(2, 3, ck["size"], ck["size"]).astype(np.float32)
with torch.no_grad():
    torch_grade = M.coral_expected_grade(net(torch.from_numpy(x)).numpy())
onnx_grade, _ = MD.OnnxGrader(art / "grader.onnx")(x)

drift = float(np.abs(torch_grade - onnx_grade).max())
print(f"max |torch - onnx| = {drift:.2e}")
assert drift < 1e-4, "exported graph diverged from the validated checkpoint"
print("parity OK — safe to deploy")
""")


# ------------------------------------------------------------------- Kaggle
def build_kaggle():
    cells = [
        md("""
# Drishti — training the DR grader (Kaggle)

Trains the ordinal (CORAL) diabetic-retinopathy grader on real fundus corpora
and exports `grader.onnx` for the API to serve.

**Self-contained.** No clone, no repository access. Run the cells top to bottom.

---

## Before you run

**1. Attach the data.** Right panel → *Add Input* → search and attach:

| Dataset | Kaggle slug | Notes |
|---|---|---|
| APTOS 2019 | `aptos2019-blindness-detection` | Competition data — accept the rules first |
| EyePACS | `diabetic-retinopathy-detection` | 88k images, ~88 GB. Optional for a first run |
| IDRiD | upload as a private dataset | Not redistributable; needed for the lesion benchmark |
| Messidor-2 | upload as a private dataset | Keep as external validation |

Start with **APTOS alone** — it trains in about two hours and tells you whether
everything works before you commit to EyePACS.

**2. GPU on.** Settings → Accelerator → *GPU P100* (or T4 x2).

**3. Internet on.** Needed for ImageNet weights, which are not optional here.

Nothing is downloaded: Kaggle mounts the corpora read-only under
`/kaggle/input`, and the loaders resolve them from there automatically.
"""),
        md("## 1 · Environment"),
        code("""
!pip install -q timm onnx 2>&1 | tail -2

import torch, timm
print("torch", torch.__version__, "| timm", timm.__version__)
if torch.cuda.is_available():
    print("gpu  ", torch.cuda.get_device_name(0),
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
else:
    print("gpu   NONE — enable the accelerator in Settings")
"""),
        md("## 2 · Project source"),
        source_cell("/kaggle/working/drishti"),
        code("""
from pathlib import Path
CACHE_DIR = Path("/kaggle/working/cache")
"""),
    ]
    cells += shared_cells(
        artifacts_expr='Path("/kaggle/working/artifacts")',
        resume_hint=("**Kaggle kills sessions at 12 hours.** Every epoch "
                     "checkpoints to `artifacts/last.pt` on the working disk, "
                     "which persists across a *saved* run. To continue, re-run "
                     "this cell with `--resume` uncommented."))
    cells += [
        md("""
## Export and download

`grader.onnx` plus `grader.json` (cut-points, backbone, input size) is
everything the server needs — it loads the frozen graph and never imports the
training code.

Download both from the notebook's **Output** tab, drop them into `artifacts/`
beside the deployment, and restart. `/api/health` will then report the model as
available and grading will stop returning 503.
"""),
        code("""
for f in sorted(art.iterdir()):
    print(f"{f.name:22} {f.stat().st_size/1e6:9.1f} MB")
"""),
        parity_cell(),
        md("""
---

## If something goes wrong

**`No images found`** — the dataset is not attached, or its folder name differs
from what the loader expects. Check what is actually mounted:

```python
from pathlib import Path
for p in sorted(Path("/kaggle/input").iterdir()):
    print(p.name, "→", [c.name for c in p.iterdir()][:6])
```

Then pass `--data-root`, or send me the layout and I will widen the loader paths.

**`REFUSED TO SAVE: model predicts only N distinct grades`** — working as
intended. The model collapsed. Keep `--pretrained` and make sure internet is on.

**CUDA out of memory** — halve `--batch-size` and add `--accum 2`, which keeps
the effective batch size while holding half as much in memory.

**Session killed mid-run** — uncomment `--resume`. It picks up from the last
completed epoch, optimiser and scheduler state included.
"""),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"},
                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


# -------------------------------------------------------------------- Colab
def build_colab():
    cells = [
        md("""
# Drishti — training the DR grader (Google Colab)

Trains the ordinal (CORAL) diabetic-retinopathy grader on real fundus corpora
and exports `grader.onnx` for the API to serve.

**Self-contained.** No clone, no repository access. Run the cells top to bottom.

---

## Before you run

**1. Turn on the GPU.** *Runtime → Change runtime type → T4 GPU*. Do this
**first** — changing it later restarts the runtime and wipes everything.

**2. Mount Drive.** This is not optional on Colab. The local disk is destroyed
when the runtime disconnects, and free Colab disconnects readily — on idle, and
at roughly 12 hours regardless. Checkpoints go to Drive so a dropped session
costs one epoch instead of the whole run.

**3. Have a Kaggle API token ready** if you want APTOS or EyePACS. Get it from
<https://www.kaggle.com/settings> → *API* → *Create New Token*, which downloads
`kaggle.json`. The next cells will ask you to upload it.

> ⚠️ `kaggle.json` is a credential. It is written to `~/.kaggle/` inside this
> throwaway runtime only — never to Drive, and never into the notebook. Anyone
> with it can act as you on Kaggle.

### A note on disk

Colab gives roughly 80 GB (free) or 200 GB (Pro) of local disk. **EyePACS is
~88 GB and will not fit on the free tier.** Options, best first:

1. Train on **APTOS + IDRiD** — the realistic free-tier path, and APTOS is the
   closest public proxy to an Indian screening population anyway.
2. Use the **resized EyePACS mirror** (`tanlikesmath/diabetic-retinopathy-resized`,
   ~10 GB), which the loader already recognises.
3. Colab Pro with a high-RAM/disk runtime for the full corpus.
"""),
        md("## 1 · GPU"),
        code("""
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "No GPU. Runtime -> Change runtime type -> T4 GPU, then run this cell "
        "again. Changing it now will restart the runtime.")
p = torch.cuda.get_device_properties(0)
print("torch", torch.__version__)
print(f"gpu   {p.name} ({p.total_memory/1e9:.0f} GB)")
if p.total_memory < 14e9:
    print("\\nNOTE: under 14 GB. Use --batch-size 8 --accum 2 in the training cell.")
"""),

        md("## 2 · Mount Drive"),
        code("""
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

# Everything that must survive a disconnect lives here.
DRIVE_ROOT = Path("/content/drive/MyDrive/drishti")
ARTIFACTS  = DRIVE_ROOT / "artifacts"      # checkpoints + exported model
DATA_ROOT  = Path("/content/datasets")     # local scratch: fast, disposable
CACHE_DIR  = Path("/content/cache")        # resized-image cache, disposable

for d in (ARTIFACTS, DATA_ROOT, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("checkpoints ->", ARTIFACTS, "(survives a disconnect)")
print("data        ->", DATA_ROOT, "(local scratch, wiped on disconnect)")
"""),

        md("## 3 · Dependencies"),
        code("""
!pip install -q timm onnx kaggle 2>&1 | tail -2

import timm
print("timm", timm.__version__)
"""),

        md("## 4 · Project source"),
        source_cell("/content/drishti"),

        md("""
## 5 · Get the data

Two routes. Run **either** the Kaggle download or the Drive cell — whichever
suits what you have.
"""),
        md("### 5a · From Kaggle (APTOS, EyePACS)"),
        code("""
# Upload the kaggle.json you downloaded from kaggle.com/settings.
import os, json
from pathlib import Path
from google.colab import files

kp = Path.home() / ".kaggle" / "kaggle.json"
if not kp.exists():
    print("Select your kaggle.json...")
    up = files.upload()
    name = next(iter(up))
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_bytes(up[name])
    # The Kaggle client refuses a world-readable token, which is the right call.
    kp.chmod(0o600)
print("kaggle credentials ready for user:", json.loads(kp.read_text())["username"])
"""),
        code("""
import os, subprocess
from pathlib import Path
os.environ["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle")

APTOS = DATA_ROOT / "aptos2019-blindness-detection"
if not APTOS.exists():
    # Competition data: you must accept the rules on the competition page first,
    # or this returns 403.
    subprocess.run(["kaggle", "competitions", "download",
                    "-c", "aptos2019-blindness-detection",
                    "-p", str(DATA_ROOT)], check=True)
    subprocess.run(["unzip", "-q", "-o",
                    str(DATA_ROOT / "aptos2019-blindness-detection.zip"),
                    "-d", str(APTOS)], check=True)
    (DATA_ROOT / "aptos2019-blindness-detection.zip").unlink(missing_ok=True)

print("APTOS ready:", APTOS)
print(" ", [p.name for p in sorted(APTOS.iterdir())[:6]])
"""),
        code("""
# Optional: the resized EyePACS mirror (~10 GB) — the full set will not fit on
# free Colab. The loader recognises this layout.
#
# subprocess.run(["kaggle", "datasets", "download",
#                 "-d", "tanlikesmath/diabetic-retinopathy-resized",
#                 "-p", str(DATA_ROOT), "--unzip"], check=True)
"""),
        md("""
### 5b · From Drive (IDRiD, Messidor-2)

Neither is redistributable, so upload them to Drive yourself and point the
loader at them. Expected layout:

```
MyDrive/drishti/datasets/idrid/       ← keep the "A. Segmentation" and
                                        "B. Disease Grading" folders
MyDrive/drishti/datasets/messidor2/   ← images + the grading CSV
```
"""),
        code("""
# Symlink rather than copy: these are read a few times per epoch at most, and
# copying tens of GB from Drive wastes most of a session.
drive_data = DRIVE_ROOT / "datasets"
if drive_data.exists():
    for d in drive_data.iterdir():
        link = DATA_ROOT / d.name
        if d.is_dir() and not link.exists():
            link.symlink_to(d, target_is_directory=True)
            print("linked", d.name)
else:
    print(f"No {drive_data} — skip this if you only have Kaggle data.")
"""),
        code("""
import os
os.environ["DR_DATA_ROOT"] = str(DATA_ROOT)
print("DR_DATA_ROOT =", DATA_ROOT)
print("available:", [p.name for p in sorted(DATA_ROOT.iterdir())])
"""),
    ]
    cells += shared_cells(
        artifacts_expr="ARTIFACTS",
        resume_hint=(
            "**Colab disconnects, and the local disk goes with it.** "
            "`ARTIFACTS` points at Drive, so every epoch's checkpoint survives. "
            "If the session drops, re-run the notebook from the top and "
            "uncomment `--resume` — it restores the optimiser and scheduler "
            "state too, not just the weights."))
    cells += [
        md("""
## Export and download

`grader.onnx` plus `grader.json` (cut-points, backbone, input size) is
everything the server needs — it loads the frozen graph and never imports the
training code.

Both are already on Drive under `MyDrive/drishti/artifacts/`. The cell below
also offers them as direct browser downloads.
"""),
        code("""
for f in sorted(art.iterdir()):
    print(f"{f.name:22} {f.stat().st_size/1e6:9.1f} MB")
"""),
        parity_cell(),
        code("""
# Download the two files the server needs. The .pt checkpoints stay on Drive —
# they are large and only needed to resume training.
from google.colab import files
for name in ("grader.onnx", "grader.json"):
    f = art / name
    if f.exists():
        files.download(str(f))
    else:
        print(f"{name} missing — did training finish and export?")
"""),
        md("""
---

## If something goes wrong

**`No GPU`** — *Runtime → Change runtime type → T4 GPU*. Doing this restarts the
runtime, so re-run from the top.

**`403` downloading APTOS** — you have not accepted the competition rules. Open
<https://www.kaggle.com/c/aptos2019-blindness-detection/rules>, accept, retry.

**`No images found`** — the folder name differs from what the loader expects.
Check what actually landed:

```python
from pathlib import Path
for p in sorted(DATA_ROOT.rglob("*")):
    if p.is_dir() and len(p.relative_to(DATA_ROOT).parts) <= 2:
        print(p.relative_to(DATA_ROOT))
```

Then pass `--data-root`, or send me the layout and I will widen the loader paths.

**Disk full** — EyePACS does not fit on free Colab. Use the resized mirror in
cell 5a, or train on APTOS + IDRiD.

**Runtime disconnected** — expected on free Colab. Re-run from the top and
uncomment `--resume`; Drive has the checkpoint. To lose less to a drop, lower
`--epochs` and run the cell repeatedly, each time with `--resume`.

**`REFUSED TO SAVE: model predicts only N distinct grades`** — working as
intended. The model collapsed; keep `--pretrained`.

**CUDA out of memory** — halve `--batch-size` and add `--accum 2`, which keeps
the effective batch size while holding half as much in memory.
"""),
    ]
    return {"cells": cells,
            "metadata": {
                "colab": {"provenance": [], "gpuType": "T4",
                          "toc_visible": True},
                "kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python"},
                "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


def write(nb, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} "
          f"({path.stat().st_size/1024:.0f} KB, {len(nb['cells'])} cells)")


def main():
    write(build_kaggle(), NB_DIR / "kaggle_train.ipynb")
    write(build_colab(), NB_DIR / "colab_train.ipynb")
    print(f"{len(MODULES)} modules embedded in each")


if __name__ == "__main__":
    main()
