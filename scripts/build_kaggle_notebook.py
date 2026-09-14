"""Generate a self-contained Kaggle training notebook.

The notebook embeds the project's real module source verbatim and writes it to
disk at runtime. That matters for two reasons: the repository is private, so a
Kaggle kernel cannot clone it, and embedding the ACTUAL source means the
notebook can never drift from the code that was tested.

Regenerate after changing anything under dr/ or core/:

    python scripts/build_kaggle_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kaggle_train.ipynb"

# Training needs only dr/. core/ is included so the IDRiD lesion benchmark can
# run in the same kernel.
MODULES = [
    "dr/__init__.py", "dr/metrics.py", "dr/datasets.py", "dr/splits.py",
    "dr/transforms.py", "dr/model.py", "dr/torchdata.py", "dr/train.py",
    "dr/eval_lesions.py",
    "core/__init__.py", "core/quality.py", "core/preprocess.py",
    "core/lesions.py", "core/features.py",
]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip().splitlines(True)}


def build_source_cell():
    """One cell that materialises every module from an embedded dict."""
    files = {}
    for rel in MODULES:
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"missing module: {rel}")
        files[rel] = p.read_text(encoding="utf-8")

    payload = json.dumps(files, ensure_ascii=False, indent=0)
    return code(f'''
# The project source, embedded verbatim. Written to disk rather than cloned:
# the repository is private, so a Kaggle kernel cannot reach it, and embedding
# the real source means this notebook cannot drift from the tested code.
import json, sys, os
from pathlib import Path

PROJECT = Path("/kaggle/working/drishti") if Path("/kaggle").exists() else Path("./drishti_pkg")
_SOURCES = json.loads(r"""{payload}""")

for rel, text in _SOURCES.items():
    dest = PROJECT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

print(f"wrote {{len(_SOURCES)}} modules to {{PROJECT}}")
print("  " + "\\n  ".join(sorted(_SOURCES)))
'''.strip())


CELLS = [
    md("""
# Drishti — training the DR grader

Trains the ordinal (CORAL) diabetic-retinopathy grader on real fundus corpora
and exports `grader.onnx` for the API to serve.

**Self-contained.** No clone, no pip install of the project, no repository
access. Run the cells top to bottom.

---

## Before you run

**1. Attach the data.** Right panel → *Add Input* → search and attach:

| Dataset | Kaggle slug | Notes |
|---|---|---|
| APTOS 2019 | `aptos2019-blindness-detection` | Competition data — accept the rules first |
| EyePACS | `diabetic-retinopathy-detection` | 88k images, ~88 GB. Optional for a first run |
| IDRiD | upload as a private dataset | Not redistributable; needed for the lesion benchmark |
| Messidor-2 | upload as a private dataset | Keep as external validation |

Start with **APTOS alone** — it trains in roughly an hour and tells you whether
everything works before you commit to EyePACS.

**2. GPU on.** Settings → Accelerator → *GPU P100* (or T4 x2).

**3. Internet on.** Needed for ImageNet weights, which are not optional here —
see the note above the training cell.

Nothing is downloaded: Kaggle mounts the corpora read-only under
`/kaggle/input`, and the loaders resolve them from there automatically.
"""),

    md("## 1 · Environment"),
    code("""
!pip install -q timm onnx 2>&1 | tail -2

import torch, timm
print("torch", torch.__version__)
print("timm ", timm.__version__)
if torch.cuda.is_available():
    print("gpu  ", torch.cuda.get_device_name(0),
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
else:
    print("gpu   NONE — enable the accelerator in Settings, or this will take days")
"""),

    md("## 2 · Project source"),
    None,   # placeholder, replaced by build_source_cell()

    md("""
## 3 · What the data looks like

Check the corpora resolve, and look hard at the grade distribution. These
datasets run roughly 73% grade 0 and under 3% grade 4, and that imbalance — not
model capacity — is the main reason DR models miss sight-threatening disease.
"""),
    code("""
import dr.datasets as D

DATASETS = ["aptos"]          # add "idrid", "eyepacs" once attached

records = D.load(DATASETS)
if not records:
    raise SystemExit("No images found. Attach a dataset in the right-hand panel.")
print()
print(D.describe(records))
"""),

    md("""
## 4 · Split

Grouped by patient, stratified by grade. Both matter:

* **Grouping** — EyePACS carries both eyes of a patient and the two are highly
  correlated. A random *image* split puts one eye in train and the other in
  validation, so the score is partly memorisation. The assertion below fails
  the run rather than reporting an inflated number.
* **Stratification** — grades 3 and 4 are a few percent of these corpora. An
  unstratified fold can contain almost no severe disease, which makes its
  sensitivity estimate meaningless.
"""),
    code("""
import dr.splits as S

train_idx, val_idx = S.train_val_split(records, val_fraction=0.2, seed=0)
S.assert_no_patient_leakage(records, train_idx, val_idx)
print("no patient appears in both sides\\n")
print(S.summarise_split(records, train_idx, val_idx))
"""),

    md("""
## 5 · Cache the resized images  *(optional, do it once)*

Decoding multi-megapixel JPEGs — not the GPU — is what makes an epoch slow. On
EyePACS this is the difference between hours and minutes per epoch. Skip it for
a small APTOS-only run; it costs more than it saves there.
"""),
    code("""
# from dr.torchdata import build_cache
# build_cache(records, size=512, cache_dir="/kaggle/working/cache", workers=4)
"""),

    md("""
## 6 · Train

**`--pretrained` is not optional.** From random initialisation on a cohort this
size the network collapses to predicting one class for every image: 20%
accuracy, and *worse than having no CNN at all*, because fusion would drag every
grade toward that class while Grad-CAM produced convincing-looking saliency that
meant nothing. The trainer refuses to save such a model.

**Kaggle kills sessions at 12 hours.** Every epoch checkpoints to
`artifacts/last.pt`. To continue, re-run this cell with `--resume` uncommented.

Rough timings on a P100 with APTOS (~3,660 images):

| Backbone | Size | Batch | Per epoch |
|---|---|---|---|
| `tf_efficientnet_b0_ns` | 384 | 24 | ~3 min |
| `tf_efficientnet_b3_ns` | 512 | 12 | ~6 min |
| `tf_efficientnet_b4_ns` | 640 | 8 | ~12 min |

Resolution matters more than depth here — microaneurysms are only a few pixels
across, so dropping below 384 px removes the earliest sign of disease from the
image entirely.
"""),
    code("""
from dr.train import main as train_main

ARTIFACTS = "/kaggle/working/artifacts"

train_main([
    "--datasets", *DATASETS,
    "--size", "512",
    "--backbone", "tf_efficientnet_b3_ns",
    "--epochs", "12",
    "--batch-size", "12",
    "--lr", "3e-4",
    "--workers", "2",
    "--out", ARTIFACTS,
    # "--cache-dir", "/kaggle/working/cache",
    # "--external", "messidor2",              # never trained on — a true external score
    # "--resume", f"{ARTIFACTS}/last.pt",     # uncomment to continue a killed session
])
"""),

    md("""
## 7 · Results

Read **QWK first**. It is what this task is scored on, and the only common
metric that understands the grades are ordinal — confusing grade 0 with grade 4
is far worse than confusing 3 with 4, and plain accuracy scores those
identically.

Then read **referable sensitivity**: that is what a screening programme is
actually accountable for. The NHS DR screening standard is ≥85% sensitivity and
≥80% specificity for referable disease.
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
## 8 · Lesion benchmark  *(IDRiD only)*

IDRiD is the only public corpus with **pixel-level lesion masks**, which makes
it the only way to score the morphological segmenter's "where" channel
honestly. Grade-only corpora can assess it indirectly at best.

Skip this cell if IDRiD is not attached.
"""),
    code("""
# from dr.eval_lesions import main as eval_lesions
# eval_lesions([])
"""),

    md("""
## 9 · Export and download

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
    code("""
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
"""),

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

Then pass `--data-root` explicitly, or tell me the layout and I will widen the
loader's candidate paths.

**`REFUSED TO SAVE: model predicts only N distinct grades`** — working as
intended. The model collapsed. Use `--pretrained` (it is on by default; make
sure internet is enabled so the weights can download), or train on more data.

**CUDA out of memory** — halve `--batch-size` and add `--accum 2`, which keeps
the effective batch size while holding half as much in memory.

**Session killed mid-run** — uncomment `--resume` in the training cell. It picks
up from the last completed epoch, optimiser and scheduler state included.
"""),
]


def main():
    cells = [c for c in CELLS]
    cells[cells.index(None)] = build_source_cell()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB, {len(cells)} cells, {len(MODULES)} modules embedded)")


if __name__ == "__main__":
    main()
