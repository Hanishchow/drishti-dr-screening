"""Unified access to the four public DR corpora.

Every loader returns the same record shape, so training, evaluation and the
lesion benchmark never need to know which corpus they are looking at:

    {"image_path": str, "grade": int 0-4, "patient_id": str, "eye": "L"|"R"|None,
     "dataset": str, "split": str}

Corpora and what each is for:

  IDRiD      516 graded images, and 81 with PIXEL-LEVEL lesion masks
             (microaneurysm, haemorrhage, hard exudate, soft exudate). The only
             public set that can score the morphological segmenter's "where"
             channel, which is why it is worth its small size.
  APTOS-2019 3,662 images, ICDR 0-4, Indian population (Aravind Eye Hospital).
             The closest public proxy to the deployment population.
  EyePACS    88,702 images, ICDR 0-4. Large, noisy, heavily imbalanced; the
             corpus a CNN needs to actually generalise.
  Messidor-2 1,748 images. Held out as an EXTERNAL validation set and never
             trained on, so the reported number is not a within-corpus score.

Paths are resolved through `DR_DATA_ROOT` (env var) or an explicit root, so the
same code runs on Kaggle (/kaggle/input/...) and locally without edits.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Kaggle mounts each dataset read-only under /kaggle/input/<slug>.
KAGGLE_INPUT = Path("/kaggle/input")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}

LESION_CLASSES = {
    "MA": "Microaneurysms",
    "HEM": "Haemorrhages",
    "EX": "Hard Exudates",
    "CWS": "Soft Exudates",
}


@dataclass
class Record:
    image_path: str
    grade: int
    patient_id: str
    dataset: str
    eye: str | None = None
    split: str = "train"
    lesion_masks: dict = field(default_factory=dict)   # class -> mask path

    def as_dict(self):
        return {"image_path": self.image_path, "grade": int(self.grade),
                "patient_id": self.patient_id, "dataset": self.dataset,
                "eye": self.eye, "split": self.split,
                "lesion_masks": dict(self.lesion_masks)}


class DatasetUnavailable(RuntimeError):
    """Raised when a corpus is not present on disk.

    Carries the download instructions rather than just failing, because the
    single most common way this pipeline breaks for a new user is a dataset
    that was never fetched.
    """


def data_root(explicit=None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("DR_DATA_ROOT")
    if env:
        return Path(env)
    if KAGGLE_INPUT.exists():
        return KAGGLE_INPUT
    return Path(__file__).resolve().parent.parent / "datasets"


def _first_existing(root: Path, candidates):
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    return None


def _index_images(folder: Path):
    """Map bare stem -> path, so a CSV id can be resolved regardless of the
    extension or the nesting the corpus happens to use."""
    out = {}
    for p in folder.rglob("*"):
        if p.suffix in IMAGE_SUFFIXES and p.is_file():
            out.setdefault(p.stem, p)
    return out


def _read_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _col(row, *names):
    """Fetch a column by any of several spellings; corpora disagree on case
    and separators (`Retinopathy grade`, `diagnosis`, `level`, ...)."""
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in row.items() if k}
    for n in names:
        key = re.sub(r"[^a-z0-9]", "", n.lower())
        if key in norm:
            return norm[key]
    return None


# --------------------------------------------------------------- APTOS 2019
def load_aptos(root=None, split="train"):
    root = data_root(root)
    base = _first_existing(root, ["aptos2019-blindness-detection", "aptos2019",
                                  "aptos", "APTOS2019"])
    if base is None:
        raise DatasetUnavailable(
            "APTOS 2019 not found. On Kaggle add the dataset "
            "'aptos2019-blindness-detection'; locally run\n"
            "  kaggle competitions download -c aptos2019-blindness-detection\n"
            f"and unpack under {root}/aptos2019-blindness-detection")

    csv_path = _first_existing(base, ["train.csv", "train_1.csv"])
    img_dir = _first_existing(base, ["train_images", "train"])
    if csv_path is None or img_dir is None:
        raise DatasetUnavailable(f"APTOS present at {base} but train.csv/train_images missing")

    index = _index_images(img_dir)
    records = []
    for row in _read_csv(csv_path):
        ident = _col(row, "id_code", "image", "id")
        grade = _col(row, "diagnosis", "level", "grade")
        if ident is None or grade is None:
            continue
        path = index.get(str(ident))
        if path is None:
            continue
        # APTOS ships one image per patient with no laterality metadata, so the
        # image id IS the grouping key. Treating each image as its own patient
        # is correct here and must not be copied to corpora where it is not.
        records.append(Record(str(path), int(grade), f"aptos:{ident}",
                              "aptos", None, split))
    return records


# ------------------------------------------------------------------ EyePACS
def load_eyepacs(root=None, split="train"):
    root = data_root(root)
    base = _first_existing(root, ["diabetic-retinopathy-detection", "eyepacs",
                                  "diabetic-retinopathy-resized"])
    if base is None:
        raise DatasetUnavailable(
            "EyePACS not found. On Kaggle add 'diabetic-retinopathy-detection' "
            "(or the resized mirror 'diabetic-retinopathy-resized'); locally\n"
            "  kaggle competitions download -c diabetic-retinopathy-detection")

    csv_path = _first_existing(base, ["trainLabels.csv", "trainLabels/trainLabels.csv",
                                      "train.csv", "trainLabels_cropped.csv"])
    img_dir = _first_existing(base, ["train", "resized_train", "train_images",
                                     "resized_train_cropped"])
    if csv_path is None or img_dir is None:
        raise DatasetUnavailable(f"EyePACS present at {base} but labels/images missing")

    index = _index_images(img_dir)
    records = []
    for row in _read_csv(csv_path):
        ident = _col(row, "image", "id_code")
        grade = _col(row, "level", "diagnosis", "grade")
        if ident is None or grade is None:
            continue
        path = index.get(str(ident))
        if path is None:
            continue
        # EyePACS ids are '<patient>_<left|right>'. Both eyes of one patient are
        # strongly correlated, so the patient -- not the image -- is the unit
        # that must stay on one side of a split.
        m = re.match(r"^(\d+)_(left|right)$", str(ident))
        pid = f"eyepacs:{m.group(1)}" if m else f"eyepacs:{ident}"
        eye = {"left": "L", "right": "R"}.get(m.group(2)) if m else None
        records.append(Record(str(path), int(grade), pid, "eyepacs", eye, split))
    return records


# -------------------------------------------------------------------- IDRiD
def load_idrid(root=None, split="train", with_masks=True):
    """IDRiD grading set, with lesion masks attached where they exist."""
    root = data_root(root)
    base = _first_existing(root, ["idrid", "IDRiD", "indian-diabetic-retinopathy-image-dataset",
                                  "diabetic-retinopathy-segmentation"])
    if base is None:
        raise DatasetUnavailable(
            "IDRiD not found. Download from "
            "https://idrid.grand-challenge.org/ (free registration) and unpack "
            f"under {root}/idrid, keeping the 'A. Segmentation' and "
            "'B. Disease Grading' folders.")

    grade_csv = None
    for p in base.rglob("*.csv"):
        name = p.name.lower()
        if "groundtruth" in name.replace(" ", "") or "grading" in name or "label" in name:
            if "train" in name or split == "train":
                grade_csv = p
                break
    if grade_csv is None:
        raise DatasetUnavailable(f"IDRiD found at {base} but no grading CSV located")

    grading_imgs = {}
    for d in base.rglob("*"):
        if d.is_dir() and "grading" in d.name.lower().replace(" ", ""):
            grading_imgs.update(_index_images(d))
    if not grading_imgs:
        grading_imgs = _index_images(base)

    masks_by_stem = _idrid_masks(base) if with_masks else {}

    records = []
    for row in _read_csv(grade_csv):
        ident = _col(row, "Image name", "image", "id")
        grade = _col(row, "Retinopathy grade", "retinopathygrade", "grade", "level")
        if ident is None or grade is None:
            continue
        stem = str(ident).strip()
        path = grading_imgs.get(stem)
        if path is None:
            continue
        records.append(Record(str(path), int(grade), f"idrid:{stem}", "idrid",
                              None, split, masks_by_stem.get(stem, {})))
    return records


def _idrid_masks(base: Path):
    """Locate per-class lesion masks.

    IDRiD names them '<stem>_MA.tif', '<stem>_HE.tif', '<stem>_EX.tif',
    '<stem>_SE.tif' inside folders whose names carry the class. Both the
    filename suffix and the parent folder are checked, because the two IDRiD
    distributions on Kaggle differ in which one they preserve.
    """
    suffix_map = {"ma": "MA", "he": "HEM", "ex": "EX", "se": "CWS"}
    folder_map = {"microaneurysm": "MA", "haemorrhage": "HEM", "hemorrhage": "HEM",
                  "hardexudate": "EX", "softexudate": "CWS", "opticdisc": None}
    out = {}
    for p in base.rglob("*"):
        if p.suffix.lower() not in {".tif", ".tiff", ".png"} or not p.is_file():
            continue
        m = re.match(r"^(.*?)_([A-Za-z]{2})$", p.stem)
        cls = None
        if m and m.group(2).lower() in suffix_map:
            stem, cls = m.group(1), suffix_map[m.group(2).lower()]
        else:
            folder = re.sub(r"[^a-z]", "", p.parent.name.lower())
            for key, val in folder_map.items():
                if key in folder:
                    stem, cls = p.stem, val
                    break
        if cls:
            out.setdefault(stem, {})[cls] = str(p)
    return out


def load_idrid_segmentation(root=None):
    """Only the images that carry pixel-level lesion ground truth."""
    return [r for r in load_idrid(root, with_masks=True) if r.lesion_masks]


# --------------------------------------------------------------- Messidor-2
def load_messidor2(root=None, split="external"):
    root = data_root(root)
    base = _first_existing(root, ["messidor2", "messidor-2", "Messidor-2", "messidor"])
    if base is None:
        raise DatasetUnavailable(
            "Messidor-2 not found. Request access at "
            "https://www.adcis.net/en/third-party/messidor2/ and unpack under "
            f"{root}/messidor2 with its grading CSV.")

    csv_path = None
    for p in base.rglob("*.csv"):
        if any(k in p.name.lower() for k in ("grade", "label", "diagnos", "abnormal")):
            csv_path = p
            break
    if csv_path is None:
        raise DatasetUnavailable(f"Messidor-2 at {base} but no grading CSV found")

    index = _index_images(base)
    records = []
    for row in _read_csv(csv_path):
        ident = _col(row, "image_id", "image", "id", "imagename")
        grade = _col(row, "adjudicated_dr_grade", "dr_grade", "grade", "diagnosis", "level")
        if ident is None or grade in (None, ""):
            continue
        stem = Path(str(ident)).stem
        path = index.get(stem)
        if path is None:
            continue
        m = re.match(r"^(\d+)_", stem)
        pid = f"messidor2:{m.group(1)}" if m else f"messidor2:{stem}"
        records.append(Record(str(path), int(float(grade)), pid, "messidor2", None, split))
    return records


# ------------------------------------------------------------------ registry
LOADERS = {
    "aptos": load_aptos,
    "eyepacs": load_eyepacs,
    "idrid": load_idrid,
    "messidor2": load_messidor2,
}


def load(names, root=None, strict=False):
    """Load and concatenate several corpora by name.

    With strict=False a missing corpus is reported and skipped, so a run that
    has APTOS but not EyePACS still trains instead of dying at import time.
    """
    if isinstance(names, str):
        names = [names]
    records, missing = [], []
    for n in names:
        if n not in LOADERS:
            raise KeyError(f"unknown dataset '{n}'; known: {sorted(LOADERS)}")
        try:
            found = LOADERS[n](root)
            records.extend(found)
            print(f"  {n:10} {len(found):>6} images")
        except DatasetUnavailable as e:
            if strict:
                raise
            missing.append(f"{n}: {e}")
    for m in missing:
        print(f"  SKIPPED {m}")
    return records


def grade_histogram(records):
    h = [0] * 5
    for r in records:
        if 0 <= r.grade <= 4:
            h[r.grade] += 1
    return h


def describe(records):
    h = grade_histogram(records)
    total = max(sum(h), 1)
    lines = [f"{len(records)} images, {len({r.patient_id for r in records})} patients"]
    for g, c in enumerate(h):
        lines.append(f"  grade {g}: {c:>6}  ({c / total:5.1%})")
    return "\n".join(lines)
