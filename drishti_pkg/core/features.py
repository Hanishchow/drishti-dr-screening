"""Lesion inventory -> fixed-length clinical feature vector.

Every feature here is a quantity an ophthalmologist would recognise and could
check by hand. That is the point: the feature-based grader is not a fallback
for want of a GPU, it is the auditable half of the system. When the CNN and
this model disagree, the disagreement itself is reportable.
"""
import numpy as np

# Order is fixed and used for both training and explanation.
FEATURE_NAMES = [
    "ma_count", "hem_count", "ex_count", "cws_count",
    "ma_area_frac", "hem_area_frac", "ex_area_frac", "cws_area_frac",
    "dark_count", "bright_count", "total_count",
    "ma_mean_size_um", "hem_mean_size_um", "ex_mean_size_um",
    "hem_max_size_um", "ex_max_size_um",
    "macula_lesion_count", "macula_ex_count", "min_ex_dist_to_macula_um",
    "quadrants_with_hem", "quadrants_with_ma",
    "vessel_density", "lesion_spatial_spread",
]

# Readable labels for the clinician-facing explanation panel.
FEATURE_LABELS = {
    "ma_count": "microaneurysms",
    "hem_count": "haemorrhages",
    "ex_count": "hard exudates",
    "cws_count": "cotton-wool spots",
    "quadrants_with_hem": "retinal quadrants containing haemorrhage",
    "quadrants_with_ma": "retinal quadrants containing microaneurysms",
    "macula_ex_count": "exudates within 1500 um of the fovea",
    "min_ex_dist_to_macula_um": "closest exudate to the fovea",
    "hem_max_size_um": "largest haemorrhage",
    "vessel_density": "visible vessel density",
}

RETINA_AREA_UM2 = np.pi * (13000.0 / 2) ** 2


def _quadrant(l, centre):
    return (0 if l.x < centre[0] else 1) + (0 if l.y < centre[1] else 2)


def extract(lesion_map, prep):
    by = {"MA": [], "HEM": [], "EX": [], "CWS": []}
    for l in lesion_map.lesions:
        by[l.kind].append(l)
    counts = {k: len(v) for k, v in by.items()}
    burden = lesion_map.burden()

    def mean_size(k):
        return float(np.mean([l.major_axis_um for l in by[k]])) if by[k] else 0.0

    def max_size(k):
        return float(np.max([l.major_axis_um for l in by[k]])) if by[k] else 0.0

    centre = (prep.green.shape[1] // 2, prep.green.shape[0] // 2)
    quad_hem = len({_quadrant(l, centre) for l in by["HEM"]})
    quad_ma = len({_quadrant(l, centre) for l in by["MA"]})

    macula_lesions = lesion_map.macula_involved()
    macula_ex = [l for l in macula_lesions if l.kind == "EX"]
    min_ex_dist = (min((l.dist_to_macula_um for l in by["EX"]), default=13000.0))

    vessel_density = float(lesion_map.vessels.sum()) / max(float(prep.mask.sum()), 1.0)

    if lesion_map.lesions:
        pts = np.array([[l.x, l.y] for l in lesion_map.lesions], np.float32)
        spread = float(np.mean(np.std(pts, axis=0))) * prep.microns_per_px
    else:
        spread = 0.0

    values = {
        "ma_count": counts["MA"],
        "hem_count": counts["HEM"],
        "ex_count": counts["EX"],
        "cws_count": counts["CWS"],
        "ma_area_frac": burden["MA"] / RETINA_AREA_UM2,
        "hem_area_frac": burden["HEM"] / RETINA_AREA_UM2,
        "ex_area_frac": burden["EX"] / RETINA_AREA_UM2,
        "cws_area_frac": burden["CWS"] / RETINA_AREA_UM2,
        "dark_count": counts["MA"] + counts["HEM"],
        "bright_count": counts["EX"] + counts["CWS"],
        "total_count": sum(counts.values()),
        "ma_mean_size_um": mean_size("MA"),
        "hem_mean_size_um": mean_size("HEM"),
        "ex_mean_size_um": mean_size("EX"),
        "hem_max_size_um": max_size("HEM"),
        "ex_max_size_um": max_size("EX"),
        "macula_lesion_count": len(macula_lesions),
        "macula_ex_count": len(macula_ex),
        "min_ex_dist_to_macula_um": min_ex_dist,
        "quadrants_with_hem": quad_hem,
        "quadrants_with_ma": quad_ma,
        "vessel_density": vessel_density,
        "lesion_spatial_spread": spread,
    }
    vector = np.array([float(values[n]) for n in FEATURE_NAMES], np.float32)
    return vector, values
