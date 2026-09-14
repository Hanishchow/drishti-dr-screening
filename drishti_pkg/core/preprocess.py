"""Geometric and photometric normalisation.

Every downstream stage assumes a square image with the retinal disc centred and
tangent to the border, so lesion sizes can be reported in microns via a single
scale factor.

MATLAB equivalents: imcrop, imresize, adapthisteq, imopen.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .quality import field_of_view_mask

TARGET = 512
# A typical 45-degree fundus field spans ~13000 um across the retinal disc.
FOV_WIDTH_MICRONS = 13000.0


@dataclass
class Prepared:
    bgr: np.ndarray        # normalised colour image, TARGET x TARGET
    green: np.ndarray      # CLAHE-equalised green channel, uint8
    mask: np.ndarray       # retinal FOV, uint8 {0,1}
    microns_per_px: float
    # Geometry of the crop that produced this image, in ORIGINAL image
    # coordinates: (x0, y0, side). Anything that needs to compare against the
    # original frame -- ground-truth masks, clinician annotations, a prior
    # visit's lesion map -- must go through the same transform, or it lands a
    # few percent off and silently mismatches.
    crop_origin: tuple = (0, 0)
    crop_side: int = TARGET

    @property
    def um2_per_px(self):
        return self.microns_per_px ** 2

    def map_from_original(self, arr, nearest=True):
        """Bring an original-resolution array into this image's coordinates."""
        x0, y0 = self.crop_origin
        side = self.crop_side
        pad = side  # generous, matches the padding used during the crop
        padded = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        sub = padded[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        return cv2.resize(sub, (TARGET, TARGET), interpolation=interp)

    def map_point_from_original(self, x, y):
        x0, y0 = self.crop_origin
        s = TARGET / float(self.crop_side)
        return int(round((x - x0) * s)), int(round((y - y0) * s))


def _crop_to_fov(bgr, mask):
    """Square crop tight to the retinal disc. Returns the crop plus its origin
    and side in ORIGINAL coordinates, so the transform can be replayed."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return bgr, mask, (0, 0), max(bgr.shape[:2])
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = side // 2
    ox, oy = int(cx - half), int(cy - half)
    # Pad rather than clamp, so an off-centre retina stays centred after crop.
    pad = side
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    pmask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    crop = padded[oy + pad:oy + pad + side, ox + pad:ox + pad + side]
    cmask = pmask[oy + pad:oy + pad + side, ox + pad:ox + pad + side]
    return crop, cmask, (ox, oy), side


def prepare(bgr) -> Prepared:
    mask = field_of_view_mask(bgr)
    crop, cmask, origin, side = _crop_to_fov(bgr, mask)
    if crop.size == 0:
        crop, cmask, origin, side = bgr, mask, (0, 0), max(bgr.shape[:2])

    crop = cv2.resize(crop, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
    cmask = cv2.resize(cmask, (TARGET, TARGET), interpolation=cv2.INTER_NEAREST)
    # Erode slightly: the vignette boundary otherwise reads as a huge dark lesion.
    cmask = cv2.erode(cmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    green = crop[:, :, 1]
    green = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(green)
    green = cv2.bitwise_and(green, green, mask=cmask)

    microns_per_px = FOV_WIDTH_MICRONS / TARGET
    return Prepared(bgr=crop, green=green, mask=cmask,
                    microns_per_px=microns_per_px,
                    crop_origin=origin, crop_side=side)


def fill_outside_fov(green, mask):
    """Replace the black surround with retinal-like intensity.

    Without this, every morphological operator sees a step edge of ~150 grey
    levels at the FOV rim and returns a response far larger than any lesion,
    which swamps the detector's dynamic range. Mirror-style inpainting makes the
    rim morphologically invisible.
    """
    inside = green[mask > 0]
    fill = float(np.median(inside)) if inside.size else 0.0
    out = green.astype(np.float32)
    out[mask == 0] = fill
    # Smooth across the seam so the fill does not itself create an edge.
    blurred = cv2.GaussianBlur(out, (0, 0), 6)
    seam = cv2.dilate(1 - mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    seam = (seam > 0) & (mask > 0)
    out[mask == 0] = blurred[mask == 0]
    out[seam] = blurred[seam]
    return np.clip(out, 0, 255).astype(np.uint8)


def flatten_illumination(green, mask, sigma=25):
    """Remove the slow background gradient so top-hat responses are comparable
    between the bright posterior pole and the dim periphery."""
    filled = fill_outside_fov(green, mask).astype(np.float32)
    bg = cv2.GaussianBlur(filled, (0, 0), sigma)
    ref = float(np.median(filled[mask > 0])) if (mask > 0).any() else 0.0
    return np.clip(filled - bg + ref, 0, 255).astype(np.uint8)
