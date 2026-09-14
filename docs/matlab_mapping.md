# MATLAB port guide

PS 26038 is a MathWorks problem statement and asks for a MATLAB-based pipeline.
This prototype is written in Python because MATLAB was not available on the
build machine, but every operator was chosen to have a direct Image Processing
Toolbox or Deep Learning Toolbox equivalent. This document is the port map.

## Function correspondence

### `core/quality.py` — capture-time gate

| Python | MATLAB |
|---|---|
| `cv2.cvtColor(..., BGR2GRAY)` | `rgb2gray` |
| `cv2.GaussianBlur(img, (0,0), s)` | `imgaussfilt(I, s)` |
| `cv2.morphologyEx(..., MORPH_CLOSE/OPEN)` | `imclose` / `imopen` |
| `cv2.connectedComponentsWithStats` | `bwconncomp` + `regionprops` |
| `cv2.erode(mask, strel)` | `imerode(mask, strel('disk', r))` |
| `np.linalg.lstsq` (illumination plane fit) | `A\z` (mldivide) |

The focus metric is a difference of Gaussians ratio and ports directly:

```matlab
g   = imgaussfilt(double(gray), 0.8);
mid = imgaussfilt(g, 2.0*scale) - imgaussfilt(g, 8.0*scale);
sharpness = std(mid(interior)) / std(g(interior)) * 1000;
```

### `core/preprocess.py` — normalisation

| Python | MATLAB |
|---|---|
| `cv2.createCLAHE(...).apply(g)` | `adapthisteq(g, 'ClipLimit', 0.01, 'NumTiles', [8 8])` |
| `cv2.resize(..., INTER_AREA)` | `imresize(..., 'bilinear')` with antialiasing on |
| `cv2.copyMakeBorder` | `padarray` |

### `core/lesions.py` — segmentation

| Python | MATLAB |
|---|---|
| `cv2.morphologyEx(..., MORPH_BLACKHAT)` | `imbothat(I, strel('disk', r))` |
| `cv2.morphologyEx(..., MORPH_TOPHAT)` | `imtophat(I, strel('disk', r))` |
| `skimage.filters.frangi` | `fibermetric(I, 'ObjectPolarity', 'dark')` |
| `cv2.fitEllipse` | `regionprops('MajorAxisLength','MinorAxisLength','Orientation')` |
| `cv2.arcLength` / `contourArea` | `regionprops('Perimeter','Area')` or `'Circularity'` |
| `cv2.spatialGradient` | `imgradientxy` |
| `cv2.minMaxLoc` | `[~, idx] = max(I(:))` |

The multi-scale top-hat is the heart of the segmenter and is arguably cleaner
in MATLAB:

```matlab
acc = zeros(size(flat));
for r = [2 4 7 11]
    acc = max(acc, imbothat(flat, strel('disk', r)));
end
```

### `core/grade.py` — classifiers

| Python | MATLAB |
|---|---|
| `HistGradientBoostingClassifier` | `fitcensemble(X, y, 'Method','LogitBoost')` |
| `torchvision.models.efficientnet_b0` | `efficientnetb0` (Deep Learning Toolbox) |
| `predict_proba` | `predict(mdl, X)` with score output |

### `core/explain.py` — explainability

| Python | MATLAB |
|---|---|
| manual Grad-CAM hooks | `gradCAM(net, img, label)` — built in, much simpler |
| `cv2.applyColorMap(..., COLORMAP_JET)` | `ind2rgb(gray2ind(cam), jet(256))` |
| `cv2.findContours` + `circle` | `bwboundaries` + `viscircles` |

### `sim/district.py` — simulation

Pure logic, no toolbox dependency. `heapq` becomes a sorted array or a
`PriorityQueue` from a `containers.Map`-backed helper; alternatively SimEvents
models this natively.

## Three places the port needs care

**1. Structuring element radius is resolution-bound.**
`DARK_SCALES = (2, 4, 7, 11)` are pixel radii at the 512 px working resolution
(~25 µm/px). If you change `preprocess.TARGET`, these must scale with it or the
detector silently stops finding microaneurysms. MATLAB's `strel('disk', r)`
also uses a different default decomposition (`'periodic line'` approximation)
than OpenCV's `MORPH_ELLIPSE` — pass `strel('disk', r, 0)` to disable the
approximation and match OpenCV's exact disc.

**2. Thresholding is MAD-based, not percentile-based.**
This is deliberate and is the single most important detail to preserve. An
earlier version used `prctile(response, 99.3)`, which forces a fixed fraction of
pixels to be declared lesions no matter what the image contains — so a perfectly
healthy retina still returned a full lesion inventory. The threshold must stay
anchored to the noise floor:

```matlab
med = median(vals);
mad = median(abs(vals - med)) * 1.4826;
thr = max(med + k*max(mad,1), floor_value);
```

**3. Coordinate systems must be replayable.**
`preprocess.prepare` records `crop_origin` and `crop_side` so that any
original-resolution annotation can be mapped into the working frame with
`map_from_original`. Dropping this and simply resizing the full frame introduces
a ~6% scale error that is invisible to the eye but destroyed lesion recall
(F1 0.23 vs 0.78) when ground truth was compared at a 4 px tolerance. Keep the
transform explicit in the MATLAB version too.

## What is NOT ported

`data/synth.py` exists to make the repo runnable without a dataset. For a
MATLAB submission backed by EyePACS/APTOS/IDRiD, it can be dropped entirely —
replace it with an `imageDatastore` over the real corpus.
