"""
Classical solder-void highlighting on a real PCB CT layer, for slide figures.

Pipeline (on one depth layer of NPY/slice_data.npy):
  1. ROIs: threshold bright solder, merge nearby pads, connected components,
     bounding box + margin around each component (fully automatic).
  2. Voids: per solder blob, candidates = holes of the blob after a small
     closing (binary_fill_holes(sealed) - blob) plus dark spots enclosed in the
     joint relative to a local solder level (grey closing); drop specks, spots
     touching the joint boundary, grey component bodies and low-contrast blobs.
     Void fraction = void area / filled blob area (standard solder-void metric).

Layer/region were found by template-matching the presenter's GUI screenshot
against the volume (run with --match to redo it).

Run from the repo root:
    uv run python demo/demo_void_highlight.py
    uv run python demo/demo_void_highlight.py --match SCREENSHOT.png
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from scipy import ndimage as ndi
from skimage import measure, morphology

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "NPY" / "slice_data.npy"
OUT = Path(__file__).resolve().parent / "demo_output" / "void_highlight"

# Result of --match against "Pasted image 20261112084753.png" (r = 0.987):
# slices 158..160 (mean), no flip/rotation, screenshot scale 0.348 px/voxel.
Z_SLAB = (158, 161)  # [start, stop)
ROWS = (270, 1850)
COLS = (227, 1374)

# Detection parameters (intensities are raw uint8 CT values of the slab mean;
# board/traces ~55-100, solder ~125-210, multi-Otsu splits at ~112 and ~149).
SMOOTH_SIGMA = 1.0  # light gaussian denoise before thresholding
SOLDER_T = 115.0  # global solder threshold (in the histogram gap above the board)
MIN_SOLDER_PX = 400  # drop bright specks that are not joints
SEAL_RADIUS = 5  # per-blob binary closing before fill_holes, seals narrow void mouths
LOCAL_R = 20  # grey-closing radius: local "solder level" ignoring holes up to ~40 px
LOCAL_DELTA = 25.0  # a pixel is dark if it is this much below the local solder level
MIN_VOID_PX = 10  # ~3.5 px equivalent diameter; smaller is at noise level
MAX_VOID_FRAC_OF_BLOB = 0.35  # bigger "holes" are component bodies, not voids
VOID_MAX_MIN = 125.0  # darkest pixel of a void must be below this (rejects grey bodies)
RING_CONTRAST = 25.0  # mean(3 px ring of solder around void) - mean(void) must exceed
ROI_MERGE_PX = 12  # dilation used to group neighbouring pads into one ROI
ROI_MARGIN = 10

DPI = 200
VOID_RGBA = (1.0, 0.15, 0.0, 0.6)  # red-orange, 60% alpha
ROI_COLOR = "#00d5ff"  # cyan


def load_layer():
    v = np.load(DATA, mmap_mode="r")
    sub = np.asarray(v[ROWS[0]:ROWS[1], COLS[0]:COLS[1], Z_SLAB[0]:Z_SLAB[1]])
    return sub.astype(np.float32).mean(axis=2)


def match_screenshot(screenshot, zrange=(150, 170)):
    """Multi-scale/flip template match of a GUI screenshot against the volume."""
    from PIL import Image

    scr = np.asarray(Image.open(screenshot).convert("L")).astype(np.float32)
    v = np.load(DATA, mmap_mode="r")
    stack = np.asarray(v[:, :, zrange[0]:zrange[1]]).astype(np.float32)
    results = []
    for a in range(zrange[0], zrange[1]):
        for b in range(a, min(a + 5, zrange[1])):
            img = stack[:, :, a - zrange[0]:b - zrange[0] + 1].mean(2)
            for s in np.arange(0.30, 0.40, 0.002):
                small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                for flip in ("none", "lr", "ud", "rot180"):
                    t = {"none": scr, "lr": scr[:, ::-1], "ud": scr[::-1],
                         "rot180": scr[::-1, ::-1]}[flip]
                    t = np.ascontiguousarray(t)
                    if small.shape[0] < t.shape[0] or small.shape[1] < t.shape[1]:
                        continue
                    r = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, (x, y) = cv2.minMaxLoc(r)
                    results.append((mx, a, b, s, flip, x / s, y / s,
                                    scr.shape[1] / s, scr.shape[0] / s))
    results.sort(reverse=True)
    for mx, a, b, s, flip, x, y, w, h in results[:8]:
        print(f"r={mx:.4f} slices {a}-{b} scale {s:.3f} {flip:6s} "
              f"rows {y:.0f}:{y + h:.0f} cols {x:.0f}:{x + w:.0f}")


def to_display(img):
    lo, hi = np.percentile(img, [0.5, 99.8])
    return np.clip((img - lo) / (hi - lo), 0, 1)


def segment(img):
    """Solder blobs -> per-blob voids (filled - solder, plus enclosed dark spots)."""
    g = ndi.gaussian_filter(img, SMOOTH_SIGMA)
    solder = morphology.remove_small_objects(g > SOLDER_T, max_size=MIN_SOLDER_PX)
    k = 2 * LOCAL_R + 1
    ref = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dark_all = g < ref - LOCAL_DELTA
    blobs, _ = ndi.label(solder)
    voids = np.zeros_like(solder)
    joints = []
    pad = SEAL_RADIUS + 2
    for i, sl0 in enumerate(ndi.find_objects(blobs), start=1):
        sl = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in sl0)
        blob = blobs[sl] == i
        sealed = ndi.binary_closing(blob, morphology.disk(SEAL_RADIUS)) | blob
        filled = ndi.binary_fill_holes(sealed)
        edge = filled & ~ndi.binary_erosion(filled)
        # (a) classic: holes in the thresholded solder blob
        holes = filled & ~blob & ~edge
        if SEAL_RADIUS:  # sealed-off pixels only count where they are actually dark
            holes &= ndi.binary_fill_holes(blob) | dark_all[sl]
        # (b) dark spots inside the joint that the global threshold merged into the
        #     solder (pads vary 125-210 in brightness); drop any touching the edge
        dl, _ = ndi.label(filled & dark_all[sl])
        touching = np.unique(dl[edge])
        dark_in = (dl > 0) & ~np.isin(dl, touching[touching > 0])
        cand, nc = ndi.label(holes | dark_in)
        keep = np.zeros_like(blob)
        g_sl = g[sl]
        for j in range(1, nc + 1):
            h = cand == j
            area = h.sum()
            if area < MIN_VOID_PX or area > MAX_VOID_FRAC_OF_BLOB * filled.sum():
                continue
            ring = ndi.binary_dilation(h, iterations=3) & ~h & filled
            if not ring.any() or g_sl[h].min() > VOID_MAX_MIN:
                continue
            if g_sl[ring].mean() - g_sl[h].mean() < RING_CONTRAST:
                continue
            keep |= h
        voids[sl] |= keep
        joints.append(dict(label=i, slice=sl0, area=int(filled.sum()),
                           void_area=int(keep.sum()),
                           void_frac=float(keep.sum() / filled.sum()),
                           n_voids=int(ndi.label(keep)[1])))
    return solder, blobs, voids, joints


def auto_rois(solder):
    """Group pads that lie within ROI_MERGE_PX of each other, box them + margin,
    then merge any boxes that still overlap."""
    grouped = ndi.binary_dilation(solder, morphology.disk(ROI_MERGE_PX))
    lab, _ = ndi.label(grouped)
    H, W = solder.shape
    boxes = []
    for sl in ndi.find_objects(lab):
        d = ROI_MERGE_PX - ROI_MARGIN  # undo the dilation, then add the margin
        boxes.append([max(0, sl[0].start + d), min(H, sl[0].stop - d),
                      max(0, sl[1].start + d), min(W, sl[1].stop - d)])
    merged = True
    while merged:
        merged = False
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                A, B = boxes[a], boxes[b]
                if A[0] < B[1] and B[0] < A[1] and A[2] < B[3] and B[2] < A[3]:
                    boxes[a] = [min(A[0], B[0]), max(A[1], B[1]),
                                min(A[2], B[2]), max(A[3], B[3])]
                    del boxes[b]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in sorted(boxes)]


def new_fig(h_px, w_px, scale=1.0):
    fig = plt.figure(figsize=(w_px * scale / DPI, h_px * scale / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    return fig, ax


def show_gray(ax, disp, interp="nearest"):
    ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation=interp)
    ax.set_xlim(-0.5, disp.shape[1] - 0.5)
    ax.set_ylim(disp.shape[0] - 0.5, -0.5)


def draw_voids(ax, voids, lw=0.0):
    padded = np.pad(voids.astype(float), 1)
    for cnt in measure.find_contours(padded, 0.5):
        xy = np.c_[cnt[:, 1] - 1, cnt[:, 0] - 1]
        ax.add_patch(Polygon(xy, closed=True, facecolor=VOID_RGBA,
                             edgecolor=VOID_RGBA[:3], lw=lw))


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, pad_inches=0)
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", help="GUI screenshot to locate in the volume, then exit")
    p.add_argument("--zoom-roi", type=int, default=None,
                   help="index of the ROI to zoom on (default: most voided)")
    args = p.parse_args()
    if args.match:
        match_screenshot(args.match)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    img = load_layer()
    disp = to_display(img)
    solder, blobs, voids, joints = segment(img)
    rois = auto_rois(solder)
    H, W = img.shape

    # per-ROI void stats (aggregate the joints whose blob centroid lies in the ROI)
    roi_stats = []
    for k, (r0, r1, c0, c1) in enumerate(rois):
        js = [j for j in joints
              if r0 <= (j["slice"][0].start + j["slice"][0].stop) / 2 < r1
              and c0 <= (j["slice"][1].start + j["slice"][1].stop) / 2 < c1]
        a = sum(j["area"] for j in js)
        va = sum(j["void_area"] for j in js)
        roi_stats.append(dict(k=k, joints=js, frac=va / a if a else 0.0, void_area=va))

    print(f"layer z={Z_SLAB[0]}..{Z_SLAB[1] - 1} rows {ROWS} cols {COLS}; "
          f"{len(joints)} solder blobs, {len(rois)} ROIs")
    for s in roi_stats:
        r0, r1, c0, c1 = rois[s["k"]]
        print(f"ROI {s['k']:2d} rows {r0:4d}:{r1:4d} cols {c0:4d}:{c1:4d} "
              f"void {100 * s['frac']:5.1f}%  per joint: "
              + ", ".join(f"{100 * j['void_frac']:.1f}%({j['n_voids']})"
                          for j in s["joints"]))

    # 0. clean render
    fig, ax = new_fig(H, W)
    show_gray(ax, disp)
    save(fig, "47-real-scan-slice.png")

    # 1. auto ROIs
    fig, ax = new_fig(H, W)
    show_gray(ax, disp)
    for r0, r1, c0, c1 in rois:
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False,
                               edgecolor=ROI_COLOR, lw=1.2))
    save(fig, "47-step1-roi.png")

    # 2a. full view with voids
    fig, ax = new_fig(H, W)
    show_gray(ax, disp)
    draw_voids(ax, voids)
    save(fig, "47-step2-voids-full.png")

    # 2b. zoom on one ROI
    k = args.zoom_roi
    if k is None:
        k = max(roi_stats, key=lambda s: s["void_area"])["k"]
    r0, r1, c0, c1 = rois[k]
    zd, zv = disp[r0:r1, c0:c1], voids[r0:r1, c0:c1]
    scale = 1000 / max(zd.shape)  # ~1000 px on the long side
    fig, ax = new_fig(*zd.shape, scale=scale)
    show_gray(ax, zd, interp="bicubic")
    draw_voids(ax, zv, lw=0.6)
    save(fig, "47-step2-voids.png")

    # 2c. side by side
    gap = 0.03
    h_in, w_in = zd.shape[0] * scale / DPI, zd.shape[1] * scale / DPI
    fig = plt.figure(figsize=(2 * w_in * (1 + gap / 2), h_in), dpi=DPI)
    fw = 1 / (2 + gap)
    for i, overlay in enumerate((False, True)):
        ax = fig.add_axes([i * fw * (1 + gap), 0, fw, 1])
        ax.set_axis_off()
        show_gray(ax, zd, interp="bicubic")
        if overlay:
            draw_voids(ax, zv, lw=0.6)
    save(fig, "47-step2-voids-pair.png")
    print(f"zoom ROI {k}: rows {r0}:{r1} cols {c0}:{c1}, "
          f"void fraction {100 * roi_stats[k]['frac']:.1f}%")


if __name__ == "__main__":
    main()
