"""
Compact, single-row slide figure of the dual-energy / Alvarez-Macovski (AM)
material-mapping capability that already exists in xrayrecon.

    Board  ->  low / high kVp scans  ->  AM decomposition  ->  Material map

Geometry: the PCB phantom from make_am_process_diagram.py (build_phantom /
MATERIALS), re-labelled with real xrayrecon materials (FR4, COPPER, SOLDER, AIR).
Physics: xrayrecon.alvarez_macovski -- the same calls the app uses
(simulate_am_projections -> reconstruct_am_projections) with the default
industrial acquisition (triangular 160 / 320 kVp spectra), log-spaced LUT
decomposition per ray, FBP of the two basis sinograms, Z_eff / rho_e, and the
material-regularized nearest-reference classification.

The notebook-era physics in make_am_process_diagram.py is NOT used: its
photoelectric basis f1(E) = E**-3 is ~1e-5 at 50 keV, so a1 carries essentially
no signal and its Z_eff panel is streak noise. xrayrecon normalizes
f1 = (50 keV / E)**3, which is why it recovers Z_eff.

Outputs (default --outdir is demo/demo_output/am_slide_diagram):
    42-am-material-pipeline.png / .svg   4-panel flow
    42-am-material-minimal.png  / .svg   2-panel fallback (board -> material map)

Run from repo root:
    uv run python demo/make_am_slide_diagram.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from skimage.transform import iradon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
# The phantom geometry comes from the notebook script, which stays in notebooks/.
sys.path.insert(0, os.path.join(REPO, "notebooks", "Spectral-Xray"))
sys.path.insert(0, os.path.join(REPO, "xrayrecon"))

import make_am_process_diagram as legacy  # noqa: E402  (phantom geometry only)
from xrayrecon.alvarez_macovski import (  # noqa: E402
    classify_material_volume,
    reconstruct_am_projections,
    simulate_am_projections,
)
from xrayrecon.am_acquisition import AMAcquisitionConfig  # noqa: E402
from xrayrecon.materials import (  # noqa: E402
    MATERIAL_NAMES,
    MATERIALS_DB,
    material_labels_to_zeff_rhoe,
)

DEFAULT_OUTDIR = os.path.join(HERE, "demo_output", "am_slide_diagram")

# legacy phantom material key -> xrayrecon MATERIALS_DB key
LEGACY_TO_DB = {"air": "AIR", "fr4": "FR4", "copper": "COPPER", "solder": "SOLDER"}

# Restrained display palette (sits on a plain navy/white template).
SHOWN = ["FR4", "COPPER", "SOLDER"]
PALETTE = {
    "AIR": "#ffffff",
    "FR4": "#c7d3bd",     # pale sage (glass-epoxy)
    "COPPER": "#c46a2a",  # copper
    "SOLDER": "#4d5d78",  # slate (Sn-based solder)
}
OTHER_COLOR = "#a8a8a8"   # any other DB material the classifier picks
LEGEND_NAME = {"FR4": "FR4", "COPPER": "Cu", "SOLDER": "Solder"}

TEXT = "#1f1f1f"
MUTED = "#555555"
ARROW = "#8a8a8a"
FRAME = "#bdbdbd"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "svg.fonttype": "path",
})


# ============================================================================
# Simulation (xrayrecon)
# ============================================================================

def phantom_labels(N: int) -> np.ndarray:
    """Legacy PCB phantom geometry as an xrayrecon material-ID image (H, W)."""
    _, a1, a2 = legacy.build_phantom(N)
    labels = np.full((N, N), MATERIAL_NAMES.index("AIR"), dtype=np.int64)
    for key, (v1, v2) in legacy.MATERIALS.items():
        labels[(a1 == v1) & (a2 == v2)] = MATERIAL_NAMES.index(LEGACY_TO_DB[key])
    return labels


MEDIAN = 3          # median-filter size (pixels) before classification
ZEFF_WEIGHT = 0.05  # Z_eff weight in classify_material_volume (rho_e weight = 1)


def run(N: int = 512, n_angles: int = 720, board_cm: float = 2.56):
    labels = phantom_labels(N)
    props = material_labels_to_zeff_rhoe(labels[None])          # (1, H, W, 2)

    cfg = AMAcquisitionConfig.industrial_default(num_angles=n_angles)
    cfg = AMAcquisitionConfig.from_dict(
        {**cfg.to_dict(), "pixel_size_cm": board_cm / N})

    proj = simulate_am_projections(props, cfg)
    res = reconstruct_am_projections(proj, output_shape=(N, N))

    theta = proj.theta_deg
    px = cfg.pixel_size_cm

    def fbp(sino):  # conventional single-energy CT image (1/cm)
        return iradon(sino, theta=theta, circle=False, output_size=N) / px

    zeff = np.asarray(res["zeff_recovered"][0])
    rhoe = np.asarray(res["rhoe_recovered"][0])
    # Material map: xrayrecon's nearest-reference classifier over the full
    # 14-material MATERIALS_DB, on lightly (3x3) median-filtered Z_eff and rho_e.
    # The app's default regularization (5x5 median on rho_e only, Z_eff weight
    # 0.001) is kept for comparison as labels_app; on this phantom it labels
    # partial-volume copper edges as solder, because it ignores Z_eff.
    labels_rec = classify_material_volume(
        median_filter(zeff, MEDIAN), median_filter(rhoe, MEDIAN),
        zeff_weight=ZEFF_WEIGHT)

    out = dict(
        labels_gt=labels,
        mu_lo=fbp(proj.low_log_zhw[0]),
        mu_hi=fbp(proj.high_log_zhw[0]),
        zeff=zeff,
        rhoe=rhoe,
        labels_rec=labels_rec,
        labels_app=np.asarray(res["material_labels_recovered"][0]),
        lo_label=cfg.low_spectrum.label,
        hi_label=cfg.high_spectrum.label,
    )
    report(out)
    return out


def report(R):
    gt, rec, app = R["labels_gt"], R["labels_rec"], R["labels_app"]
    print("Per-material recovery (raw Z_eff median, raw rho_e median, "
          "% voxels correct: figure classifier / app-default regularization):")
    for name in ["AIR"] + SHOWN:
        m = gt == MATERIAL_NAMES.index(name)
        ref = MATERIALS_DB[name]
        print(f"  {name:7s} ref Z={ref.z_eff:5.1f} rho_e={ref.electron_density:5.2f}"
              f" | rec Z={np.median(R['zeff'][m]):5.1f}"
              f" rho_e={np.median(R['rhoe'][m]):5.2f}"
              f" | correct {100 * np.mean(rec[m] == gt[m]):5.1f}%"
              f" / {100 * np.mean(app[m] == gt[m]):5.1f}%")
    other = ~np.isin(rec, [MATERIAL_NAMES.index(n) for n in ["AIR"] + SHOWN])
    print(f"  voxels labelled as some other DB material: {other.sum()} "
          f"({100 * other.mean():.2f}%)")


# ============================================================================
# Rendering helpers
# ============================================================================

def board_crop(N: int, below: float, above: float, side: float = 0.05):
    """Row/col slices cropping the square image to the board rectangle plus
    margins (phantom units; the board spans x in +-0.80, y in +-0.55)."""
    x = np.linspace(-1, 1, N)
    cols = np.where(np.abs(x) <= 0.80 + side)[0]
    rows = np.where((x >= -0.55 - below) & (x <= 0.55 + above))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def label_rgb(labels: np.ndarray) -> np.ndarray:
    lut = np.zeros((len(MATERIAL_NAMES), 3))
    for i, name in enumerate(MATERIAL_NAMES):
        lut[i] = matplotlib.colors.to_rgb(PALETTE.get(name, OTHER_COLOR))
    return lut[labels]


class Canvas:
    """Place axes in inches on a white figure."""

    def __init__(self, w_in, h_in):
        self.W, self.H = w_in, h_in
        self.fig = plt.figure(figsize=(w_in, h_in), facecolor="white")
        self.bg = self.fig.add_axes([0, 0, 1, 1])
        self.bg.set_xlim(0, w_in)
        self.bg.set_ylim(0, h_in)
        self.bg.axis("off")

    def image(self, x, y, w, h, img, frame=FRAME, **kw):
        ax = self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H])
        ax.imshow(img, origin="lower", aspect="auto", interpolation="lanczos", **kw)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(frame)
            sp.set_linewidth(0.8)
        return ax

    def tag(self, ax, text, color="white"):
        ax.text(0.035, 0.965, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=9.5, color=color)

    def arrow(self, x0, x1, y):
        self.bg.add_patch(FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.3, color=ARROW, shrinkA=0, shrinkB=0))

    def label(self, x, y, text, size=13):
        self.bg.text(x, y, text, ha="center", va="top", fontsize=size,
                     color=TEXT)

    def legend(self, xc, y, size=10.5):
        """Single centered row of swatches + names."""
        sw, gap_txt, gap_item = 0.13, 0.05, 0.16
        # approximate text widths (in) so the row can be centred
        char_w = size / 72 * 0.55
        items = [(PALETTE[n], LEGEND_NAME[n]) for n in SHOWN]
        widths = [sw + gap_txt + len(t) * char_w for _, t in items]
        total = sum(widths) + gap_item * (len(items) - 1)
        x = xc - total / 2
        for (c, t), w in zip(items, widths):
            self.bg.add_patch(Rectangle((x, y - sw / 2), sw, sw, fc=c,
                                        ec="#7a7a7a", lw=0.5))
            self.bg.text(x + sw + gap_txt, y, t, ha="left", va="center",
                         fontsize=size, color=MUTED)
            x += w + gap_item

    def save(self, path_noext, dpi=200):
        for ext in ("png", "svg"):
            p = f"{path_noext}.{ext}"
            self.fig.savefig(p, dpi=dpi, facecolor="white")
            print(f"Saved: {p}")
        plt.close(self.fig)


# ============================================================================
# Figures
# ============================================================================

def prepare_images(R):
    N = R["labels_gt"].shape[0]
    # Middle (grayscale) panels: extra headroom above the board for the tags.
    # End (colour) panels: same total height, board centred.
    rs, cs = board_crop(N, below=0.05, above=0.22)
    crop = lambda a: a[rs, cs]  # noqa: E731
    rs_e, cs_e = board_crop(N, below=0.135, above=0.135)
    crop_e = lambda a: a[rs_e, cs_e]  # noqa: E731

    board = crop(R["rhoe"]) > 0.5           # mask Z_eff to the object
    zeff = np.where(board, crop(R["zeff"]), 0.0)
    return dict(
        gt=label_rgb(crop_e(R["labels_gt"])),
        rec=label_rgb(crop_e(R["labels_rec"])),
        lo=crop(R["mu_lo"]), hi=crop(R["mu_hi"]),
        zeff=zeff, rhoe=crop(R["rhoe"]),
        aspect=(cs.stop - cs.start) / (rs.stop - rs.start),
    )


def kvp(label):
    return label.split("-")[-1].replace("kVp", "")


def pipeline_figure(R, I, outdir):
    asp = I["aspect"]
    h = 2.05                       # end-panel image height (in)
    g = 0.07                       # gap between stacked sub-images
    hs = (h - g) / 2               # stacked sub-image height
    w_end, w_mid = asp * h, asp * hs
    a = 0.40                       # arrow gap
    lm = 0.08                      # outer margin
    W = 2 * lm + 2 * w_end + 2 * w_mid + 3 * a
    y_img = 0.66
    H = y_img + h + 0.06

    c = Canvas(W, H)
    x = lm
    xs = []
    for w in (w_end, w_mid, w_mid, w_end):
        xs.append((x, w))
        x += w + a

    # 1 Board (ground-truth materials)
    c.image(xs[0][0], y_img, w_end, h, I["gt"])

    # 2 low / high kVp conventional CT images (shared gray scale per image)
    x2 = xs[1][0]
    lo_lim = np.percentile(I["lo"], [1, 99.7])
    hi_lim = np.percentile(I["hi"], [1, 99.7])
    ax = c.image(x2, y_img + hs + g, w_mid, hs, I["lo"], cmap="gray",
                 vmin=lo_lim[0], vmax=lo_lim[1])
    c.tag(ax, f"{kvp(R['lo_label'])} kVp")
    ax = c.image(x2, y_img, w_mid, hs, I["hi"], cmap="gray",
                 vmin=hi_lim[0], vmax=hi_lim[1])
    c.tag(ax, f"{kvp(R['hi_label'])} kVp")

    # 3 AM decomposition -> Z_eff (photoelectric/Compton ratio), rho_e (Compton)
    x3 = xs[2][0]
    ax = c.image(x3, y_img + hs + g, w_mid, hs, I["zeff"], cmap="gray",
                 vmin=0, vmax=70)
    c.tag(ax, r"$Z_{\mathrm{eff}}$")
    ax = c.image(x3, y_img, w_mid, hs, I["rhoe"], cmap="gray",
                 vmin=0, vmax=np.percentile(I["rhoe"], 99.7))
    c.tag(ax, r"$\rho_e$")

    # 4 Material map (xrayrecon regularized classification)
    c.image(xs[3][0], y_img, w_end, h, I["rec"])

    # arrows
    yc = y_img + h / 2
    for (x0, w0), (x1, _) in zip(xs[:-1], xs[1:]):
        c.arrow(x0 + w0 + 0.07, x1 - 0.07, yc)

    # labels
    y_lab = y_img - 0.09
    lo_k, hi_k = kvp(R["lo_label"]), kvp(R["hi_label"])
    for (x0, w0), text in zip(xs, ["Board", f"{lo_k} / {hi_k} kVp",
                                   "AM decomposition", "Material map"]):
        c.label(x0 + w0 / 2, y_lab, text)
    # one legend, under the material map (the board panel uses the same colours)
    c.legend(xs[3][0] + w_end / 2, y_lab - 0.38)

    print(f"pipeline figure: {W:.2f} x {H:.2f} in (aspect {W / H:.2f}:1)")
    c.save(os.path.join(outdir, "42-am-material-pipeline"))


def minimal_figure(R, I, outdir):
    asp = I["aspect"]
    h = 2.05
    w = asp * h
    a = 0.55
    lm = 0.08
    W = 2 * lm + 2 * w + a
    y_img = 0.66
    H = y_img + h + 0.06

    c = Canvas(W, H)
    x0, x1 = lm, lm + w + a
    c.image(x0, y_img, w, h, I["gt"])
    c.image(x1, y_img, w, h, I["rec"])
    c.arrow(x0 + w + 0.08, x1 - 0.08, y_img + h / 2)
    y_lab = y_img - 0.09
    c.label(x0 + w / 2, y_lab, "Board")
    c.label(x1 + w / 2, y_lab, "Material map (dual-energy AM)")
    c.legend(W / 2, y_lab - 0.38)

    print(f"minimal figure: {W:.2f} x {H:.2f} in (aspect {W / H:.2f}:1)")
    c.save(os.path.join(outdir, "42-am-material-minimal"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--angles", type=int, default=720)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    R = run(N=args.N, n_angles=args.angles)
    I = prepare_images(R)
    pipeline_figure(R, I, args.outdir)
    minimal_figure(R, I, args.outdir)


if __name__ == "__main__":
    main()
