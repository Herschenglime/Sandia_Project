"""Slide 39 figures: CT slice through the seeded blister, and its label, same framing.

usage: python make_slide39.py RUN.npz OUT_DIR [--half 60] [--z0 66] [--z1 118] [--exag 20]
RUN.npz must come from demo_delamination_poc.py --save-labels.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

p = argparse.ArgumentParser()
p.add_argument("npz", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--half", type=int, default=60, help="half width of the XY window (px)")
p.add_argument("--z0", type=int, default=66)
p.add_argument("--z1", type=int, default=118)
p.add_argument("--exag", type=float, default=20.0)
a = p.parse_args()

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "axes.titlesize": 12, "axes.labelsize": 11})

d = np.load(a.npz)
recon, label, mids = d["recon_defect"], d["new_air_label"], d["material_ids"]
dx, dy, dz = d["spacing_mm"]
zi = int(d["interface_z"])
lift = d["lift_slices"]
cr, cc = np.rint(np.argwhere(lift > 0).mean(0)).astype(int)  # blister centroid
row = int(np.rint(np.argwhere(lift == lift.max())[:, 0].mean()))  # section through the peak
plan_z = zi + 1  # one slice above the interface, as in the PoC figure

h = a.half
rs = slice(max(cr - h, 0), cr + h)
cs = slice(max(cc - h, 0), cc + h)
zs = slice(a.z0, a.z1 + 1)

# Plan view: full copper range. Section: air black, FR4 mid-grey, copper white.
plan_win = np.percentile(recon[mids != 0], [1, 99])
fr4 = np.median(recon[mids == 4])
sec_win = (-0.2 * fr4, 2.8 * fr4)

plan = recon[plan_z, rs, cs]
sec = recon[zs, row, cs]
plan_lab = label[plan_z, rs, cs]
sec_lab = label[zs, row, cs]

# Physical extents (mm); z is height above the board's bottom face.
w_mm = (cs.stop - cs.start) * dx
plan_ext = (0, w_mm, (rs.stop - rs.start) * dy, 0)
sec_ext = (0, w_mm, zs.start * dz, zs.stop * dz)
row_mm = (row - rs.start + 0.5) * dy
X = "×"

red_cmap = ListedColormap([(0, 0, 0, 0), "#d62728"])


def layout():
    plan_h = plan_ext[2]
    sec_h = (sec_ext[3] - sec_ext[2]) * a.exag
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.2, 5.2 * (plan_h + sec_h) / w_mm + 1.1),
        gridspec_kw=dict(height_ratios=[plan_h, sec_h], hspace=0.32))
    return fig, ax1, ax2


def finish(fig, ax1, ax2, path, titles):
    ax1.set_title(titles[0])
    ax2.set_title(titles[1])
    for ax in (ax1, ax2):
        ax.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax2.set_ylabel("z (mm)")
    ax1.axhline(row_mm, color="#1f77b4", lw=1.2, ls="--")
    # Name the copper layers (unlifted heights) on the right of the section.
    sec_ax = ax2.secondary_yaxis("right")
    sec_ax.set_yticks([(z0 + 2) * dz for z0 in (70, 84, 98)], ["L6", "L7", "TOP"])
    sec_ax.tick_params(length=0)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


# 1. CT slice
fig, ax1, ax2 = layout()
ax1.imshow(plan, cmap="gray", vmin=plan_win[0], vmax=plan_win[1], extent=plan_ext,
           interpolation="nearest")
ax2.imshow(sec, cmap="gray", vmin=sec_win[0], vmax=sec_win[1], extent=sec_ext, origin="lower",
           aspect=a.exag, interpolation="nearest")
finish(fig, ax1, ax2, a.out / "39-flaw-injection-slice.png",
       ("CT, in-plane at L7 (simulated)", f"CT, cross-section (depth {X}{a.exag:g})"))

# 2. Label: faint CT underlay + red label
fig, ax1, ax2 = layout()
kw = dict(cmap="gray", interpolation="nearest", alpha=0.25)
ax1.imshow(plan, extent=plan_ext, vmin=plan_win[0], vmax=plan_win[1], **kw)
ax1.imshow(plan_lab, cmap=red_cmap, vmin=0, vmax=1, extent=plan_ext, interpolation="nearest")
ax2.imshow(sec, extent=sec_ext, origin="lower", aspect=a.exag, vmin=sec_win[0],
           vmax=sec_win[1], **kw)
ax2.imshow(sec_lab, cmap=red_cmap, vmin=0, vmax=1, extent=sec_ext, origin="lower",
           aspect=a.exag, interpolation="nearest")
finish(fig, ax1, ax2, a.out / "39-delamination-mask.png",
       ("Label, in-plane at L7", f"Label, cross-section (depth {X}{a.exag:g})"))
