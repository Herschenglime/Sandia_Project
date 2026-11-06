"""
Delamination proof of concept: inject an open delamination into a voxelized
PCB, simulate CT of the clean and defective boards, and save a slide figure.

Pipeline:
  1. Voxelize a Gerber stackup (default: PCBCT picture1). By default the
     non-trace area of each copper layer is filled with FR4 (resin), as in
     configs/jonathan_stackup.json, so the seeded gap is the only air inside
     the board.
  2. Pick the FR4 substrate slab directly above a copper layer (default: the
     L5-GND plane) and, inside an irregular 2D blob, replace the bottom
     ``--gap-slices`` FR4 slices with AIR. Only FR4 voxels are changed, and
     the boolean ground-truth mask is kept.
  3. Run ``simulate_ct_volume`` on the full Z stack (or ``--z-range local``)
     for both volumes, over a small XY crop centred on the defect
     (``--crop 0`` for the full board).
  4. Save a composite figure plus one PNG per panel, and an .npz that the
     standalone 3D viewer (``--view-3d``) can reopen.

Run from the xrayrecon/ directory:
    python demo_delamination_poc.py                      # 256x256 CPU default
    python demo_delamination_poc.py --backend cuda --crop 0
    python demo_delamination_poc.py --view-3d demo_output/delamination_poc/<run>.npz
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from xrayrecon.materials import MATERIAL_NAMES, MATERIALS_DB

DEFAULT_STACKUP = Path(
    "/home/paul/dev/Sandia_Project/CAD_CT_Dataset_PCBCT/picture1/picture1_stackup.json"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "demo_output" / "delamination_poc"

AIR_ID = MATERIAL_NAMES.index("AIR")
COPPER_ID = MATERIAL_NAMES.index("COPPER")
FR4_ID = MATERIAL_NAMES.index("FR4")

# One colour per meaning, shared by every panel and the 3D view.
GAP_RGB = (0.90, 0.10, 0.10)
COPPER_RGB = (1.00, 0.55, 0.00)  # MATERIAL_COLORS["COPPER"]
FR4_RGB = (0.80, 0.75, 0.55)
AIR_RGB = (0.05, 0.05, 0.05)
SECTION_RGB = (0.00, 0.63, 1.00)
ZOOM_ASPECT = 2.0  # display stretch of the zoomed XZ panels, on top of the voxel aspect


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stackup", type=Path, default=DEFAULT_STACKUP)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu",
                   help="projection/backprojection backend for simulate_ct_volume")
    p.add_argument("--layer", default="picture-1-L5-GND",
                   help="copper layer whose FR4 substrate slab (directly above it) delaminates")
    p.add_argument("--gap-slices", type=int, default=3,
                   help="air gap thickness in Z slices (x resolution_z, e.g. 3 -> 30 um)")
    p.add_argument("--blob-radius-mm", type=float, default=3.0,
                   help="nominal blob radius in mm before the random perturbation")
    p.add_argument("--blob-roughness", type=float, default=0.35,
                   help="0 = ellipse, larger = more irregular outline")
    p.add_argument("--center", type=int, nargs=2, metavar=("ROW", "COL"), default=None,
                   help="defect centre in board pixels (default: auto, on solid copper near traces)")
    p.add_argument("--crop", type=int, default=256,
                   help="XY crop size (px) around the defect for simulation; 0 = full board")
    p.add_argument("--copper-fill", choices=["fr4", "air"], default="fr4",
                   help="material between traces on copper layers (the stackup JSON says air)")
    p.add_argument("--z-range", choices=["full", "local"], default="full",
                   help="simulate the whole stack, or only the copper layers around the defect")
    p.add_argument("--z-margin", type=int, default=4,
                   help="with --z-range local: extra slices beyond the copper layers on either side")
    p.add_argument("--num-angles", type=int, default=None,
                   help="projection angles (default: max(H, W) of the simulated crop)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", action="store_true",
                   help="also run the naive threshold + morphology baseline (stretch goal)")
    p.add_argument("--view-3d", type=Path, default=None, metavar="NPZ",
                   help="open a saved run in a standalone 3D window instead of running the pipeline")
    return p.parse_args()


def irregular_blob(shape, center, radius_px, roughness, rng):
    """One connected, hole-free blob: an ellipse whose radius is perturbed by
    a smoothed random field, so the outline is irregular but not fragmented."""
    rows, cols = np.indices(shape, dtype=np.float32)
    aspect = rng.uniform(0.7, 1.0)
    angle = rng.uniform(0, np.pi)
    dr, dc = rows - center[0], cols - center[1]
    u = dr * np.cos(angle) + dc * np.sin(angle)
    v = (-dr * np.sin(angle) + dc * np.cos(angle)) / aspect
    r_norm = np.hypot(u, v) / radius_px

    noise = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32),
                                sigma=max(radius_px / 3, 1))
    noise /= noise.std() + 1e-9
    blob = r_norm < 1.0 + roughness * noise

    labels, n = ndi.label(blob)
    if n == 0:
        raise ValueError("Blob came out empty; increase --blob-radius-mm")
    center_label = labels[center[0], center[1]]
    if center_label == 0:
        center_label = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndi.binary_fill_holes(labels == center_label)


def auto_center(material_ids, plane_z, trace_z, half):
    """Centre the defect on solid plane copper, in a window with many traces
    on the neighbouring signal layer, so the crop shows some structure."""
    plane = (material_ids[plane_z] == COPPER_ID).astype(np.float32)
    traces = (material_ids[trace_z] == COPPER_ID).astype(np.float32)
    edges = np.abs(np.diff(traces, axis=0, prepend=0)) + np.abs(np.diff(traces, axis=1, prepend=0))
    score = ndi.uniform_filter(plane, 96) * ndi.uniform_filter(edges, 2 * half)
    h, w = plane.shape
    valid = np.zeros_like(score, dtype=bool)
    valid[half:h - half, half:w - half] = True  # keep the crop inside the board
    if valid.any():
        score[~valid] = -1
    return np.unravel_index(np.argmax(score), score.shape)


def crop_window(center, size, extent):
    if size <= 0 or size >= extent:
        return slice(0, extent)
    start = int(np.clip(center - size // 2, 0, extent - size))
    return slice(start, start + size)


def naive_baseline(recon, fr4_mask, k=4.0, min_voxels=50):
    """Threshold well below typical FR4 inside the design FR4 mask, clean up
    with a 3D open then close, and keep large connected components."""
    fr4_vals = recon[fr4_mask]
    med = np.median(fr4_vals)
    sigma = 1.4826 * np.median(np.abs(fr4_vals - med))  # robust std
    thresh = med - k * sigma
    cand = (recon < thresh) & fr4_mask
    st = ndi.generate_binary_structure(3, 1)
    cand = ndi.binary_opening(cand, structure=st)
    cand = ndi.binary_closing(cand, structure=st) & fr4_mask
    labels, n = ndi.label(cand, structure=st)
    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero(sizes >= min_voxels)
    keep = keep[keep != 0]
    return np.isin(labels, keep), dict(median=float(med), sigma=float(sigma),
                                       threshold=float(thresh), components=int(len(keep)))


def dice(a, b):
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else 1.0


def material_rgb(labels, gap):
    rgb = np.empty(labels.shape + (3,), dtype=np.float32)
    rgb[:] = AIR_RGB
    rgb[labels == FR4_ID] = FR4_RGB
    rgb[labels == COPPER_ID] = COPPER_RGB
    rgb[gap] = GAP_RGB
    return rgb


def material_rgba_3d(labels, gap):
    """(Z, H, W) labels -> uint8 RGBA for GLVolumeItem: copper translucent
    orange, FR4 barely visible, air invisible, seeded gap opaque red."""
    rgba = np.zeros(labels.shape + (4,), dtype=np.uint8)
    rgba[labels == FR4_ID] = (205, 190, 140, 2)
    rgba[labels == COPPER_ID] = (255, 140, 0, 25)
    rgba[gap] = (255, 30, 30, 255)
    return rgba


def view_3d(npz_path):
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    d = np.load(npz_path)
    rgba = material_rgba_3d(d["material_ids"], d["gt_mask"])
    # (Z, H, W) -> GL (x = W, y = H flipped so it matches the top-view figure, z = Z)
    vol = np.ascontiguousarray(rgba[:, ::-1].transpose(2, 1, 0, 3))
    nx, ny, nz = vol.shape[:3]
    spacing = d["spacing_mm"]

    app = pg.mkQApp("Delamination 3D")
    view = gl.GLViewWidget()
    view.setWindowTitle(f"Seeded delamination (red), Z exaggerated {spacing[0] / spacing[2]:.0f}x"
                        f" - {Path(npz_path).name}")
    view.setBackgroundColor((30, 30, 34))
    item = gl.GLVolumeItem(vol, smooth=False)
    item.translate(-nx / 2, -ny / 2, -nz / 2)
    view.addItem(item)
    grid = gl.GLGridItem()
    grid.setSize(nx, ny)
    grid.setSpacing(16, 16)
    grid.translate(0, 0, -nz / 2)
    view.addItem(grid)
    view.setCameraPosition(distance=1.8 * max(nx, ny), elevation=25, azimuth=-60)
    view.resize(1200, 800)
    view.show()
    pg.exec()


# ── Figure ──────────────────────────────────────────────────────────────────

def _xz_extent(x0, x1, z0, z1):
    return (x0 - 0.5, x1 - 0.5, z0 - 0.5, z1 - 0.5)


def _z_ticks(ax, S, z0, z1, with_gap=False):
    ticks = [(z, name) for z, name in S.cu_ticks if z0 <= z < z1]
    if with_gap:
        ticks.append((S.gap_mid, "gap"))
    ticks.sort()
    ax.set_yticks([z for z, _ in ticks])
    ax.set_yticklabels([name for _, name in ticks], fontsize=8)
    for lbl, (_, name) in zip(ax.get_yticklabels(), ticks):
        if name == "gap":
            lbl.set_color(GAP_RGB)
            lbl.set_fontweight("bold")


def draw_where(ax, S):
    import matplotlib.patches as mpatches

    img = np.empty(S.board_plane.shape + (3,), dtype=np.float32)
    img[:] = FR4_RGB
    img[S.board_plane] = np.array(FR4_RGB) * 0.85
    img[S.board_traces] = COPPER_RGB
    img[S.board_blob] = GAP_RGB
    ax.imshow(img, interpolation="nearest")
    rs, cs = S.rows, S.cols
    ax.add_patch(mpatches.Rectangle((cs.start - 0.5, rs.start - 0.5), cs.stop - cs.start,
                                    rs.stop - rs.start, fill=False, ec="black", lw=1.5, ls="--"))
    ax.plot([cs.start, cs.stop - 1], [S.row_global] * 2, color=SECTION_RGB, lw=1.2)
    bar = 10 / S.spacing[0]
    h = img.shape[0]
    ax.plot([10, 10 + bar], [h - 14] * 2, color="black", lw=3)
    ax.text(10 + bar / 2, h - 22, "10 mm", ha="center", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("A. Where: board top view", fontsize=12, fontweight="bold", loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=COPPER_RGB, label=f"{S.trace_name} traces (above)"),
        mpatches.Patch(color=np.array(FR4_RGB) * 0.85, label=f"{S.plane_name} plane (below)"),
        mpatches.Patch(color=GAP_RGB, label="seeded air gap"),
        mpatches.Patch(fill=False, ec="black", ls="--", label="simulated crop"),
        mpatches.Patch(color=SECTION_RGB, label="cross-section line"),
    ], loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, fontsize=7, frameon=False)


def draw_model_xz(ax, S, zoom=False):
    sec = material_rgb(S.labels[:, S.row, :], S.gt[:, S.row, :])
    x0, x1, z0, z1 = S.zoom if zoom else (0, sec.shape[1], 0, sec.shape[0])
    ax.imshow(sec[z0:z1, x0:x1], origin="lower", interpolation="nearest",
              extent=_xz_extent(x0, x1, z0, z1), aspect=ZOOM_ASPECT if zoom else 1.0)
    _z_ticks(ax, S, z0, z1, with_gap=zoom)
    ax.set_xticks([])
    if zoom:
        ax.set_title(f"B'. Zoom: {S.gap_um:.0f} um air layer (Z x{S.exag * ZOOM_ASPECT:.0f})",
                     fontsize=11, fontweight="bold", loc="left")
    else:
        _zoom_box(ax, S)
        ax.set_title(f"B. What we seeded: voxel model, XZ section (Z x{S.exag:.0f})",
                     fontsize=12, fontweight="bold", loc="left")


def _zoom_box(ax, S):
    import matplotlib.patches as mpatches

    x0, x1, z0, z1 = S.zoom
    ax.add_patch(mpatches.Rectangle((x0 - 0.5, z0 - 0.5), x1 - x0, z1 - z0,
                                    fill=False, ec="white", lw=1.2, ls="--"))


def draw_ct_xz(ax, S, data, zoom=False, contour=False, title=None):
    x0, x1, z0, z1 = S.zoom if zoom else (0, data.shape[2], 0, data.shape[0])
    sec = data[z0:z1, S.row, x0:x1]
    ax.imshow(sec, cmap="gray", vmin=S.vmin, vmax=S.vmax, origin="lower", interpolation="nearest",
              extent=_xz_extent(x0, x1, z0, z1), aspect=ZOOM_ASPECT if zoom else 1.0)
    if contour:
        gt = S.gt[z0:z1, S.row, x0:x1].astype(float)
        if gt.any():
            ax.contour(np.arange(x0, x1), np.arange(z0, z1), gt, levels=[0.5],
                       colors=[GAP_RGB], linestyles="--", linewidths=1.0)
    _z_ticks(ax, S, z0, z1, with_gap=zoom)
    ax.set_xticks([])
    if not zoom:
        _zoom_box(ax, S)
    if title:
        ax.set_title(title, fontsize=11 if zoom else 12, fontweight="bold", loc="left")


def draw_ct_plan(ax, S):
    ax.imshow(S.defect[S.z_gap], cmap="gray", vmin=S.vmin, vmax=S.vmax, interpolation="nearest")
    traces = (S.labels[S.trace_z] == COPPER_ID).astype(float)
    ax.contour(traces, levels=[0.5], colors=[COPPER_RGB], linewidths=0.5, alpha=0.4)
    ax.contour(S.gt[S.z_gap].astype(float), levels=[0.5], colors=[GAP_RGB],
               linestyles="--", linewidths=1.2)
    if S.pred is not None:
        ax.contour(S.pred[S.z_gap].astype(float), levels=[0.5], colors=["#30b0ff"], linewidths=1.0)
    ax.axhline(S.row, color=SECTION_RGB, lw=1)
    ax.set_xticks([]); ax.set_yticks([])
    title = "D. Simulated CT at gap depth (defective)"
    if S.pred is not None:
        title += f"\n    baseline in blue, Dice {S.dice:.2f}"
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.text(0.02, 0.02, f"orange: {S.trace_name} trace outlines\nred dashed: ground truth\n"
            "clean board here = uniform FR4", transform=ax.transAxes, fontsize=7,
            color="white", va="bottom", bbox=dict(fc="black", alpha=0.5, lw=0))


def make_figures(stem, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = (f"Seeded open delamination: {S.gap_um:.0f} um air gap between the {S.plane_name} plane "
             f"and FR4, ~{S.blob_area_mm2:.0f} mm$^2$ (picture1, synthetic)")
    footer = ("Gerber -> voxel model -> slice-wise parallel-beam FBP (skimage radon/iradon), no noise. "
              f"Copper-layer gaps filled with {S.copper_fill.upper()}. Voxel "
              f"{S.spacing[0] * 1000:.0f} x {S.spacing[1] * 1000:.0f} x {S.spacing[2] * 1000:.0f} um. "
              "Single sample.\nBlack/white banding inside signal-copper layers is FBP streaking "
              "from dense copper, not air; the seeded gap is the only air in the model.")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.75, 1.0], left=0.07, right=0.99,
                          top=0.89, bottom=0.09, wspace=0.22, hspace=0.32)
    draw_where(fig.add_subplot(gs[0, 0]), S)
    draw_model_xz(fig.add_subplot(gs[0, 1]), S)
    draw_model_xz(fig.add_subplot(gs[0, 2]), S, zoom=True)
    draw_ct_plan(fig.add_subplot(gs[1, 0]), S)
    draw_ct_xz(fig.add_subplot(gs[1, 1]), S, S.defect,
               title=f"C. What CT sees: simulated CT, same section (Z x{S.exag:.0f})")
    sub = gs[1, 2].subgridspec(2, 1, hspace=0.35)
    draw_ct_xz(fig.add_subplot(sub[0]), S, S.clean, zoom=True, title="C'. Zoom: clean board")
    draw_ct_xz(fig.add_subplot(sub[1]), S, S.defect, zoom=True, contour=True,
               title="C''. Zoom: delaminated board")
    fig.text(0.012, 0.70, "VOXEL MODEL\n(what we seeded)", rotation=90, va="center", ha="center",
             fontsize=12, fontweight="bold", color="#444444")
    fig.text(0.012, 0.28, "SIMULATED CT\n(what a scan shows)", rotation=90, va="center",
             ha="center", fontsize=12, fontweight="bold", color="#444444")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.012, footer, ha="center", fontsize=9, color="#555555")
    paths = [stem.with_name(stem.name + ".png")]
    fig.savefig(paths[0], dpi=200)
    plt.close(fig)

    panels = {
        "A_where": ((6, 6.5), [lambda ax: draw_where(ax, S)]),
        "B_model_xz": ((10, 5), [lambda ax: draw_model_xz(ax, S)]),
        "B_model_zoom": ((6, 4), [lambda ax: draw_model_xz(ax, S, zoom=True)]),
        "C_ct_xz": ((10, 5), [lambda ax: draw_ct_xz(ax, S, S.defect, title="Simulated CT, XZ section")]),
        "C_ct_zoom": ((6, 6), [
            lambda ax: draw_ct_xz(ax, S, S.clean, zoom=True, title="Clean board"),
            lambda ax: draw_ct_xz(ax, S, S.defect, zoom=True, contour=True, title="Delaminated board"),
        ]),
        "D_ct_plan": ((6, 6), [lambda ax: draw_ct_plan(ax, S)]),
    }
    for name, (size, drawers) in panels.items():
        fig, axes = plt.subplots(len(drawers), 1, figsize=size, squeeze=False)
        for ax, draw in zip(axes[:, 0], drawers):
            draw(ax)
        fig.tight_layout()
        paths.append(stem.with_name(f"{stem.name}_panel{name}.png"))
        fig.savefig(paths[-1], dpi=200)
        plt.close(fig)
    return paths


# ── Pipeline ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.view_3d is not None:
        view_3d(args.view_3d)
        return

    from types import SimpleNamespace
    from xrayrecon.ct_simulation import simulate_ct_volume
    from xrayrecon.stackup import StackupConfig
    from xrayrecon.stackup_voxelize import voxelize_stackup_config

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = perf_counter()

    # 1. Voxelize: vol is (Z, H, W, 2) = [z_eff, rel. electron density], Z
    # runs bottom -> top in stackup order; material_ids is (Z, H, W).
    config = StackupConfig.from_json(args.stackup)
    if args.copper_fill == "fr4":
        for layer_cfg in config.layers:
            layer_cfg.background_material = "FR4"
    vol, layer_names, layer_slices, _, spacing, material_ids = voxelize_stackup_config(
        config, args.stackup, return_material_ids=True)
    t_vox = perf_counter() - t0
    print(f"Voxelized {args.stackup.name}: vol {vol.shape}, spacing (x, y, z) mm {spacing}, {t_vox:.1f} s")

    slab_name = f"{args.layer}_substrate"
    if slab_name not in layer_names:
        raise ValueError(f"No substrate slab above '{args.layer}'. Layers: {layer_names}")
    li = layer_names.index(args.layer)
    cu_slices = layer_slices[li]
    slab = layer_slices[layer_names.index(slab_name)]
    next_cu = layer_slices[li + 2] if li + 2 < len(layer_slices) else slab[-1:]
    if not 1 <= args.gap_slices <= len(slab):
        raise ValueError(f"--gap-slices must be within 1..{len(slab)}")
    gap_z = slab[:args.gap_slices]  # the FR4 slices touching the copper layer

    # 2. Defect: irregular blob, FR4 voxels only.
    half = (args.crop or max(vol.shape[1:3])) // 2
    center = tuple(args.center) if args.center else auto_center(
        material_ids, cu_slices[0], next_cu[0], min(half, min(vol.shape[1:3]) // 2))
    radius_px = args.blob_radius_mm / spacing[0]
    blob2d = irregular_blob(vol.shape[1:3], center, radius_px, args.blob_roughness, rng)

    gt = np.zeros(material_ids.shape, dtype=bool)
    gt[gap_z] = blob2d[None]
    gt &= material_ids == FR4_ID
    air = MATERIALS_DB["AIR"]
    vol_def = vol.copy()
    vol_def[gt, 0] = air.z_eff
    vol_def[gt, 1] = air.electron_density
    mids_def = material_ids.copy()
    mids_def[gt] = AIR_ID

    # 3. Simulate a Z range and XY crop around the defect.
    if args.z_range == "full":
        z0, z1 = 0, vol.shape[0]
    else:
        z0 = max(0, cu_slices[0] - args.z_margin)
        z1 = min(vol.shape[0], next_cu[-1] + 1 + args.z_margin)
    rs = crop_window(center[0], args.crop, vol.shape[1])
    cs = crop_window(center[1], args.crop, vol.shape[2])
    sub = (slice(z0, z1), rs, cs)
    print(f"Defect centre (row, col) {center}, gap slices {gap_z[0]}..{gap_z[-1]}, "
          f"{int(gt.sum())} voxels; simulating Z {z0}..{z1 - 1}, rows {rs.start}..{rs.stop - 1}, "
          f"cols {cs.start}..{cs.stop - 1} on {args.backend}")

    t1 = perf_counter()
    recon_clean = simulate_ct_volume(vol[sub], num_angles=args.num_angles,
                                     projection_backend=args.backend, action_name="delam_clean")
    recon_def = simulate_ct_volume(vol_def[sub], num_angles=args.num_angles,
                                   projection_backend=args.backend, action_name="delam_defect")
    t_sim = perf_counter() - t1
    gt_sub = gt[sub]
    labels_sub = mids_def[sub]
    fr4_sub = material_ids[sub] == FR4_ID

    # 4. Optional naive baseline.
    pred = dice_val = base_info = None
    if args.baseline:
        pred, base_info = naive_baseline(recon_def, fr4_sub)
        dice_val = dice(pred, gt_sub)
        base_info.update(dice=dice_val, detected=bool((pred & gt_sub).any()),
                         false_positive_voxels=int((pred & ~gt_sub).sum()))
        print(f"Baseline: {base_info}")

    # 5. Save data, then figures.
    short = lambda n: n.replace("picture-1-", "")
    params = dict(
        stackup=str(args.stackup), layer=args.layer, copper_fill=args.copper_fill,
        gap_slices=args.gap_slices, gap_um=args.gap_slices * spacing[2] * 1000,
        gap_z_global=[int(gap_z[0]), int(gap_z[-1])],
        blob_radius_mm=args.blob_radius_mm, blob_roughness=args.blob_roughness,
        blob_area_mm2=float(blob2d.sum() * spacing[0] * spacing[1]), seed=args.seed,
        center_rc=[int(center[0]), int(center[1])], crop=args.crop, z_range=args.z_range,
        sim_z=[z0, z1], sim_rows=[rs.start, rs.stop], sim_cols=[cs.start, cs.stop],
        sim_shape_zhw=list(recon_clean.shape), spacing_mm_xyz=list(spacing),
        backend=args.backend, num_angles=args.num_angles or max(recon_clean.shape[1:]),
        voxelize_s=round(t_vox, 1), simulate_s=round(t_sim, 1), baseline=base_info,
    )
    tag = f"gap{args.gap_slices}_r{args.blob_radius_mm:g}_seed{args.seed}_{args.backend}"
    if args.crop == 0:
        tag += "_full"
    if args.copper_fill == "air":
        tag += "_cuair"
    stem = args.out / f"delamination_{tag}"
    npz_path = stem.with_suffix(".npz")
    np.savez_compressed(npz_path, recon_clean=recon_clean, recon_defect=recon_def,
                        gt_mask=gt_sub, fr4_mask=fr4_sub, material_ids=labels_sub,
                        spacing_mm=np.asarray(spacing), origin_zrc=np.array([z0, rs.start, cs.start]))
    stem.with_suffix(".json").write_text(json.dumps(params, indent=2, default=lambda o: o.item()))

    trace_name = layer_names[li + 2] if li + 2 < len(layer_names) else args.layer
    fr4_level = float(np.median(recon_clean[fr4_sub]))
    gap_lo, gap_hi = int(gap_z[0] - z0), int(gap_z[-1] - z0)
    row = int(center[0] - rs.start)
    col = int(center[1] - cs.start)
    zoom_half = int(round(4.5 / spacing[0]))  # +-4.5 mm around the defect centre
    S = SimpleNamespace(
        labels=labels_sub, gt=gt_sub, clean=recon_clean, defect=recon_def, pred=pred, dice=dice_val,
        row=row, row_global=int(center[0]), rows=rs, cols=cs, spacing=spacing,
        z_gap=(gap_lo + gap_hi) // 2, gap_mid=(gap_lo + gap_hi) / 2, trace_z=int(next_cu[0] - z0),
        zoom=(max(0, col - zoom_half), min(recon_clean.shape[2], col + zoom_half),
              max(0, int(cu_slices[0] - z0) - 4), min(recon_clean.shape[0], int(next_cu[-1] - z0) + 5)),
        cu_ticks=[((sl[0] + sl[-1]) / 2 - z0, short(n) + " Cu") for n, sl in zip(layer_names, layer_slices)
                  if not n.endswith("_substrate")],
        exag=spacing[0] / spacing[2], vmin=0.0, vmax=2.5 * fr4_level,
        board_plane=material_ids[cu_slices[0]] == COPPER_ID,
        board_traces=material_ids[next_cu[0]] == COPPER_ID, board_blob=blob2d,
        plane_name=short(args.layer), trace_name=short(trace_name), copper_fill=args.copper_fill,
        gap_um=params["gap_um"], blob_area_mm2=params["blob_area_mm2"],
    )
    paths = make_figures(stem, S)
    print(f"Saved {paths[0]} (+{len(paths) - 1} panel PNGs) and {npz_path.name}; total "
          f"{perf_counter() - t0:.1f} s (voxelize {t_vox:.1f} s, simulate {t_sim:.1f} s)")
    print(f"3D view: python {Path(__file__).name} --view-3d {npz_path}")


if __name__ == "__main__":
    main()
