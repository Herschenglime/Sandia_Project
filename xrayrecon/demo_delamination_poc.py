"""
Delamination proof of concept: inject an open delamination into a voxelized
PCB, simulate CT of the clean and defective boards, and save a slide figure.

Pipeline:
  1. Voxelize a Gerber stackup (default: PCBCT picture1).
  2. Pick the FR4 substrate slab directly above a copper layer (default: the
     L5-GND plane) and, inside an irregular 2D blob, replace the bottom
     ``--gap-slices`` FR4 slices with AIR. Only FR4 voxels are changed, and
     the boolean ground-truth mask is kept.
  3. Run ``simulate_ct_volume`` on a Z sub-range around the defect (the copper
     layer, the slab, and the next copper layer) for both volumes, over a
     small XY crop centred on the defect (``--crop 0`` for the full board).
  4. Plot clean vs. defective at an XY slice inside the gap and an XZ
     cross-section through the blob, next to the ground truth.

Run from the xrayrecon/ directory:
    python demo_delamination_poc.py                      # 256x256 CPU default
    python demo_delamination_poc.py --backend cuda --crop 0
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from xrayrecon.ct_simulation import simulate_ct_volume
from xrayrecon.materials import MATERIAL_NAMES, MATERIALS_DB
from xrayrecon.stackup import StackupConfig
from xrayrecon.stackup_voxelize import voxelize_stackup_config

DEFAULT_STACKUP = Path(
    "/home/paul/dev/Sandia_Project/CAD_CT_Dataset_PCBCT/picture1/picture1_stackup.json"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "demo_output" / "delamination_poc"

AIR_ID = MATERIAL_NAMES.index("AIR")
COPPER_ID = MATERIAL_NAMES.index("COPPER")
FR4_ID = MATERIAL_NAMES.index("FR4")


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
    p.add_argument("--z-margin", type=int, default=4,
                   help="extra slices simulated beyond the copper layers on either side")
    p.add_argument("--num-angles", type=int, default=None,
                   help="projection angles (default: max(H, W) of the simulated crop)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", action="store_true",
                   help="also run the naive threshold + morphology baseline (stretch goal)")
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


def make_figure(path, clean, defect, gt, z_gap, row_xz, z_labels, spacing, params, pred=None, dice_val=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Window: FR4 mid-grey, air black, copper saturated white.
    fr4_level = float(np.median(clean[z_gap]))
    vmin, vmax = 0.0, 2.5 * fr4_level
    z_exag = 4.0  # display px per Z slice relative to one XY px
    aspect_xz = z_exag
    true_exag = z_exag * spacing[0] / spacing[2]

    ncols = 4 if pred is not None else 3
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 7.2),
                             gridspec_kw=dict(height_ratios=[1, 0.5]))
    mask_cmap = ListedColormap(["#111111", "#ff3b30"])
    pred_cmap = ListedColormap(["#111111", "#30b0ff"])

    cols = [("Clean CT", clean, "gray", vmin, vmax),
            ("Defective CT", defect, "gray", vmin, vmax),
            ("Ground-truth gap", gt.astype(np.uint8), mask_cmap, 0, 1)]
    if pred is not None:
        cols.append((f"Naive baseline (Dice {dice_val:.2f})", pred.astype(np.uint8), pred_cmap, 0, 1))

    h, w = clean.shape[1:]
    for c, (title, data, cmap, lo, hi) in enumerate(cols):
        ax = axes[0, c]
        ax.imshow(data[z_gap], cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.axhline(row_xz, color="#ffcc00", lw=1, ls="--")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f"XY slice in gap\n({h * spacing[0]:.1f} x {w * spacing[1]:.1f} mm)", fontsize=12)

        ax = axes[1, c]
        ax.imshow(data[:, row_xz, :], cmap=cmap, vmin=lo, vmax=hi, origin="lower",
                  aspect=aspect_xz, interpolation="nearest")
        ax.set_xticks([])
        if c == 0:
            ax.set_yticks([z for z, _ in z_labels])
            ax.set_yticklabels([name for _, name in z_labels], fontsize=9)
            ax.set_ylabel(f"XZ section at dashed line\n(Z exaggerated {true_exag:.0f}x)", fontsize=12)
        else:
            ax.set_yticks([])

    fig.suptitle(
        f"Synthetic open delamination: {params['gap_um']:.0f} um air gap above {params['layer_short']}, "
        f"~{params['blob_area_mm2']:.0f} mm$^2$ blob",
        fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             "Gerber -> voxel -> slice-wise parallel-beam FBP (skimage radon/iradon), no noise. "
             f"Voxel {spacing[0] * 1000:.0f} x {spacing[1] * 1000:.0f} x {spacing[2] * 1000:.0f} um. Single sample.",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = perf_counter()

    # 1. Voxelize: vol is (Z, H, W, 2) = [z_eff, rel. electron density], Z
    # runs bottom -> top in stackup order; material_ids is (Z, H, W).
    config = StackupConfig.from_json(args.stackup)
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

    # 3. Simulate a Z sub-range and XY crop around the defect.
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
    fr4_sub = material_ids[sub] == FR4_ID

    # 4. Optional naive baseline.
    pred = dice_val = base_info = None
    if args.baseline:
        pred, base_info = naive_baseline(recon_def, fr4_sub)
        dice_val = dice(pred, gt_sub)
        base_info.update(dice=dice_val, detected=bool((pred & gt_sub).any()),
                         false_positive_voxels=int((pred & ~gt_sub).sum()))
        print(f"Baseline: {base_info}")

    # 5. Figure.
    z_gap_local = int(gap_z[len(gap_z) // 2] - z0)
    row_xz = int(center[0] - rs.start)
    short = lambda n: n.replace("picture-1-", "")
    z_labels = []
    for name, sl in zip(layer_names, layer_slices):
        mid = (sl[0] + sl[-1]) / 2 - z0
        if 0 <= mid < z1 - z0 and not name.endswith("_substrate"):
            z_labels.append((mid, short(name) + " Cu"))
    z_labels.append(((gap_z[0] + gap_z[-1]) / 2 - z0, "gap"))
    z_labels.sort()
    params = dict(
        stackup=str(args.stackup), layer=args.layer, layer_short=short(args.layer),
        gap_slices=args.gap_slices, gap_um=args.gap_slices * spacing[2] * 1000,
        gap_z_global=[int(gap_z[0]), int(gap_z[-1])],
        blob_radius_mm=args.blob_radius_mm, blob_roughness=args.blob_roughness,
        blob_area_mm2=float(blob2d.sum() * spacing[0] * spacing[1]), seed=args.seed,
        center_rc=[int(center[0]), int(center[1])], crop=args.crop,
        sim_z=[z0, z1], sim_rows=[rs.start, rs.stop], sim_cols=[cs.start, cs.stop],
        sim_shape_zhw=list(recon_clean.shape), spacing_mm_xyz=list(spacing),
        backend=args.backend, num_angles=args.num_angles or max(recon_clean.shape[1:]),
        voxelize_s=round(t_vox, 1), simulate_s=round(t_sim, 1), baseline=base_info,
    )
    tag = f"gap{args.gap_slices}_r{args.blob_radius_mm:g}_seed{args.seed}_{args.backend}"
    if args.crop == 0:
        tag += "_full"
    fig_path = args.out / f"delamination_{tag}.png"
    np.savez_compressed(args.out / f"delamination_{tag}.npz", recon_clean=recon_clean,
                        recon_defect=recon_def, gt_mask=gt_sub, fr4_mask=fr4_sub)
    (args.out / f"delamination_{tag}.json").write_text(json.dumps(params, indent=2, default=lambda o: o.item()))
    make_figure(fig_path, recon_clean, recon_def, gt_sub, z_gap_local, row_xz, z_labels,
                spacing, params, pred, dice_val)
    print(f"Saved {fig_path} (total {perf_counter() - t0:.1f} s: voxelize {t_vox:.1f} s, "
          f"simulate {t_sim:.1f} s)")


if __name__ == "__main__":
    main()
