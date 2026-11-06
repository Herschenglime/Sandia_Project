"""
Delamination proof of concept: seed a blister into a voxelized PCB, simulate
CT of the clean and blistered boards, and view the result in 3D.

See demo_delamination_poc.md for how the blister is built and how the scan is
simulated.

Run from the xrayrecon/ directory:
    python demo_delamination_poc.py --backend cuda       # whole board
    python demo_delamination_poc.py --crop 256           # 256x256 crop, quick CPU check
    python demo_delamination_poc.py --view-3d demo_output/delamination_poc/<run>.npz
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import spsolve

from xrayrecon.materials import MATERIAL_NAMES, MATERIALS_DB

DEFAULT_STACKUP = Path(
    "/home/paul/dev/Sandia_Project/CAD_CT_Dataset_PCBCT/picture1/picture1_stackup.json"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "demo_output" / "delamination_poc"

AIR_ID = MATERIAL_NAMES.index("AIR")
COPPER_ID = MATERIAL_NAMES.index("COPPER")
FR4_ID = MATERIAL_NAMES.index("FR4")

GAP_RGB = (0.00, 0.72, 0.85)
COPPER_RGB = (0.85, 0.50, 0.15)
FR4_RGB = (0.80, 0.75, 0.55)
AUTO_CENTER_HALF = 128  # auto_center scores a 256 px window, independent of --crop


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stackup", type=Path, default=DEFAULT_STACKUP)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu",
                   help="projection/backprojection backend for simulate_ct_volume")
    p.add_argument("--layer", default="picture-1-L7-POW2",
                   help="copper layer that lifts: the blister opens directly beneath it")
    p.add_argument("--peak-um", type=float, default=50.0,
                   help="blister height at its thickest point (rounded to Z slices)")
    p.add_argument("--blob-radius-mm", type=float, default=3.0,
                   help="nominal blister radius in mm before the random perturbation")
    p.add_argument("--blob-roughness", type=float, default=0.35,
                   help="0 = ellipse, larger = more irregular outline")
    p.add_argument("--center", type=int, nargs=2, metavar=("ROW", "COL"), default=None,
                   help="blister centre in board pixels (default: auto, on solid plane copper)")
    p.add_argument("--crop", type=int, default=0,
                   help="XY crop size (px) around the blister for simulation; 0 = full board "
                        "(e.g. 256 for a quick CPU check)")
    p.add_argument("--copper-fill", choices=["fr4", "air"], default="fr4",
                   help="material between traces on copper layers (the stackup JSON says air)")
    p.add_argument("--num-angles", type=int, default=None,
                   help="projection angles (default: max(H, W) of the simulated crop)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view-3d", type=Path, default=None, metavar="NPZ",
                   help="open a saved run in a standalone 3D window instead of running the pipeline")
    p.add_argument("--z-exag", type=float, default=5.0,
                   help="with --view-3d: vertical exaggeration (1 = true scale)")
    return p.parse_args()


# ── Blister ─────────────────────────────────────────────────────────────────

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


def blister_profile(blob):
    """Normalised blister height (0..1) over an arbitrary outline.

    Solve the membrane equation  -lap(u) = 1  inside the blob with u = 0 on
    its edge, normalise, and square. For a circle this is exactly the
    clamped-plate deflection under uniform pressure, (1 - r^2/a^2)^2: a dome
    that meets the board tangentially at the edge. For an irregular outline
    it gives a smooth dome that follows the outline."""
    pts = np.argwhere(blob)
    idx = np.full(blob.shape, -1)
    idx[blob] = np.arange(len(pts))
    n = len(pts)
    ri, ci, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = pts[:, 0] + dr, pts[:, 1] + dc
        inside = (r >= 0) & (r < blob.shape[0]) & (c >= 0) & (c < blob.shape[1])
        nb = np.full(n, -1)
        nb[inside] = idx[r[inside], c[inside]]
        ok = nb >= 0
        ri.append(np.flatnonzero(ok)); ci.append(nb[ok]); vals.append(-np.ones(ok.sum()))
    a = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(ri), np.concatenate(ci))),
                          shape=(n, n))
    u = np.zeros(blob.shape)
    u[blob] = spsolve(a, np.ones(n))
    return (u / u.max()) ** 2


def lift_above(arr, lift, z_i, fill):
    """Shift every voxel at or above slice z_i up by lift[row, col] slices and
    fill the opened gap with `fill`. The array needs at least lift.max()
    empty slices on top."""
    out = arr.copy()
    top = arr.shape[0]
    for k in range(1, int(lift.max()) + 1):
        cols = lift == k
        out[z_i + k:, cols] = arr[z_i:top - k, cols]
        out[z_i:z_i + k, cols] = fill
    return out


def auto_center(material_ids, plane_z, trace_z, half):
    """Centre the blister on solid plane copper, in a window with many traces
    on a neighbouring signal layer, so the crop shows some structure."""
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


# ── 3D view ─────────────────────────────────────────────────────────────────

def view_3d(npz_path, z_exag):
    """Cutaway of the blistered model: copper and the gap as shaded surfaces,
    the cut face as a material section (or simulated CT), and the board top
    as a translucent sheet. Keys: C cut face model/CT, T top sheet, S save PNG."""
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from skimage.measure import marching_cubes

    d = np.load(npz_path)
    mids, gap, recon = d["material_ids"], d["gt_mask"], d["recon_defect"]
    cut_row = int(d["center_rc_crop"][0])
    dx, dy, dz = d["spacing_mm"]
    sz = dz * z_exag
    nz, h, w = mids.shape
    keep = np.zeros(h, dtype=bool)
    keep[:cut_row + 1] = True  # far half; the camera looks at the cut from the removed side

    def to_gl(z, row, col):
        # GL x = column, y = rows flipped (matches a top-down image), z = up
        return np.column_stack([col * dx, (h - 1 - row) * dy, z * sz])

    def surface(mask, color):
        m = np.pad(mask[:, keep], 1).astype(np.float32)
        m = ndi.gaussian_filter(m, 0.5)
        step = 1 if m.size < 1e7 else 2  # keep full-board meshes interactive
        verts, faces, _, _ = marching_cubes(m, 0.45, step_size=step)
        verts -= 1  # undo the pad
        md = gl.MeshData(vertexes=to_gl(*verts.T), faces=faces)
        return gl.GLMeshItem(meshdata=md, smooth=True, shader="shaded", color=color)

    def section_image(rgba, y_off):
        # rgba is (Z, W, 4) for the cut row; GLImageItem wants (W, Z, 4)
        item = gl.GLImageItem(np.ascontiguousarray(rgba.transpose(1, 0, 2)), smooth=False)
        item.scale(dx, sz, 1)
        item.translate(-0.5 * dx, -0.5 * sz, 0)
        item.rotate(90, 1, 0, 0)
        item.translate(0, (h - 1 - cut_row - 0.5) * dy - y_off, 0)
        return item

    app = pg.mkQApp("Delamination 3D")

    class View(gl.GLViewWidget):
        def keyPressEvent(self, ev):
            key = ev.text().lower()
            if key == "c":
                model_face.setVisible(not model_face.visible())
                ct_face.setVisible(not ct_face.visible())
            elif key == "t":
                top_sheet.setVisible(not top_sheet.visible())
            elif key == "s":
                n = len(list(Path(npz_path).parent.glob(Path(npz_path).stem + "_3d_*.png")))
                path = Path(npz_path).with_name(f"{Path(npz_path).stem}_3d_{n}.png")
                self.grabFramebuffer().save(str(path))
                print(f"Saved {path}")
            else:
                super().keyPressEvent(ev)

    view = View()
    view.setWindowTitle(f"Seeded blister (cyan), Z x{z_exag:g} - C: cut face model/CT, "
                        "T: top sheet, S: save PNG")
    view.setBackgroundColor("w")

    view.addItem(surface(mids == COPPER_ID, COPPER_RGB + (1.0,)))
    view.addItem(surface(gap, GAP_RGB + (1.0,)))

    sec = mids[:, cut_row, :]
    rgba = np.zeros(sec.shape + (4,), dtype=np.uint8)
    rgba[sec == FR4_ID] = (*(int(255 * c) for c in FR4_RGB), 255)
    rgba[sec == COPPER_ID] = (*(int(255 * c) for c in COPPER_RGB), 255)
    rgba[gap[:, cut_row, :]] = (*(int(255 * c) for c in GAP_RGB), 255)
    model_face = section_image(rgba, 0.01)
    view.addItem(model_face)

    ct = recon[:, cut_row, :]
    board = mids[:, cut_row, :] != AIR_ID
    lo, hi = np.percentile(ct[board], [1, 99])
    grey = (np.clip((ct - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    below_top = np.arange(nz)[:, None] <= nz - 1 - np.argmax(board[::-1], axis=0)
    ct_rgba = np.stack([grey, grey, grey, np.where(below_top, 255, 0).astype(np.uint8)], axis=-1)
    ct_face = section_image(ct_rgba, 0.01)
    ct_face.setVisible(False)
    view.addItem(ct_face)

    # Board top: highest non-air voxel in each kept column.
    solid = mids[:, keep] != AIR_ID
    top = nz - 1 - np.argmax(solid[::-1], axis=0)
    rows = np.flatnonzero(keep)[::-1]  # ascending GL y
    top_sheet = gl.GLSurfacePlotItem(
        x=np.arange(w) * dx, y=(h - 1 - rows) * dy, z=((top[::-1] + 0.5) * sz).T,
        shader="shaded", color=FR4_RGB + (0.35,), glOptions="translucent")
    top_sheet.setDepthValue(10)
    view.addItem(top_sheet)

    # Start on the blister where it meets the cut face; zoom out for the whole board.
    cut_col = int(d["center_rc_crop"][1])
    center = pg.Vector(cut_col * dx, (h - 1 - cut_row) * dy, int(d["interface_z"]) * sz)
    view.setCameraPosition(pos=center, distance=36, elevation=28, azimuth=-62)
    view.resize(1400, 900)
    view.show()
    pg.exec()


# ── Figure ──────────────────────────────────────────────────────────────────

def make_figure(path, clean, defect, board, spacing, title):
    """Two CT slices side by side, shared grey window, scale bar only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(clean[board], [1, 99])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7 * clean.shape[0] / clean.shape[1] + 1.4))
    for ax, img, name in zip(axes, (clean, defect), ("Original Board", "Simulated Delamination")):
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(name, fontsize=14)
        ax.set_axis_off()
    bar_mm = 10 if clean.shape[1] * spacing[0] > 40 else 5
    bar = bar_mm / spacing[0]
    hgt = clean.shape[0]
    pad = hgt / 32
    axes[0].plot([pad, pad + bar], [hgt - pad] * 2, color="black", lw=4, solid_capstyle="butt")
    axes[0].text(pad + bar / 2, hgt - 1.8 * pad, f"{bar_mm} mm", ha="center", fontsize=11,
                 bbox=dict(fc="white", ec="none", pad=1))
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── Pipeline ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.view_3d is not None:
        view_3d(args.view_3d, args.z_exag)
        return

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

    if args.layer not in layer_names or args.layer.endswith("_substrate"):
        raise ValueError(f"--layer must be a copper layer. Layers: {layer_names}")
    li = layer_names.index(args.layer)
    z_i = int(layer_slices[li][0])  # interface: the blister opens just below this slice
    peak = int(round(args.peak_um / 1000 / spacing[2]))
    if peak < 1:
        raise ValueError("--peak-um is below one Z slice")

    # Empty headroom so the lifted layers stay inside the volume (same for both).
    air = MATERIALS_DB["AIR"]
    pad = peak + 2
    vol = np.concatenate([vol, np.broadcast_to(np.array([air.z_eff, air.electron_density], vol.dtype),
                                               (pad,) + vol.shape[1:])])
    material_ids = np.concatenate([material_ids, np.full((pad,) + material_ids.shape[1:], AIR_ID,
                                                         material_ids.dtype)])

    # 2. Blister: irregular outline, clamped-plate dome, layers above lifted.
    above = layer_slices[li + 2] if li + 2 < len(layer_slices) else layer_slices[li - 2]
    center = tuple(args.center) if args.center else auto_center(
        material_ids, z_i, int(above[0]), AUTO_CENTER_HALF)
    blob2d = irregular_blob(vol.shape[1:3], center, args.blob_radius_mm / spacing[0],
                            args.blob_roughness, rng)
    lift = np.rint(peak * blister_profile(blob2d)).astype(int)

    vol_def = lift_above(vol, lift, z_i, np.array([air.z_eff, air.electron_density]))
    mids_def = lift_above(material_ids, lift, z_i, AIR_ID)
    gt = np.zeros(material_ids.shape, dtype=bool)
    for k in range(1, peak + 1):
        gt[z_i + k - 1, lift >= k] = True

    # 3. Simulate the full Z stack over an XY crop around the blister.
    rs = crop_window(center[0], args.crop, vol.shape[1])
    cs = crop_window(center[1], args.crop, vol.shape[2])
    sub = (slice(None), rs, cs)
    print(f"Blister centre (row, col) {center}, under {args.layer} (slice {z_i}), peak {peak} slices, "
          f"{int(gt.sum())} gap voxels; simulating {vol.shape[0]} slices, rows {rs.start}..{rs.stop - 1}, "
          f"cols {cs.start}..{cs.stop - 1} on {args.backend}")
    t1 = perf_counter()
    recon_clean = simulate_ct_volume(vol[sub], num_angles=args.num_angles,
                                     projection_backend=args.backend, action_name="delam_clean")
    recon_def = simulate_ct_volume(vol_def[sub], num_angles=args.num_angles,
                                   projection_backend=args.backend, action_name="delam_defect")
    t_sim = perf_counter() - t1

    # 4. Save data, then the figure.
    short = args.layer.replace("picture-1-", "")
    params = dict(
        stackup=str(args.stackup), layer=args.layer, interface_z=z_i, copper_fill=args.copper_fill,
        peak_um=args.peak_um, peak_slices=peak, headroom_slices=pad,
        blob_radius_mm=args.blob_radius_mm, blob_roughness=args.blob_roughness,
        blob_area_mm2=float(blob2d.sum() * spacing[0] * spacing[1]), seed=args.seed,
        gap_voxels=int(gt.sum()), center_rc=[int(center[0]), int(center[1])], crop=args.crop,
        sim_rows=[rs.start, rs.stop], sim_cols=[cs.start, cs.stop],
        sim_shape_zhw=list(recon_clean.shape), spacing_mm_xyz=list(spacing),
        backend=args.backend, num_angles=args.num_angles or max(recon_clean.shape[1:]),
        noise=None, voxelize_s=round(t_vox, 1), simulate_s=round(t_sim, 1),
    )
    tag = f"{short}_peak{args.peak_um:g}um_r{args.blob_radius_mm:g}_seed{args.seed}_{args.backend}"
    if args.crop == 0:
        tag += "_full"
    if args.copper_fill == "air":
        tag += "_cuair"
    stem = args.out / f"blister_{tag}"
    npz_path = stem.with_suffix(".npz")
    center_crop = np.array([center[0] - rs.start, center[1] - cs.start])
    np.savez_compressed(npz_path, recon_clean=recon_clean, recon_defect=recon_def,
                        gt_mask=gt[sub], material_ids=mids_def[sub], lift_slices=lift[rs, cs],
                        spacing_mm=np.asarray(spacing), origin_zrc=np.array([0, rs.start, cs.start]),
                        center_rc_crop=center_crop, interface_z=z_i)
    stem.with_suffix(".json").write_text(json.dumps(params, indent=2, default=lambda o: o.item()))

    # Plan view one slice above the interface: in the clean board this is the
    # lifted plane's copper; in the blistered board it is air wherever the
    # blister is at least two slices tall.
    plan_z = z_i + (1 if peak >= 2 else 0)
    png = stem.with_suffix(".png")
    make_figure(png, recon_clean[plan_z], recon_def[plan_z], material_ids[plan_z, rs, cs] != AIR_ID,
                spacing, f"Simulated CT, slice through the {short} plane")
    print(f"Saved {png.name}, {npz_path.name}; total {perf_counter() - t0:.1f} s "
          f"(voxelize {t_vox:.1f} s, simulate {t_sim:.1f} s)")
    print(f"3D view: python {Path(__file__).name} --view-3d {npz_path}")


if __name__ == "__main__":
    main()
