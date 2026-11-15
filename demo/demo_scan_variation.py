"""
Scan-condition variation: the same board patch simulated under different
dose (I0), number of projection angles, and voxel size.

Mirrors ``xrayrecon.ct_simulation.simulate_ct_volume`` (one Z slice, 2D
attenuation proxy z_eff * electron_density, skimage radon/iradon with
circle=False, ramp filter, centre crop back to the slice shape) but adds a
quantum-noise model in the PROJECTION domain, which simulate_ct_volume lacks:

    p      = line integral of the proxy along each ray, in proxy*mm
    N      ~ Poisson(I0 * exp(-k * p))          (clipped to >= 1 count)
    p'     = -ln(N / I0) / k
    image  = FBP(p')                             (iradon, ramp filter)

k (1/(proxy*mm)) is chosen so that the most attenuating ray through the
native-resolution patch transmits ``--t-min`` (default 10%). The proxy is not a
calibrated linear attenuation coefficient, so k only fixes a plausible
transmission range; see the printed effective copper/FR4 mu.

Voxel-size variation block-averages the proxy 2x in X, Y and Z before
projection (averaging mu is the correct partial-volume rule) and keeps the
photon fluence fixed, so each coarse detector bin collects factor^2 times the
counts of a native bin (2x wider in-plane, 2x thicker slice).

Run from the repo root:
    uv run python demo/demo_scan_variation.py
    uv run python demo/demo_scan_variation.py --out-dir some/folder
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from skimage.transform import iradon, radon

REPO = Path(__file__).resolve().parent.parent
DEFAULT_STACKUP = (REPO / "dataset_relative_path_stackups" / "CAD_CT_Dataset_PCBCT"
                   / "picture1" / "picture1_stackup.json")
DEFAULT_OUT = Path(__file__).resolve().parent / "demo_output" / "scan_variation"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stackup", type=Path, default=DEFAULT_STACKUP)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--layer", default="picture-1-L6-S2",
                   help="copper layer whose middle Z slice is simulated")
    p.add_argument("--origin", type=int, nargs=2, default=(80, 320), metavar=("ROW", "COL"),
                   help="top-left corner of the XY crop, in board pixels")
    p.add_argument("--size", type=int, default=256, help="XY crop size in pixels")
    p.add_argument("--copper-fill", choices=["fr4", "air"], default="fr4",
                   help="material between traces on copper layers (the stackup JSON says air)")
    p.add_argument("--doses", type=float, nargs="+", default=(1e5, 1e4, 1e3),
                   help="I0 = unattenuated counts per native detector bin")
    p.add_argument("--views", type=int, nargs="+", default=(720, 90),
                   help="projection-angle counts (over 180 deg) for the native-voxel rows")
    p.add_argument("--coarse", type=int, default=2,
                   help="voxel coarsening factor for the voxel-size row (0 = skip)")
    p.add_argument("--coarse-views", type=int, default=720)
    p.add_argument("--t-min", type=float, default=0.10,
                   help="transmission of the most attenuating native ray (sets k)")
    p.add_argument("--window", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="display window in proxy units (default: from the noiseless reference)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--save-npz", action="store_true",
                   help="also save the reconstructed panels to scan_variation.npz")
    return p.parse_args()


# ── Physics ─────────────────────────────────────────────────────────────────

def center_crop(img, shape):
    """Same centre crop/pad as ct_simulation._match_reconstruction_shape."""
    out = np.zeros(shape, dtype=img.dtype)
    ch, cw = min(img.shape[0], shape[0]), min(img.shape[1], shape[1])
    sy, sx = (img.shape[0] - ch) // 2, (img.shape[1] - cw) // 2
    ty, tx = (shape[0] - ch) // 2, (shape[1] - cw) // 2
    out[ty:ty + ch, tx:tx + cw] = img[sy:sy + ch, sx:sx + cw]
    return out


def sinogram_mm(atten, pixel_mm, n_views):
    """Line integrals (proxy*mm) at n_views angles over [0, 180)."""
    theta = np.linspace(0.0, 180.0, n_views, endpoint=False)
    return radon(atten, theta=theta, circle=False) * pixel_mm, theta


def add_poisson(p_mm, k, i0, rng):
    """Projection-domain quantum noise: Poisson counts, clipped to >= 1."""
    if i0 is None:
        return p_mm
    counts = rng.poisson(i0 * np.exp(-k * p_mm)).astype(np.float64)
    np.maximum(counts, 1.0, out=counts)
    return -np.log(counts / i0) / k


def fbp(p_mm, theta, pixel_mm, shape):
    """Ramp-filter FBP, returned in proxy units (same as the input slice)."""
    rec = iradon(p_mm / pixel_mm, theta=theta, filter_name="ramp", circle=False)
    return center_crop(rec, shape).astype(np.float32)


def block_mean(a, f):
    z, h, w = (s - s % f for s in a.shape)
    a = a[:z, :h, :w]
    return a.reshape(z // f, f, h // f, f, w // f, f).mean(axis=(1, 3, 5))


# ── Figure ──────────────────────────────────────────────────────────────────

def fmt_dose(i0):
    e = int(round(np.log10(i0)))
    sup = str(e).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return f"I₀ = 10{sup}" if np.isclose(i0, 10.0 ** e) else f"I₀ = {i0:.0e}"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def show(ax, img, extent_mm, window):
    ax.imshow(img, cmap="gray", vmin=window[0], vmax=window[1],
              extent=(0, extent_mm, extent_mm, 0), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("0.4")


def scale_bar(ax, extent_mm, length_mm=5.0, x_frac=0.60):
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.5, foreground="black")]
    x0, y0 = x_frac * extent_mm, 0.93 * extent_mm
    ax.plot([x0, x0 + length_mm], [y0, y0], color="white", lw=3, solid_capstyle="butt",
            path_effects=[pe.Stroke(linewidth=5, foreground="black"), pe.Normal()])
    ax.text(x0 + length_mm / 2, y0 - 0.03 * extent_mm, f"{length_mm:g} mm", color="white",
            ha="center", va="bottom", fontsize=9, fontweight="bold", path_effects=halo)


def grid_figure(path, panels, row_labels, col_labels, extent_mm, window, dpi):
    plt = setup_mpl()
    nr, nc = len(row_labels), len(col_labels)
    cell = 2.1
    fig, axes = plt.subplots(nr, nc, figsize=(0.55 + cell * nc, 0.3 + cell * nr), squeeze=False)
    for r in range(nr):
        for c in range(nc):
            show(axes[r, c], panels[r][c], extent_mm, window)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=11)
    for c in range(nc):
        axes[0, c].set_title(col_labels[c], fontsize=11, pad=4)
    scale_bar(axes[0, 0], extent_mm)
    fig.subplots_adjust(left=0.55 / (0.55 + cell * nc), right=0.995,
                        top=1 - 0.3 / (0.3 + cell * nr), bottom=0.005,
                        wspace=0.04, hspace=0.04)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def row_figure(path, panels, labels, extent_mm, window, dpi):
    plt = setup_mpl()
    n = len(panels)
    cell = 2.3
    fig, axes = plt.subplots(1, n, figsize=(cell * n, cell + 0.55), squeeze=False)
    for ax, img, lab in zip(axes[0], panels, labels):
        show(ax, img, extent_mm, window)
        ax.set_title(lab, fontsize=11, pad=4, linespacing=1.15)
    scale_bar(axes[0, 0], extent_mm)
    fig.subplots_adjust(left=0.005, right=0.995, top=cell / (cell + 0.55), bottom=0.005,
                        wspace=0.04)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    from xrayrecon.stackup import StackupConfig
    from xrayrecon.stackup_voxelize import voxelize_stackup_config
    from xrayrecon.materials import MATERIALS_DB

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = perf_counter()

    config = StackupConfig.from_json(args.stackup)
    if args.copper_fill == "fr4":
        for layer_cfg in config.layers:
            layer_cfg.background_material = "FR4"
    vol, names, slices, _, spacing, _ = voxelize_stackup_config(config, args.stackup)
    print(f"Voxelized {args.stackup.name}: vol {vol.shape}, spacing (x, y, z) mm {spacing}, "
          f"{perf_counter() - t0:.1f} s")
    if args.layer not in names:
        raise ValueError(f"--layer not in stackup: {names}")
    if spacing[0] != spacing[1]:
        raise ValueError("anisotropic XY spacing is not supported")
    px = float(spacing[0])

    # Attenuation proxy (same convention as simulate_ct_volume) over the crop.
    r0, c0 = args.origin
    rs, cs = slice(r0, r0 + args.size), slice(c0, c0 + args.size)
    if rs.stop > vol.shape[1] or cs.stop > vol.shape[2]:
        raise ValueError("crop runs off the board")
    zs = slices[names.index(args.layer)]
    f = max(args.coarse, 1)
    z_mid = int(zs[len(zs) // 2])
    z_lo = z_mid - (z_mid - int(zs[0])) % f  # f-aligned Z block inside the layer
    zblock = slice(z_lo, z_lo + f)
    atten3d = (vol[zblock, rs, cs, 0] * vol[zblock, rs, cs, 1]).astype(np.float64)
    del vol
    atten = atten3d[z_mid - z_lo]
    shape = atten.shape
    extent_mm = args.size * px

    # k from the most attenuating native ray (finely sampled angles).
    p_ref, _ = sinogram_mm(atten, px, 720)
    k = -np.log(args.t_min) / p_ref.max()
    cu, fr4 = MATERIALS_DB["COPPER"], MATERIALS_DB["FR4"]
    print(f"Slice z={z_mid} of {args.layer}, crop rows {rs.start}-{rs.stop - 1}, "
          f"cols {cs.start}-{cs.stop - 1} ({extent_mm:.1f} mm)")
    print(f"max line integral {p_ref.max():.0f} proxy*mm -> k = {k:.3e} per proxy*mm "
          f"(T_min = {args.t_min:.2f}); effective mu: copper "
          f"{k * cu.z_eff * cu.electron_density:.3f}/mm, FR4 "
          f"{k * fr4.z_eff * fr4.electron_density:.4f}/mm")

    # Rows: native voxels at each view count, then the coarse-voxel row.
    rows = [(f"{v} views", atten, px, v, 1) for v in args.views]
    if args.coarse > 1:
        coarse = block_mean(atten3d, f)[0]
        rows.append((f"{f}× voxel", coarse, px * f, args.coarse_views, f * f))

    clean, noisy, timings = {}, {}, []
    for label, img, pix, n_views, gain in rows:
        t = perf_counter()
        p, theta = sinogram_mm(img, pix, n_views)
        clean[label] = fbp(p, theta, pix, img.shape)
        for i0 in args.doses:
            noisy[label, i0] = fbp(add_poisson(p, k, i0 * gain, rng), theta, pix, img.shape)
        dt = perf_counter() - t
        timings.append((label, dt))
        print(f"  {img.shape[0]} px @ {pix:.2f} mm, {n_views} views, "
              f"{len(args.doses)} doses + clean: {dt:.1f} s; min transmission "
              f"{np.exp(-k * p.max()):.3f}")

    ref_label = rows[0][0]
    if args.window is None:
        lo, hi = np.percentile(clean[ref_label], [0.5, 99.5])
        pad = 0.12 * (hi - lo)
        window = (lo - pad, hi + pad)
    else:
        window = tuple(args.window)
    print(f"display window {window[0]:.1f} .. {window[1]:.1f} proxy units "
          f"(FR4 {fr4.z_eff * fr4.electron_density:.1f}, Cu {cu.z_eff * cu.electron_density:.1f})")

    labels = [r[0] for r in rows]
    grid_figure(args.out_dir / "41-scan-variation-grid.png",
                [[noisy[lab, i0] for i0 in args.doses] for lab in labels],
                labels, [fmt_dose(i0) for i0 in args.doses], extent_mm, window, args.dpi)

    hi_dose, lo_dose = max(args.doses), min(args.doses)
    native_v, sparse_v = args.views[0], min(args.views)
    row_panels = [noisy[labels[0], hi_dose], noisy[labels[0], lo_dose],
                  noisy[f"{sparse_v} views", hi_dose]]
    row_labels = [f"Reference\n{fmt_dose(hi_dose)}, {native_v} views",
                  f"Low dose\n{fmt_dose(lo_dose)}", f"Sparse views\n{sparse_v} views"]
    if args.coarse > 1:
        row_panels.append(noisy[labels[-1], hi_dose])
        row_labels.append(f"Coarse voxels\n{f}× voxel ({px * f:.1f} mm)")
    row_figure(args.out_dir / "41-scan-variation-row.png", row_panels, row_labels,
               extent_mm, window, args.dpi)

    if args.save_npz:
        np.savez_compressed(
            args.out_dir / "scan_variation.npz", atten=atten.astype(np.float32),
            k=k, doses=np.array(args.doses), window=np.array(window),
            **{f"{lab}|I0={i0:g}": v for (lab, i0), v in noisy.items()},
            **{f"{lab}|clean": v for lab, v in clean.items()})
    print(f"Wrote {args.out_dir / '41-scan-variation-grid.png'} and "
          f"{args.out_dir / '41-scan-variation-row.png'} ({perf_counter() - t0:.1f} s total)")


if __name__ == "__main__":
    main()
