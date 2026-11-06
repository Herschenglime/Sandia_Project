# Delamination PoC: seeded blister and simulated CT

`demo_delamination_poc.py` takes the PCBCT `picture1` board, builds a voxel model from its Gerbers, and seeds a blister under one copper layer. It then simulates a CT scan of both the clean and the blistered board, and shows the blistered model in a small 3D viewer. It is a proof of concept for a figure, not a defect model: everything below is a deliberate simplification.

```bash
cd xrayrecon
uv run --extra cuda python demo_delamination_poc.py --backend cuda     # whole board
uv run --extra cuda python demo_delamination_poc.py --crop 256         # 256x256 crop around the blister, CPU, ~95 s
uv run --extra cuda python demo_delamination_poc.py --view-3d demo_output/delamination_poc/<run>.npz
```

On GNOME Wayland, prefix the `--view-3d` command with `QT_QPA_PLATFORM=wayland`; otherwise Qt falls back to X11 and OpenGL fails to initialise.

## 1. Board model

The stackup JSON (`CAD_CT_Dataset_PCBCT/picture1/picture1_stackup.json`) is voxelized with `stackup_voxelize.voxelize_stackup_config`.

- **Grid:** voxels are 0.1 × 0.1 mm in XY and 0.01 mm in Z. The full board is 611 × 511 px and 112 slices.
- **Layers:**
  - Each copper layer is 35 µm thick, which comes out as 4 slices.
  - The voxelizer then adds the layer's `gap` of FR4 above it (0.1 mm, 10 slices). It does this after every layer, including the last, so the TOP copper ends up under 100 µm of FR4 rather than on the surface. That is a quirk of the voxelizer; this demo leaves it alone.
- **Copper planes:** layer order from the bottom is BOTTOM, L2-GND, L3-S1, L4-POW1, L5-GND, L6-S2, L7-POW2, TOP. The planes are L2-GND, L4-POW1, L5-GND and L7-POW2, each about 85–89% copper. The others are signal layers.
- **What a voxel holds:** each voxel stores the material's effective atomic number and relative electron density (`materials.MATERIALS_DB`). An integer material-ID volume is kept alongside.
- **Between traces:** the stackup JSON sets the area between traces on copper layers to AIR. This demo overrides it in memory to FR4 (`--copper-fill fr4`, the default), as `configs/jonathan_stackup.json` does. Without that, every signal layer would be full of air and the seeded gap would not be the only air in the board. `--copper-fill air` keeps the JSON's value.
- **Headroom:** a few slices of air are added on top of both the clean and the blistered volume (`peak + 2`), so the lifted layers have room.

## 2. How the blister is made

A blister is a local delamination where the separated layers swell outward. A common cause is trapped moisture or volatiles turning to vapour during reflow. The demo models one at the interface directly under a copper plane near the top of the board (default `--layer picture-1-L7-POW2`, so the gap opens between the L7-POW2 plane and the FR4 below it).

1. **Outline.** `irregular_blob` makes one connected, hole-free 2D shape:
   - it starts from an ellipse with a random aspect ratio and angle, radius `--blob-radius-mm` (3 mm);
   - the edge is pushed in and out by smoothed random noise (`--blob-roughness`);
   - the result is seeded by `--seed`.
2. **Centre.** By default it sits on solid L7 copper, in a 256 px window with many traces on the layer above (`auto_center`), at board pixel (356, 340) for picture1. The location doesn't depend on `--crop`. Use `--center ROW COL` to override.
3. **Height profile.** `blister_profile` solves the membrane equation −∇²u = 1 inside the outline, with u = 0 on the edge, normalises u to 1, and squares it.
   - For a circle this is exactly the deflection of a clamped plate under uniform pressure, (1 − r²/a²)². The dome meets the board tangentially at the edge, so the gap thins smoothly to zero instead of ending in a step.
   - For an irregular outline it gives a smooth dome that follows the outline.
   - It is scaled to `--peak-um` (50 µm) and rounded to whole Z slices: 5 at the peak, giving visible 10 µm terraces.
4. **Lift.** For each (x, y) column, every voxel at or above the interface slice moves up by that column's gap height (`lift_above`). The opened slices become air. This is applied to the attenuation volume and to the material IDs.
   - The lifted plane, the FR4 above it, the TOP traces and the board's top surface all bow up by the same amount.
   - The ground-truth mask (`gt_mask`) is exactly the opened voxels.

**Why lift everything above by the same amount:** the material above the interface is about 0.28 mm thick, while the blister is about 6 mm across. A cover that thin over a span that wide bends like a sheet: its top and bottom move nearly together. That is also why the blister sits near the top. Deep in the board, a thick, stiff stack sits above the gap, and delaminations there tend to stay thin and flat.

**What the model leaves out:**
- The material below the gap sagging slightly.
- Cracked resin at the edge, and partial collapse after cooling.
- The layers above stretching or thinning over the bump (slopes here are only a few degrees).
- Any data on realistic blister heights: 50 µm is a chosen value.

## 3. How the scan is simulated

Both volumes go through `ct_simulation.simulate_ct_volume` over the whole board by default (`--crop N` simulates an N × N px window around the blister instead) and the full Z stack. For each Z slice independently:

1. **Attenuation proxy:** μ = effective atomic number × relative electron density.
   - FR4 ≈ 11, copper ≈ 214, air ≈ 0.007.
   - This is a contrast proxy, not a calibrated attenuation coefficient at any beam energy.
2. **Forward projection:** a parallel-beam Radon transform (`skimage.transform.radon`, or the CUDA projector with `--backend cuda`). It uses `num_angles` evenly spaced angles over 180°, by default max(H, W) (611 for the whole board). `circle=False` keeps the corners of the slice.
3. **Reconstruction:** ramp-filtered back-projection (`iradon`, or the CUDA backprojector), cropped back to the slice size.

**What the output contains:** this produces the real artifacts of filtered back-projection: streaks and ringing around dense copper, for example the black-and-white banding inside signal layers.

**What it does not model:**
- noise
- a polychromatic beam or beam hardening
- detector blur
- cone-beam geometry
- mixing between slices

Because each slice is reconstructed on its own, Z resolution is effectively perfect, so even a one-slice (10 µm) gap shows up. A real scan of a whole board would use coarser, isotropic voxels, and the thin edge of the blister would blur into its neighbours. Detection results on this data will be optimistic.

## 4. Outputs

Written to `demo_output/delamination_poc/blister_<layer>_peak<um>_r<mm>_seed<n>_<backend>.*`:

- **`.png`:** the simulated CT slice one slice above the interface, clean next to blistered, on a shared grey scale.
  - In the clean board that slice is L7 copper.
  - In the blistered board it is air wherever the blister is at least 20 µm tall.
- **`.npz`:**
  - `recon_clean`, `recon_defect`: simulated CT, (Z, H, W)
  - `gt_mask`: the gap voxels
  - `material_ids`: the blistered model
  - `lift_slices`: the gap height per column
  - `spacing_mm`
  - `origin_zrc`: where the crop sits on the board
  - `center_rc_crop`
  - `interface_z`
- **`.json`:** every parameter and the timings.

## 5. 3D viewer

`--view-3d RUN.npz` opens a pyqtgraph OpenGL window with a cutaway through the blister centre.

- **Surfaces:** copper and the gap are shaded surfaces (marching cubes). FR4 is not drawn.
- **Cut face:** shows the material section (FR4 tan, copper orange, gap cyan).
- **Top surface:** a translucent sheet.
- **Exaggeration:** Z is exaggerated `--z-exag` times (default 5; 1 = true scale).
- **Camera:** starts aimed at the blister on the cut face; zoom out with the wheel to see the whole board.
- **Mesh detail:** large volumes (such as the whole board) are meshed at half resolution so the window stays responsive.

Keys:
- **C:** switch the cut face between the material model and the simulated CT.
- **T:** show or hide the board-top sheet.
- **S:** save a PNG next to the `.npz` (`<run>_3d_<n>.png`).
- Mouse drag rotates the view, and the wheel zooms.
