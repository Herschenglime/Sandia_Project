# demo

Standalone scripts made for the November 2026 Sandia review slides (Plus-up 1, defect detection). They use the `xrayrecon` package but are not part of it: nothing in `xrayrecon/` imports them, and they can be changed or deleted without affecting the app. They are kept as starting points for the real work.

Run everything from the repo root with `uv run python demo/<script>.py`. Outputs go to `demo/demo_output/` (ignored by git) unless a script is given another output folder.

| Script | What it does | Data it needs |
|---|---|---|
| `demo_delamination_poc.py` | Seeds a blister (open delamination) under one copper layer of a voxelized board, simulates CT of the clean and blistered boards, and saves both. `--save-labels` also stores the ground-truth label of the injected air; `--view-3d RUN.npz` opens a 3D viewer, and `--spin-out` records a rotating GIF, MP4 or still. Details in `demo_delamination_poc.md`. | PCBCT `picture1` stackup |
| `make_delamination_slide_figures.py` | Draws the in-plane and cross-section CT slices and the matching label from a `--save-labels` run. | an npz from the PoC |
| `demo_scan_variation.py` | Simulates one board patch under different doses, view counts and voxel sizes, with Poisson noise drawn on the sinogram before FBP. | PCBCT `picture1` stackup |
| `demo_void_highlight.py` | Classical solder-void detection on a real CT layer: threshold the solder, fill holes per joint, keep enclosed dark regions; draws automatic ROIs and per-joint void fractions. | `NPY/slice_data.npy` (layer z 158–160) |
| `make_gerber_overlay.py` | Flat, top-down view of a stackup's copper layers in KiCad's default colours, with plane layers dimmed. | any stackup JSON |
| `make_am_slide_diagram.py` | Compact Alvarez–Macovski diagram (board → dual-energy scans → decomposition → material map) using xrayrecon's AM code. Takes the phantom geometry from `notebooks/Spectral-Xray/make_am_process_diagram.py`. | none |
| `mock_localize_screenshot.py` | Screenshot of xrayrecon with a mock "Localize Defects" tool and hand-picked void boxes. The tool does nothing; it is a UI mock-up only. | `NPY/slice_data.npy` |

The data folders (`dataset_relative_path_stackups/`, `NPY/`) are expected unzipped at the repo root and are not tracked.
