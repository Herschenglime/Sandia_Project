# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`xrayrecon` is a PyQt5/PyQtGraph desktop app that turns 3D X-ray CT volumes of PCBs into 2D per-layer renderings, auto-segments layers along the depth axis, can synthesize volumes from Gerber files, and exports layers to [S3A](https://gitlab.com/s3a/s3a) projects for netlist annotation. See `xrayrecon/ARCHITECTURE.md` for a class diagram.

## Environment & commands

Packaging uses `uv` with `pyproject.toml` and `uv.lock` at the **repo root** (Python 3.11 per `.python-version`). The package source lives at `xrayrecon/xrayrecon/` (hatch `packages = ["xrayrecon/xrayrecon"]`). The README's `pip install -e ./xrayrecon` is out of date. There is no `setup.py` in that directory.

```bash
uv sync                                                        # install locked deps into .venv
uv run python -m xrayrecon.main                                # launch with phantom sphere data
uv run python -m xrayrecon.main --datafile path/to/volume.npy  # .npy, image dir, or single image
uv run python -m xrayrecon.main --datafile xrayrecon/quick_demo.npy
```

CLI args are parsed by `fire`, so they map directly to `main(datafile=...)`. Dependency pins matter: `numpy<2`, `pandas<2`, `setuptools<81` are required for S3A/qtextras compatibility.

There is no test suite, linter config, or build step.

## Architecture

- **Entry**: `main.py` loads a cube with `io.load_data` (returns `(Z, H, W)`) or generates `gui.helpers.phantomSpheres`, then constructs `gui.MainWin`.
- **Axis convention**: `LayerModel.forceDepthToLastAx` transposes so the *smallest* dimension becomes axis 2. Throughout the GUI the cube is `(rows, cols, depth)`, and layers are lists of slice indices along axis 2. `gui/__init__.py` sets pyqtgraph `imageAxisOrder="row-major"` globally on import.
- **`LayerModel`** (`gui/layermodel.py`, a `QObject`) owns `dataCube` plus a `PandasModel` table (`layerDf` columns: View, #, Layer Name, Slices). It emits `sigCubeChanged` / `sigLayerUpdated`, and the views (`LayerView`, `PCBViewer3D`) react to those signals. A layer image is produced by `applyProcess`, which defaults to a mean over the layer's slices. Layer info comes from a dict, a CSV (filenames containing `"grouped"` are parsed as start/end ranges, others as per-slice rows), or auto-detection.
- **Auto layer segmentation** (`layerseg.py`): `cv.matchTemplate` correlation between adjacent Z slices, computed over windows, then `scipy.signal.find_peaks` on the scores to locate layer boundaries. `LayerModel.getAutoLayerInfo` calls `layerInfoForWindow`.
- **`MainWin`** (`gui/app.py`) is the coordination hub. User actions are exposed through a qtextras `ParameterEditor` (`self.toolsEditor.registerFunction(...)`). Function parameters are declared with `@bind(...)` decorators (file pickers, limits). New tools should follow that pattern.
- **Gerber → volume pipeline**: `pygerber_integration.PyGerberRasterizer` rasterizes Gerber text into a mask. That mask is wrapped in `gerber_utils.GerberLayer`, `voxelizer.Voxelizer.voxelize_stackup` stacks the layers into a 3D volume using `materials.MATERIALS_DB` (Z_eff + density), and `MainWin._apply_gerber_volume` loads the result into the model. Multi-layer boards are described by a JSON stackup (`stackup.StackupConfig.from_json`: layers with file, thickness, gap, materials, RGBA color, plus dpi and xy/z resolution in mm). `get_material` silently falls back to AIR for unknown names, so `MainWin._material_or_error` is used to enforce valid names.
- **`cubecorrections/`**: `performAllCorrections` drops extra dimensions, auto-rotates each axis (`rotation.getOptimalRotation`), and trims empty slices. It is only applied when `LayerModel(applyCubeCorrections=True)`.
- `gui/algorithms.py` holds standalone edge-detection and circle-finding experiments (pure-Python Canny/Hough) used by the "find circles" tool.

## Other directories

- `notebooks/Spectral-Xray/`: a separate exploration of spectral CT and Alvarez–Macovski material decomposition on a hyperspectral lizard scan. Its scripts hardcode Windows paths (`E:\...`) and are not part of the package. `_prep_for_app.py` exports a cube in the `(nx, ny, nz)` layout the app expects.
- `literature/`: reference PDFs.
- `NPY/` and `NPY.zip` (untracked): local sample volumes. Don't commit large data files.
