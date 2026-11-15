"""Slide mock-up: xrayrecon with a "Localize Defects" tool registered.

Builds the real ``xrayrecon.gui.MainWin`` on a downsampled real PCB CT volume
(``NPY/slice_data.npy``), registers an extra *mock* tool in the Tools
ParameterEditor (same ``registerFunction`` pattern app.py uses), adds an
"Analysis" menu entry and a "Defect Localization" results dock, draws a few
illustrative defect boxes on the 2D layer view, then grabs the window to PNG.

Nothing here performs real defect detection; the boxes/table rows are
hand-placed illustrations for a roadmap slide. app.py is not modified.

Run from repo root:
    uv run python demo/mock_localize_screenshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from xrayrecon.windows_cuda import prepare_optional_torch_dlls

prepare_optional_torch_dlls()

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from qtextras import bindInteractorOptions as bind

from xrayrecon.gui import MainWin

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "demo_output" / "localize_mock"
OUT_FULL = OUT_DIR / "48-xrayrecon-localize-mock.png"
OUT_CROP = OUT_DIR / "48-xrayrecon-localize-mock-crop.png"

DS = 3  # xy downsample of the real volume so it loads quickly
TOOL_NAME = "Localize Defects"
LEVELS = (45, 215)  # display min/max for the solder layer


def load_volume() -> np.ndarray:
    src = REPO / "NPY" / "slice_data.npy"
    if src.exists():
        vol = np.load(src, mmap_mode="r")
        return np.ascontiguousarray(vol[::DS, ::DS, :])
    return np.load(REPO / "xrayrecon" / "quick_demo.npy")


# Solder-joint layer with visible voiding (found by template-matching against
# the volume, see demo_void_highlight.py): full-res z 158-160, region below.
SOLDER_Z = list(range(157, 162))
SOLDER_LAYER = "Solder joints"
REGION_ROWS = (270, 1850)  # full-res voxels
REGION_COLS = (227, 1374)

# Real void components from demo_void_highlight.segment() on the z 158-160 slab
# (full-res bbox rows r0:r1, cols c0:c1) plus ring contrast (solder - void, in
# raw CT units). Six of the largest/clearest, spread over the board. Only the
# "confidence" shown is mocked (a monotone map of the measured contrast).
# (r0, r1, c0, c1, contrast, label side)
VOID_DETECTIONS = [
    (1492, 1564, 429, 504, 75.4, "left"),
    (349, 390, 762, 837, 56.5, "right"),
    (1453, 1498, 687, 728, 63.3, "right"),
    (390, 419, 493, 529, 76.6, "left"),
    (450, 492, 1046, 1076, 69.7, "right"),
    (1718, 1746, 937, 964, 46.9, "left"),
]


def _conf(contrast: float) -> float:
    return float(np.clip(0.55 + 0.45 * (contrast - 45.0) / 35.0, 0.0, 0.99))


# Detections in downsampled-cube pixel coords:
# (row, col, height, width, type, confidence, label side, full-res center x, y)
MOCK_DEFECTS = []
for _r0, _r1, _c0, _c1, _con, _side in VOID_DETECTIONS:
    _m = 3  # box margin (downsampled px)
    MOCK_DEFECTS.append((
        _r0 / DS - _m, _c0 / DS - _m, (_r1 - _r0) / DS + 2 * _m, (_c1 - _c0) / DS + 2 * _m,
        "Void", round(_conf(_con), 2), _side, (_c0 + _c1) // 2, (_r0 + _r1) // 2,
    ))


def localizeDefects(
    defectTypes: list = ("Voids", "Delamination"),
    model: str = "3D U-Net (sim)",
    confidenceThreshold: float = 0.5,
    minDefectSizeVoxels: int = 8,
    layersToSearch: str = "Selected layers",
    showDefectOverlay: bool = True,
    exportReport: bool = False,
):
    """Mock: locate voids / delamination in the loaded CT volume."""
    return None


# Decorate like app.py's tools so the editor builds matching widgets.
localizeDefects = bind(
    defectTypes=dict(
        type="checklist",
        limits=["Voids", "Delamination", "Cracks", "Missing solder"],
        value=["Voids", "Delamination"],
        title="Defect types",
    ),
    model=dict(
        type="list",
        limits=["3D U-Net (sim)", "3D U-Net (real CT)", "Threshold baseline"],
        title="Model",
    ),
    confidenceThreshold=dict(limits=(0.0, 1.0), step=0.05, title="Confidence threshold"),
    minDefectSizeVoxels=dict(limits=(1, 1000), title="Min defect size (voxels)"),
    layersToSearch=dict(
        type="list",
        limits=["Selected layers", "All layers", "Current view only"],
        title="Layers to search",
    ),
    showDefectOverlay=dict(title="Show defect overlay"),
    exportReport=dict(title="Export defect report (CSV)"),
)(localizeDefects)


def add_overlay(win: MainWin):
    font = QtGui.QFont()
    font.setPointSize(10)
    font.setBold(True)
    items = []
    for r, c, h, w, kind, conf, lpos, _cx, _cy in MOCK_DEFECTS:
        hi = conf >= 0.7
        color = (255, 40, 40) if hi else (255, 170, 0)
        pen = pg.mkPen(color, width=2.5,
                       style=QtCore.Qt.PenStyle.SolidLine if hi else QtCore.Qt.PenStyle.DashLine)
        rect = QtWidgets.QGraphicsRectItem(c, r, w, h)
        rect.setPen(pen)
        rect.setZValue(1000)
        win.plt.addItem(rect)
        if lpos == "right":
            label = pg.TextItem(f"{kind} {conf:.2f}", color=color, anchor=(0, 0.5),
                                fill=pg.mkBrush(0, 0, 0, 170))
            label.setPos(c + w + 4, r + h / 2)
        elif lpos == "above":
            label = pg.TextItem(f"{kind} {conf:.2f}", color=color, anchor=(0.5, 1),
                                fill=pg.mkBrush(0, 0, 0, 170))
            label.setPos(c + w / 2, r - 3)
        else:
            label = pg.TextItem(f"{kind} {conf:.2f}", color=color, anchor=(1, 0.5),
                                fill=pg.mkBrush(0, 0, 0, 170))
            label.setPos(c - 4, r + h / 2)
        label.setFont(font)
        label.setZValue(1001)
        win.plt.addItem(label)
        items += [rect, label]
    return items


def make_results_dock(win: MainWin, layer_name: str) -> QtWidgets.QDockWidget:
    tbl = QtWidgets.QTableWidget(len(MOCK_DEFECTS), 5)
    tbl.setHorizontalHeaderLabels(["#", "Type", "Layer", "Center (x, y)", "Conf."])
    for i, (r, c, h, w, kind, conf, _, cx, cy) in enumerate(MOCK_DEFECTS):
        vals = [
            str(i + 1),
            kind,
            layer_name,
            f"({cx}, {cy})",
            f"{conf:.2f}",
        ]
        for j, v in enumerate(vals):
            it = QtWidgets.QTableWidgetItem(v)
            if j == 4:
                it.setForeground(QtGui.QBrush(QtGui.QColor(200, 0, 0) if conf >= 0.7 else QtGui.QColor(200, 120, 0)))
            tbl.setItem(i, j, it)
    tbl.verticalHeader().setVisible(False)
    tbl.verticalHeader().setDefaultSectionSize(22)
    tbl.horizontalHeader().setStretchLastSection(True)
    tbl.resizeColumnsToContents()

    summary = QtWidgets.QLabel(
        f"{len(MOCK_DEFECTS)} candidates  |  "
        f"{sum(1 for d in MOCK_DEFECTS if d[4] == 'Void')} voids  |  thr 0.50"
    )
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(2, 2, 2, 2)
    lay.addWidget(summary)
    lay.addWidget(tbl)

    dock = QtWidgets.QDockWidget("Defect Localization", win)
    dock.setObjectName("DefectLocalizationDock")
    dock.setWidget(w)
    win.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
    return dock


def main():
    vol = load_volume()
    app = pg.mkQApp()
    win = MainWin(None, None, vol)

    # --- register the new tool exactly like app.py's tools ---
    win.toolsEditor.registerFunction(localizeDefects, name=TOOL_NAME)

    # --- menu entry ---
    analysisMenu = win.menuBar().addMenu("Analysis")
    act = analysisMenu.addAction("Localize Defects...")
    analysisMenu.addAction("Show Defect Overlay").setCheckable(True)

    # Add a user-defined layer over the voided solder-joint slices, using the
    # model's own API (keeps the auto-detected layers, appends one row).
    df = win.model.layerDf
    names = list(df["Layer Name"]) + [SOLDER_LAYER]
    slices = [np.asarray(v) for v in df["Slices"]] + [np.array(SOLDER_Z)]
    win.model.setupTblRows(names, slices)
    shown_row = len(names) - 1
    resultsDock = make_results_dock(win, SOLDER_LAYER)
    win.splitDockWidget(win._plotDock, resultsDock, QtCore.Qt.Orientation.Vertical)

    win.resize(1600, 900)
    win.show()

    tree = win.toolsEditor.tree
    root = win.toolsEditor.rootParameter
    centerSplitter = win._mainWorkspacePage.findChild(QtWidgets.QSplitter)
    mainSplitter = win.centralWidget()
    if not isinstance(mainSplitter, QtWidgets.QSplitter):
        mainSplitter = mainSplitter.findChild(QtWidgets.QSplitter)

    def setup():
        # Select a layer so the 2D view shows real data
        win.tbl.selectRow(shown_row)
        win.minSlider.setValue(LEVELS[0])
        win.maxSlider.setValue(LEVELS[1])
        add_overlay(win)
        # Real 3D render of the loaded volume in the left pane
        try:
            win.highlight3DSlices(3)
            v3 = win.viewer_3d_left
            v3.resetCamera()
            v3.setCameraPosition(distance=v3.opts["distance"] * 0.5, elevation=35, azimuth=50)
        except Exception as exc:  # noqa: BLE001
            print("3D render failed:", exc)

        # Collapse every tool except the new one, and scroll it into view
        for child in root.children():
            child.setOpts(expanded=(child.name() == TOOL_NAME))
        param = root.child(TOOL_NAME)
        for it in param.items:
            tree.setCurrentItem(it)
        tree.header().resizeSection(0, 215)
        tree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerItem)
        if centerSplitter is not None:
            centerSplitter.setSizes([600, 0])  # collapse empty Gerber pane
        if mainSplitter is not None:
            mainSplitter.setSizes([170, 530])
        print("central min", win.centralWidget().minimumSizeHint(), win.centralWidget().minimumSize())
        win.centralWidget().setMinimumWidth(640)
        for sp in (mainSplitter, centerSplitter):
            if sp is not None:
                for i in range(sp.count()):
                    sp.widget(i).setMinimumWidth(0)
        win.resize(1600, 900)
        win.resizeDocks([win._toolsDock, win._plotDock], [410, 500], QtCore.Qt.Orientation.Horizontal)
        win.resizeDocks([win._plotDock, resultsDock], [600, 215], QtCore.Qt.Orientation.Vertical)

    def frame():
        vb = win.plt.getViewBox()
        vb.setRange(
            xRange=(REGION_COLS[0] / DS, REGION_COLS[1] / DS),
            yRange=(REGION_ROWS[0] / DS, REGION_ROWS[1] / DS),
            padding=0.02,
        )
        app.processEvents()

    def no_hover():
        # Avoid a stray mouse-hover highlight in the capture
        for view in (tree, win.tbl, *win.findChildren(QtWidgets.QTableWidget)):
            vp = view.viewport()
            vp.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, False)
            vp.setMouseTracking(False)
            QtWidgets.QApplication.sendEvent(vp, QtCore.QEvent(QtCore.QEvent.Type.Leave))
            vp.update()
        app.processEvents()

    def scroll():
        param = root.child(TOOL_NAME)
        for it in param.items:
            tree.scrollToItem(it, QtWidgets.QAbstractItemView.ScrollHint.PositionAtTop)
        tree.verticalScrollBar().setValue(tree.verticalScrollBar().maximum())

    def grab_full():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        frame()
        no_hover()
        pm = win.grab()
        pm.save(str(OUT_FULL))
        print("full", pm.size(), "tools", win._toolsDock.geometry(), "plot", win._plotDock.geometry())
        # Compact layout for the crop: hide the central layer table/3D panes so
        # the Tools dock and the 2D view + results docks fill the window.
        win.centralWidget().hide()
        win.splitDockWidget(win._plotDock, resultsDock, QtCore.Qt.Orientation.Horizontal)
        resultsDock.setMinimumHeight(0)
        win.resize(1320, 860)
        win.resizeDocks(
            [win._toolsDock, win._plotDock, resultsDock], [410, 520, 390],
            QtCore.Qt.Orientation.Horizontal,
        )

    def grab_crop():
        scroll()
        frame()
        no_hover()
        pm = win.grab()
        pm.save(str(OUT_CROP))
        print("crop", pm.size())
        win.close()
        app.quit()

    QtCore.QTimer.singleShot(1500, setup)
    QtCore.QTimer.singleShot(3500, scroll)
    QtCore.QTimer.singleShot(5500, grab_full)
    QtCore.QTimer.singleShot(7500, grab_crop)
    app.exec_()


if __name__ == "__main__":
    main()
