"""
Flat, top-down view of a board's copper layers stacked on top of each other,
in the style of a PCB editor (KiCad's default layer colours), for slides.

Run from the repo root:
    python xrayrecon/make_gerber_overlay.py --stackup <stackup.json> --out overlay.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# KiCad's default copper colours: F.Cu, In1.Cu ... In6.Cu, B.Cu.
FRONT = (200, 52, 52)
INNER = [(127, 200, 127), (206, 125, 44), (79, 203, 203), (219, 98, 139), (167, 165, 198), (40, 204, 217)]
BACK = (77, 127, 196)
BACKGROUND = (0, 16, 35)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stackup", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.75, help="opacity of signal and outer copper layers")
    p.add_argument("--plane-alpha", type=float, default=0.18,
                   help="opacity of plane layers (names containing GND or POW), which would otherwise cover everything")
    p.add_argument("--width", type=int, default=1600, help="output width in pixels")
    return p.parse_args()


def main():
    args = parse_args()
    from xrayrecon.materials import MATERIALS_DB
    from xrayrecon.stackup import StackupConfig
    from xrayrecon.stackup_voxelize import voxelize_stackup_config

    config = StackupConfig.from_json(args.stackup)
    vol, names, slices, *_ = voxelize_stackup_config(config, args.stackup)
    cu = MATERIALS_DB["COPPER"].z_eff
    # Substrate entries between copper layers are dielectric, not copper layers.
    masks = {n: np.isclose(vol[int(s[len(s) // 2]), :, :, 0], cu)
             for n, s in zip(names, slices) if not n.endswith("_substrate")}
    del vol

    # Stackups are listed bottom first; paint bottom to top so the top layer is on top.
    order = [n for n in names if n in masks]
    top, bottom = order[-1], order[0]
    inner = order[1:-1][::-1]  # In1 is the layer just under the top
    colour = {top: FRONT, bottom: BACK}
    for i, n in enumerate(inner):
        colour[n] = INNER[i % len(INNER)]

    h, w = next(iter(masks.values())).shape
    img = np.empty((h, w, 3), np.float32)
    img[:] = BACKGROUND
    for n in order:
        m = masks[n]
        a = args.plane_alpha if ("GND" in n or "POW" in n) else args.alpha
        img[m] = (1 - a) * img[m] + a * np.array(colour[n], np.float32)
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    if out.width != args.width:
        out = out.resize((args.width, round(out.height * args.width / out.width)), Image.LANCZOS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"wrote {args.out} ({out.width}x{out.height}); layers top to bottom: {order[::-1]}")


if __name__ == "__main__":
    main()
