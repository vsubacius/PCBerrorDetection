"""Printable sheets of the boards embedded in the firmware, for live-mode tests.

Each board is printed with four solid black squares, one diagonally outside
each corner. The firmware finds them, maps the board back to 640x640, and
compares the result with the same board's stored template and labels, so the
live result can be checked against the stored-image result for that board.

The squares' size and position are in board pixels and must match
``main/live.hpp`` (kFidSize, kFidOffset).

Print at 100% / "actual size" (turn off "fit to page"). The exact printed size
does not matter for alignment, but the whole pattern must be in the camera's
view. Use the 42 mm sheets first; if the four corner squares do not all fit
in the frame, use the 30 mm sheets.

Usage::

    python firmware/tools/make_prints.py --blob firmware/pcb_bench/models/boards.bin \\
        --out-dir prints
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOARD_PX = 640
FID_SIZE = 56.0     # board pixels; main/live.hpp kFidSize
FID_OFFSET = 60.0   # board pixels; main/live.hpp kFidOffset
DPI = 600
LETTER_IN = (8.5, 11.0)


def read_blob(path: Path):
    """Boards in the firmware blob: list of (index, name, is_clean, n_defects, test01)."""
    blob = path.read_bytes()
    assert blob[:4] == b"PCB1", "not a boards.bin blob"
    n = struct.unpack_from("<I", blob, 8)[0]
    images = 64 + 512 * n
    packed = 640 * 640 // 8
    out = []
    for i in range(n):
        meta = blob[64 + 512 * i: 64 + 512 * (i + 1)]
        name = meta[:16].rstrip(b"\0").decode()
        is_clean, n_def = struct.unpack_from("<II", meta, 16)
        start = images + 2 * packed * i
        test = np.unpackbits(np.frombuffer(blob, np.uint8, packed, start)).reshape(640, 640)
        out.append((i, name, bool(is_clean), n_def, test))
    return out


def render_pattern(test01: np.ndarray, scale: float) -> np.ndarray:
    """Gray image (0 black, 255 white) of board + corner squares.

    ``scale`` is output pixels per board pixel. The image covers the board
    plus the squares, with a white margin of one square on every side.
    Returns (image, origin) where origin is the output position of board (0, 0).
    """
    margin = FID_OFFSET + FID_SIZE          # board px from board edge to image edge
    side = BOARD_PX + 2 * margin
    size = int(round(side * scale))
    # Board pixel coordinates of each output pixel centre.
    coords = (np.arange(size) + 0.5) / scale - margin
    bx = np.floor(coords).astype(int)
    img = np.full((size, size), 255, np.uint8)
    inside = (bx >= 0) & (bx < BOARD_PX)
    iy = np.where(inside)[0]
    sub = test01[np.ix_(bx[iy], bx[iy])]
    img[np.ix_(iy, iy)] = np.where(sub > 0, 255, 0).astype(np.uint8)
    for cx in (-FID_OFFSET, BOARD_PX + FID_OFFSET):
        for cy in (-FID_OFFSET, BOARD_PX + FID_OFFSET):
            x0 = int(round((cx - FID_SIZE / 2 + margin) * scale))
            x1 = int(round((cx + FID_SIZE / 2 + margin) * scale))
            y0 = int(round((cy - FID_SIZE / 2 + margin) * scale))
            y1 = int(round((cy + FID_SIZE / 2 + margin) * scale))
            img[y0:y1, x0:x1] = 0
    origin = margin * scale
    return img, origin


def make_pdf(boards, board_mm: float, out: Path) -> None:
    dots_per_mm = DPI / 25.4
    scale = board_mm * dots_per_mm / BOARD_PX
    page_w, page_h = int(LETTER_IN[0] * DPI), int(LETTER_IN[1] * DPI)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(3.2 * dots_per_mm))
    except OSError:
        font = ImageFont.load_default()
    pages = []
    # Two boards per page, 130 mm apart, so only one is ever in the camera view.
    centres_mm = (75.0, 205.0)
    for p in range(0, len(boards), 2):
        page = Image.new("L", (page_w, page_h), 255)
        draw = ImageDraw.Draw(page)
        for slot, (index, name, is_clean, n_def, test) in enumerate(boards[p:p + 2]):
            pattern, _ = render_pattern(test, scale)
            ph = pattern.shape[0]
            cx = page_w // 2
            cy = int(centres_mm[slot] * dots_per_mm)
            page.paste(Image.fromarray(pattern), (cx - ph // 2, cy - ph // 2))
            kind = "clean (template)" if is_clean else f"{n_def} defects"
            label = (f"Board {index}  -  {name}  -  {kind}  -  board {board_mm:g} mm   "
                     f"(type {index} in the monitor)")
            # Label 22 mm above the pattern, outside the camera's view.
            ty = cy - ph // 2 - int(22 * dots_per_mm)
            tw = draw.textlength(label, font=font)
            draw.text((cx - tw / 2, ty), label, fill=0, font=font)
        pages.append(page)
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=DPI)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", default="firmware/pcb_bench/models/boards.bin")
    ap.add_argument("--out-dir", default="prints")
    ap.add_argument("--sizes-mm", default="42,30")
    args = ap.parse_args()
    boards = read_blob(Path(args.blob))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mm in (float(s) for s in args.sizes_mm.split(",")):
        out = out_dir / f"pcb_prints_{mm:g}mm.pdf"
        make_pdf(boards, mm, out)
        total = mm * (BOARD_PX + 2 * (FID_OFFSET + FID_SIZE / 2)) / BOARD_PX
        print(f"wrote {out}: {len(boards)} boards, board {mm:g} mm, "
              f"whole pattern {total:.1f} mm square")


if __name__ == "__main__":
    main()
