"""Pack real DeepPCB validation boards into a blob the benchmark firmware embeds.

Until the microSD path exists, this is how ``pcb_bench`` scores *real* boards
on the chip instead of synthetic ones. The blob carries, per board:

* the bit-packed 640x640 test and template images (51,200 bytes each);
* the tile labels (positive / ignore masks), so the chip can split its
  firings into true positives, false positives and ignored firings;
* every defect box reduced to its class and the 25-bit mask of tiles that may
  be credited with it (``eval.metrics.qualifying_tiles``, the 70% rule), so
  defect-level recall on the chip uses exactly the Python definition;
* optionally, the decisions the PC int8 model makes on each tile (``--onnx``),
  so the chip can report how many of its decisions differ from the PC.

Two of the boards are *clean*: their template is used as the test image, so
every firing on them is a false alarm.

Only boards from the validation fold are used -- never train (the model saw
them) and never the official test split (kept for reported results).

The int8 expectation re-runs the exact PTQ that ``export/to_espdl.py`` uses
(same ONNX, same calibration tiles, same setting), so it needs esp-ppq and
python ``>=3.11,<3.13``.

Usage::

    python firmware/tools/make_boards.py --run runs/input_C_s0 \\
        --onnx firmware/pcb_bench/models/student_c2.onnx \\
        --thresholds "[-2.75,-2.625,-1.5,-1.125,-4.625,-3.75]" \\
        --out firmware/pcb_bench/models/boards.bin

Blob layout (little-endian), mirrored by ``main/boards.hpp``::

    header (64 B): "PCB1", u32 version, u32 n_boards, u32 in_channels,
                   f32 thresholds_logit[6], u32 rep ('A'|'B'|'C'), pad
    meta (512 B) per board: char name[16], u32 is_clean, u32 n_defects,
                   u8 y_mask[25], u8 ignore_mask[25], u8 expect_mask[25],
                   u8 has_expect, then n <= 48 defects of
                   {u8 class, u8 pad[3], u32 tile_mask}, zero padded
    images: per board, packed test (51,200 B) then packed template (51,200 B)

Bit order inside a packed row is MSB first (``np.packbits`` default).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for sub in ("training", "data", "eval", "export"):
    sys.path.insert(0, str(_REPO / sub))

from metrics import qualifying_tiles  # noqa: E402
from tile_dataset import TileData  # noqa: E402
from deeppcb import CLASS_IDS  # noqa: E402

MAGIC = b"PCB1"
VERSION = 1
HEADER_BYTES = 64
META_BYTES = 512
MAX_DEFECTS = 48
N_TILES = 25


def class_mask(row: np.ndarray) -> int:
    """(6,) 0/1 -> bitmask with bit c set for class channel c."""
    return int(sum(1 << c for c in range(6) if row[c]))


def int8_decisions(onnx_path: Path, npz: Path, rep: str, boards: list,
                   thresholds: np.ndarray) -> list[np.ndarray]:
    """(25, 6) bool per board: what the PC int8 model fires, gate off."""
    import torch
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor
    from to_espdl import calibration_batches

    calib = calibration_batches(npz, 2 if rep in "BC" else 1, steps=32, seed=0,
                                representation=rep)
    setting = QuantizationSettingFactory.espdl_setting(8)
    setting.quantize_activation_setting.calib_algorithm = "percentile"
    graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(Path("/tmp") / "make_boards_check.espdl"),
        calib_dataloader=calib, calib_steps=len(calib),
        input_shape=[1, 2 if rep in "BC" else 1, 160, 160], target="esp32s3",
        quant_type="w8a8", setting=setting, collate_fn=lambda b: b,
        device="cpu", error_report=False, export_test_values=False, verbose=0,
    )
    ex = TorchExecutor(graph, device="cpu")
    out = []
    for x in boards:
        logits = np.concatenate([
            ex.forward(inputs=torch.from_numpy(x[i:i + 1]))[0].detach().numpy()
            for i in range(len(x))])
        out.append(logits > thresholds[None, :])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir with config.json")
    ap.add_argument("--npz", default="data/tiles/deeppcb_tiles_t160_s120.npz")
    ap.add_argument("--thresholds", required=True,
                    help="JSON list of 6 logit thresholds, chosen on int8 outputs")
    ap.add_argument("--onnx", default=None,
                    help="ONNX of the deployed model; adds PC int8 expectations")
    ap.add_argument("--n-defective", type=int, default=8)
    ap.add_argument("--n-clean", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="firmware/pcb_bench/models/boards.bin")
    args = ap.parse_args()

    config = json.loads((Path(args.run) / "config.json").read_text())
    rep = config["representation"]
    thresholds = np.array(json.loads(args.thresholds), dtype=np.float32)
    assert thresholds.shape == (6,)

    data = TileData.load(args.npz, "val")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(data), size=args.n_defective, replace=False)

    boards = []   # (name, is_clean, test, temp, y, ignore, boxes)
    for i in picks:
        boards.append((str(data.names[i]), 0, data.test[i], data.temp[i],
                       data.y[i], data.ignore[i], data.boxes[i]))
    for i in picks[:args.n_clean]:
        zeros = np.zeros_like(data.y[i])
        boards.append((str(data.names[i]) + "c", 1, data.temp[i], data.temp[i],
                       zeros, zeros.astype(bool), np.zeros((0, 5), np.int32)))

    expect = None
    if args.onnx:
        xs = []
        for _, _, test, temp, *_ in boards:
            layers = {"A": [test], "B": [test, temp],
                      "C": [test, (test ^ temp).astype(np.uint8)]}[rep]
            xs.append(np.stack([data.grid.cut(l) for l in layers], 1)
                      .astype(np.float32))
        expect = int8_decisions(Path(args.onnx), Path(args.npz), rep, xs, thresholds)

    in_channels = 1 if rep == "A" else 2
    blob = bytearray()
    header = MAGIC + struct.pack("<III6fI", VERSION, len(boards), in_channels,
                                 *thresholds.tolist(), ord(rep))
    blob += header.ljust(HEADER_BYTES, b"\0")

    summary = []
    for b, (name, is_clean, test, temp, y, ignore, boxes) in enumerate(boards):
        tiles_for_box = qualifying_tiles(boxes, data.grid)
        if len(boxes) > MAX_DEFECTS:
            raise SystemExit(f"{name}: {len(boxes)} defects > {MAX_DEFECTS}")
        meta = name.encode()[:15].ljust(16, b"\0")
        meta += struct.pack("<II", is_clean, len(boxes))
        meta += bytes(class_mask(y[t] > 0) for t in range(N_TILES))
        meta += bytes(class_mask(ignore[t]) for t in range(N_TILES))
        if expect is not None:
            meta += bytes(class_mask(expect[b][t]) for t in range(N_TILES))
            meta += b"\1"
        else:
            meta += bytes(N_TILES) + b"\0"
        for box, tiles in zip(boxes, tiles_for_box):
            mask = int(sum(1 << int(t) for t in tiles))
            meta += struct.pack("<B3xI", CLASS_IDS.index(int(box[4])), mask)
        assert len(meta) <= META_BYTES
        blob += meta.ljust(META_BYTES, b"\0")
        n_fire = int(expect[b].sum()) if expect is not None else -1
        summary.append((name, is_clean, len(boxes), n_fire))

    for _, _, test, temp, *_ in boards:
        for img in (test, temp):
            assert img.shape == (640, 640) and img.max() <= 1
            blob += np.packbits(img.astype(np.uint8), axis=-1).tobytes()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))
    print(f"wrote {out}  ({len(blob) / 1024:.0f} KB, {len(boards)} boards, "
          f"input mode {rep})")
    for name, is_clean, n_def, n_fire in summary:
        kind = "clean" if is_clean else "defective"
        extra = f", PC int8 fires {n_fire} (tile,class) pairs" if n_fire >= 0 else ""
        print(f"  {name:<10} {kind:<9} {n_def:>2} defects{extra}")


if __name__ == "__main__":
    main()
