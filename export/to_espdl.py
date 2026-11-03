"""Export the Stage-2 student to ONNX and then to a deployable ``.espdl``.

The deployment path chosen at checkpoint 2 (2026-11-03)::

    PyTorch -> ONNX (opset 18) -> ESP-PPQ -> .espdl -> ESP-DL -> ESP-IDF

Two stages, deliberately separable:

``export_onnx``
    Pure PyTorch + ONNX. Runs anywhere, including this repo's CI, and is where
    the operator check happens. BatchNorm folds into the preceding convolution
    during constant folding, which is why the convolutions carry no bias.

``quantize_to_espdl``
    Needs ESP-PPQ, which pins ``python >=3.11,<3.13`` -- newer interpreters
    cannot install it. Run this stage in Colab or a 3.12 virtualenv.

Why ONNX in the middle rather than ``espdl_quantize_torch`` directly: the ONNX
file is the artefact we can check operators on, visualise, and diff between
runs without a quantizer installed. ESP-PPQ accepts either; this project uses
``espdl_quantize_onnx`` so that every run in the experiment matrix is quantized
from a byte-identical intermediate.

Exporter choice (both verified deployable, 2026-11-03)::

    legacy  (default)  Conv Relu AveragePool GlobalAveragePool Flatten Gemm
    dynamo             Conv Relu AveragePool ReduceMean        Reshape Gemm

The legacy TorchScript exporter is the default because it emits
``GlobalAveragePool``, for which ESP-DL ships a dedicated kernel, and because
it needs no ``onnxscript``. It is deprecated in PyTorch >= 2.9, so ``dynamo``
is kept as a tested fallback: its graph is equally supported and, as a bonus,
free of the Identity artefact described below.

Both paths are forced to write a **single self-contained file**. The dynamo
exporter otherwise spills most initialisers into an ``.onnx.data`` sidecar, so
copying the ``.onnx`` alone to Colab would quietly lose the weights.

Quantization facts that shape the results (ESP-DL commit 5989f4a):

* ESP32-S3 quantizes **per-tensor**, symmetric, with **power-of-two scales**
  (``scale = 2^exponent``). ESP32-P4 gets per-channel Conv/Gemm; the S3 does
  not. Per-tensor weight scales are the known weak point of depthwise
  convolutions, so expect the depthwise layers to carry most of the int8 error
  and check ``error_report`` before blaming the architecture.
* The pooled binary input takes values ``{0, 0.25, 0.5, 0.75, 1}``, all exactly
  representable at ``exponent = -6``, so input quantization is lossless.
* ``export_test_values=True`` embeds a test input/output pair in the model;
  ``dl::Model::test()`` on the chip then checks device output against the PC,
  which is exactly the PC-vs-device parity check the experiment plan requires.
* Only ``batch_size = 1`` is supported on device.

Usage::

    python export/to_espdl.py --in-channels 1 --out export/artifacts
    python export/to_espdl.py --in-channels 1 --weights runs/control_s0/best.pt \
        --npz data/tiles/deeppcb_tiles_t160_s120.npz --espdl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "training"))
sys.path.insert(0, str(_REPO / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import TILE_SIZE, TileStudent  # noqa: E402
from operators import (  # noqa: E402
    CONV_ALLOWED_GROUPS,
    RECOMMENDED_OPSET,
    describe_op,
    unsupported_ops,
)

#: ESP-PPQ cannot be installed outside this range (checked 2026-11-03).
ESP_PPQ_PYTHON = ">=3.11,<3.13"


def external_initializers(model) -> list[str]:
    """Initialiser names whose data lives outside the ``.onnx`` file.

    Must be empty for a deployable artefact: ESP-PPQ and anyone copying the
    file around should need nothing beside it.
    """
    import onnx

    return [
        init.name for init in model.graph.initializer
        if init.HasField("data_location")
        and init.data_location == onnx.TensorProto.EXTERNAL
    ]


def strip_identity(model) -> int:
    """Remove ``Identity`` nodes, rewiring consumers onto the real producer.

    ESP-DL has no ``Identity`` kernel. PyTorch emits them when constant folding
    deduplicates identical initialisers, which happens for every untrained
    model: with default BatchNorm statistics each folded convolution bias is
    the same zero vector. Returns the number of nodes removed.
    """
    graph = model.graph
    removed = 0
    while True:
        node = next((n for n in graph.node if n.op_type == "Identity"), None)
        if node is None:
            return removed
        src, dst = node.input[0], node.output[0]
        for other in graph.node:
            for i, name in enumerate(other.input):
                if name == dst:
                    other.input[i] = src
        for out in graph.output:
            if out.name == dst:
                out.name = src
        graph.node.remove(node)
        removed += 1


def check_conv_groups(model) -> list[str]:
    """Convolutions whose ``groups`` ESP-DL cannot run (not 1 and not depthwise).

    Shapes come from the initialisers, so this works on an exported graph with
    no inference required.
    """
    weights = {init.name: tuple(init.dims) for init in model.graph.initializer}
    problems = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        groups = next((a.i for a in node.attribute if a.name == "group"), 1)
        if groups == 1:
            continue
        shape = weights.get(node.input[1])
        # Conv weight is (out_channels, in_channels / groups, kH, kW); a fully
        # depthwise convolution therefore has in_channels/groups == 1.
        if shape is None or len(shape) < 2 or shape[1] != 1:
            problems.append(
                f"{node.name or node.op_type}: groups={groups} weight={shape}; "
                f"ESP-DL allows {CONV_ALLOWED_GROUPS}"
            )
    return problems


def export_onnx(
    model: torch.nn.Module,
    path: str | Path,
    in_channels: int,
    tile: int = TILE_SIZE,
    opset: int = RECOMMENDED_OPSET,
    exporter: str = "legacy",
) -> dict:
    """Export to ONNX, strip export artefacts, and verify operator support.

    Args:
        exporter: ``"legacy"`` (TorchScript, emits GlobalAveragePool) or
            ``"dynamo"`` (torch.export, emits ReduceMean and needs onnxscript).
            Both produce graphs ESP-DL can run; see the module docstring.

    Returns a report dict; raises ``RuntimeError`` only if an operator survives
    that ESP-DL genuinely cannot run.
    """
    if exporter not in ("legacy", "dynamo"):
        raise ValueError(f"exporter must be 'legacy' or 'dynamo', got {exporter!r}")
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, in_channels, tile, tile)

    model = model.eval()
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["tile"], output_names=["logits"],
        opset_version=opset, dynamo=(exporter == "dynamo"),
        **({} if exporter == "dynamo" else {"do_constant_folding": True}),
    )

    # Resolve any external-data sidecar back into the file itself, so the
    # .onnx is self-contained wherever it is copied. The dynamo exporter
    # spills most initialisers out by default.
    graph = onnx.load(str(path), load_external_data=True)
    removed = strip_identity(graph)
    onnx.save(graph, str(path), save_as_external_data=False)
    for sidecar in path.parent.glob(path.name + ".data"):
        sidecar.unlink()
    onnx.checker.check_model(onnx.load(str(path)))
    if external_initializers(onnx.load(str(path), load_external_data=False)):
        raise RuntimeError(f"{path} still references external weight data")

    ops = sorted({n.op_type for n in graph.graph.node})
    bad = unsupported_ops(ops)
    group_problems = check_conv_groups(graph)

    report = {
        "path": str(path),
        "opset": opset,
        "exporter": exporter,
        "in_channels": in_channels,
        "nodes": len(graph.graph.node),
        "op_types": ops,
        "identity_removed": removed,
        "unsupported": bad,
        "conv_group_problems": group_problems,
        "size_bytes": path.stat().st_size,
    }
    if bad or group_problems:
        raise RuntimeError(
            "graph cannot run on ESP-DL int8:\n  "
            + "\n  ".join([describe_op(o) for o in bad] + group_problems)
        )
    return report


def calibration_batches(
    npz: str | Path | None,
    in_channels: int,
    batch_size: int = 8,
    steps: int = 32,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Calibration tiles for post-training quantization.

    Drawn from the *training* fold only -- never validation and never the
    official test split, so quantization never sees evaluation data.

    With ``npz=None`` this returns synthetic binary tiles, which is adequate
    for a toolchain dry run on an untrained model but **not** for a real PTQ
    export: the activation ranges would be wrong.
    """
    rng = np.random.default_rng(seed)
    if npz is None:
        return [
            torch.from_numpy(
                rng.integers(0, 2, (batch_size, in_channels, TILE_SIZE, TILE_SIZE))
                .astype(np.float32)
            )
            for _ in range(steps)
        ]

    from tile_dataset import TileData

    representation = {1: "A", 2: "B"}[in_channels]
    data = TileData.load(npz, "train")
    batches = []
    images = rng.permutation(len(data))
    for step in range(steps):
        # One image supplies n_tiles tiles; take as many images as the batch needs.
        picks, tiles = [], 0
        while tiles < batch_size:
            idx = int(images[(step * 4 + len(picks)) % len(images)])
            x, _, _ = data.tiles(idx, representation, None, rng)
            picks.append(x)
            tiles += len(x)
        stacked = np.concatenate(picks)[:batch_size]
        batches.append(torch.from_numpy(stacked.astype(np.float32)))
    return batches


def quantize_to_espdl(
    onnx_path: str | Path,
    espdl_path: str | Path,
    in_channels: int,
    calib: list[torch.Tensor],
    target: str = "esp32s3",
    quant_type: str = "w8a8",
    tile: int = TILE_SIZE,
    device: str = "cpu",
):
    """Quantize an ONNX graph to ``.espdl`` with ESP-PPQ.

    Writes three files beside ``espdl_path``: the ``.espdl`` binary, a ``.info``
    text dump (model structure, quantized weights, and the embedded test
    input/output), and a ``.json`` of quantization parameters.
    """
    try:
        from esp_ppq.api import espdl_quantize_onnx
    except ImportError as exc:  # pragma: no cover - depends on interpreter
        raise RuntimeError(
            f"esp-ppq is not installed. It requires python {ESP_PPQ_PYTHON}; "
            f"this interpreter is {sys.version_info.major}.{sys.version_info.minor}. "
            "Run this stage in Colab or a 3.12 virtualenv:\n"
            "  pip uninstall -y ppq\n"
            "  pip install git+https://github.com/espressif/esp-ppq.git"
        ) from exc

    Path(espdl_path).parent.mkdir(parents=True, exist_ok=True)
    return espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib,
        calib_steps=len(calib),
        input_shape=[1, in_channels, tile, tile],
        target=target,
        quant_type=quant_type,
        collate_fn=lambda batch: batch.to(device),
        device=device,
        error_report=True,
        export_test_values=True,
        verbose=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-channels", type=int, default=1, choices=[1, 2])
    ap.add_argument("--weights", default=None,
                    help="student .pt state_dict; omit for an untrained dry run")
    ap.add_argument("--out", default="export/artifacts")
    ap.add_argument("--name", default=None, help="basename; default student_c{N}")
    ap.add_argument("--espdl", action="store_true",
                    help="also quantize to .espdl (needs esp-ppq)")
    ap.add_argument("--npz", default=None, help="tile dataset for PTQ calibration")
    ap.add_argument("--target", default="esp32s3")
    ap.add_argument("--quant-type", default="w8a8",
                    choices=["w8a8", "w8a16", "w16a16", "none"])
    ap.add_argument("--calib-steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exporter", default="legacy", choices=["legacy", "dynamo"])
    args = ap.parse_args()

    name = args.name or f"student_c{args.in_channels}"
    out = Path(args.out)

    model = TileStudent(in_channels=args.in_channels)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"loaded weights from {args.weights}")
    else:
        print("no --weights: exporting an UNTRAINED model (toolchain dry run only)")

    report = export_onnx(model, out / f"{name}.onnx", args.in_channels,
                         exporter=args.exporter)
    print(f"\nONNX  {report['path']}  ({report['size_bytes'] / 1024:.1f} KB)")
    print(f"  opset {report['opset']}, {report['nodes']} nodes, "
          f"{report['exporter']} exporter")
    print(f"  operators: {', '.join(report['op_types'])}")
    if report["identity_removed"]:
        print(f"  stripped {report['identity_removed']} Identity node(s) "
              "(untrained-model constant-aliasing artefact)")
    print("  all operators supported by ESP-DL int8 and conv groups are legal")

    if not args.espdl:
        print("\n--espdl not given; stopping after ONNX.")
        return

    if args.npz is None:
        print("\nWARNING: no --npz, calibrating on synthetic random tiles. "
              "Valid for a dry run, NOT for a reported result.")
    calib = calibration_batches(args.npz, args.in_channels,
                                steps=args.calib_steps, seed=args.seed)
    quantize_to_espdl(
        out / f"{name}.onnx", out / f"{name}.espdl", args.in_channels, calib,
        target=args.target, quant_type=args.quant_type,
    )
    espdl = out / f"{name}.espdl"
    print(f"\n.espdl {espdl}  ({espdl.stat().st_size / 1024:.1f} KB)")
    print("  flash it and call dl::Model::test() to check device/PC parity")


if __name__ == "__main__":
    main()
