"""Model definitions for the gated tile cascade.

Stage 2 is :class:`TileStudent`, the team-designed depthwise-separable CNN that
every run in the experiment matrix shares. Only its ``in_channels`` changes
across the input-mode ablation (A=1, B=2, C=2); width, depth and strides are
frozen so that the gate and the distillation loss are the only things under
test.

Geometry (frozen at checkpoint 1, 2026-11-03)
---------------------------------------------
The three team targets -- ~63K params, ~1.5M MACs/tile, 16 KB peak activation
-- are not jointly satisfiable on a full 160x160 tile: the int8 input buffer
alone is 25 KB. Params and MACs are met; the activation figure is not, and is
not a binding constraint (the ESP32-S3 has 512 KB SRAM and 8 MB PSRAM). The
chosen configuration keeps full tile resolution and hits the MAC budget by
reducing 4x before the first convolution::

    AvgPool2d(2)              160x160xC_in -> 80x80xC_in
    Conv3x3 s2 -> 16, BN, ReLU   80x80     -> 40x40x16
    DSBlock 16  -> 32   s2       40x40     -> 20x20
    DSBlock 32  -> 64   s2       20x20     -> 10x10
    DSBlock 64  -> 96   s2       10x10     ->   5x5
    DSBlock 96  -> 128  s2         5x5     ->   3x3
    DSBlock 128 -> 128  s1         3x3     ->   3x3
    DSBlock 128 -> 128  s1         3x3     ->   3x3
    GlobalAvgPool + Linear(128, 6)         -> 6 logits

Why the 4x reduction comes *before* the stem convolution: a stride-2 stem on
the full tile would compute at 80x80 and cost 0.92M MACs by itself, blowing the
budget. Pooling first costs 0.23M for the same output resolution. Why *average*
pooling and not max: copper covers ~81% of a DeepPCB board, so a 2x2 OR
saturates to all-ones and destroys the signal; a 2x2 mean gives 5 levels and is
still multiply-free on the MCU (3 adds and a shift). The reduction is safe for
this dataset -- no annotated defect box is thinner than 23 px on either axis
(p1 of the min-dimension over all 10,013 boxes), so every box still spans
>= 5 px at 40x40.

The head returns **logits**, never probabilities. Nothing downstream needs a
quantized sigmoid: the losses use the ``*_with_logits`` forms, and the firmware
compares a logit against ``log(p / (1 - p))`` instead of a probability against
``p``. That keeps one more operator out of the int8 graph.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from deeppcb import NUM_CLASSES  # noqa: E402

#: Frozen Stage-2 geometry. ``(out_channels, stride)`` per separable block.
STUDENT_BLOCKS: tuple[tuple[int, int], ...] = (
    (32, 2), (64, 2), (96, 2), (128, 2), (128, 1), (128, 1),
)
STUDENT_STEM_CHANNELS = 16
#: Spatial reduction applied to the tile before the stem convolution.
STUDENT_INPUT_POOL = 2
#: Stride of the stem convolution itself (total reduction = pool * stem stride).
STUDENT_STEM_STRIDE = 2
TILE_SIZE = 160


class SeparableBlock(nn.Module):
    """3x3 depthwise conv then 1x1 pointwise conv, each with BN + ReLU.

    Convolutions carry no bias because the following BatchNorm supplies one and
    absorbs it at fusion time, which is also what the int8 exporter expects.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
        )
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_depthwise(self.depthwise(x)))
        return self.relu(self.bn_pointwise(self.pointwise(x)))


class TileStudent(nn.Module):
    """The Stage-2 multi-label tile classifier shared by every run.

    Args:
        in_channels: 1 for input mode A (test only), 2 for B (test+template)
            and C (test+XOR).
        num_classes: six, one sigmoid output per DeepPCB defect type.

    Returns raw logits of shape ``(N, num_classes)``.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()
        if in_channels not in (1, 2):
            raise ValueError(f"in_channels must be 1 or 2, got {in_channels}")
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.input_pool = nn.AvgPool2d(STUDENT_INPUT_POOL)
        self.stem = nn.Conv2d(
            in_channels, STUDENT_STEM_CHANNELS, kernel_size=3,
            stride=STUDENT_STEM_STRIDE, padding=1, bias=False,
        )
        self.bn_stem = nn.BatchNorm2d(STUDENT_STEM_CHANNELS)
        self.relu = nn.ReLU(inplace=True)

        blocks = []
        channels = STUDENT_STEM_CHANNELS
        for out_channels, stride in STUDENT_BLOCKS:
            blocks.append(SeparableBlock(channels, out_channels, stride))
            channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_stem(self.stem(self.input_pool(x))))
        x = self.blocks(x)
        return self.head(self.pool(x).flatten(1))


@dataclass
class Profile:
    """Measured cost of one forward pass over a single tile."""

    params: int
    params_trainable: int
    macs: int
    peak_activation_bytes: int
    peak_activation_with_input_bytes: int
    tensors: list[tuple[str, tuple[int, ...], int]]

    @property
    def int8_weight_bytes(self) -> int:
        """Weight bytes after int8 quantization, with BatchNorm folded away.

        BatchNorm affine parameters vanish into the preceding convolution at
        fusion time, so they do not appear in the deployed artifact.
        """
        return self.params - self.batchnorm_params

    batchnorm_params: int = 0

    def describe(self) -> str:
        return (
            f"params            {self.params:>12,}\n"
            f"  of which BN     {self.batchnorm_params:>12,}  (folded at export)\n"
            f"int8 weights      {self.int8_weight_bytes / 1024:>11.1f} KB\n"
            f"MACs / tile       {self.macs / 1e6:>11.2f} M\n"
            f"MACs / 25-tile board{self.macs * 25 / 1e6:>9.2f} M\n"
            f"peak activation   {self.peak_activation_bytes / 1024:>11.1f} KB"
            f"  (intermediates only)\n"
            f"  incl. input tile{self.peak_activation_with_input_bytes / 1024:>11.1f} KB"
        )


def profile_model(model: nn.Module, in_channels: int, tile: int = TILE_SIZE) -> Profile:
    """Count parameters, multiply-accumulates and activation bytes.

    MACs are counted from the *observed* output shapes via forward hooks rather
    than from hand arithmetic, so the number cannot drift away from the model.
    One MAC is one multiply-add; convolution bias and BatchNorm are excluded,
    since both fold into the preceding convolution before deployment.

    Peak activation assumes an int8 runtime that holds one layer's input and
    output live at once and frees everything older -- the usual arena behaviour.
    """
    was_training = model.training
    model.eval()
    tensors: list[tuple[str, tuple[int, ...], int]] = []
    macs = 0
    handles = []

    def hook(name: str):
        def fn(module, inputs, output):
            nonlocal macs
            if isinstance(module, nn.Conv2d):
                out_elems = output.shape[2] * output.shape[3] * output.shape[1]
                macs += out_elems * (
                    module.in_channels // module.groups
                ) * module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Linear):
                macs += module.in_features * module.out_features
            tensors.append((name, tuple(output.shape[1:]), int(output[0].numel())))
        return fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.AvgPool2d,
                              nn.AdaptiveAvgPool2d)):
            handles.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        model(torch.zeros(1, in_channels, tile, tile))
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    input_bytes = in_channels * tile * tile
    sizes = [n for _, _, n in tensors]
    peak = max((a + b for a, b in zip(sizes, sizes[1:])), default=0)
    peak_with_input = max(input_bytes + sizes[0], peak)

    bn_params = sum(
        p.numel() for m in model.modules()
        if isinstance(m, nn.BatchNorm2d) for p in m.parameters()
    )
    return Profile(
        params=sum(p.numel() for p in model.parameters()),
        params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        macs=macs,
        peak_activation_bytes=peak,
        peak_activation_with_input_bytes=peak_with_input,
        tensors=tensors,
        batchnorm_params=bn_params,
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Profile the Stage-2 student.")
    ap.add_argument("--in-channels", type=int, default=None, choices=[1, 2],
                    help="default: profile both 1 (mode A) and 2 (modes B/C)")
    ap.add_argument("--tile", type=int, default=TILE_SIZE)
    ap.add_argument("--layers", action="store_true", help="print every tensor")
    args = ap.parse_args()

    channels = [args.in_channels] if args.in_channels else [1, 2]
    for c in channels:
        mode = "A (test only)" if c == 1 else "B / C (test + template or XOR)"
        model = TileStudent(in_channels=c)
        prof = profile_model(model, c, args.tile)
        print(f"\n=== TileStudent, in_channels={c} -- input mode {mode} ===")
        print(f"input             {c}x{args.tile}x{args.tile}")
        print(prof.describe())
        if args.layers:
            print(f"\n{'tensor':<28}{'shape':>16}{'int8 KB':>10}")
            for name, shape, n in prof.tensors:
                print(f"{name:<28}{str(shape):>16}{n / 1024:>10.2f}")


if __name__ == "__main__":
    main()
