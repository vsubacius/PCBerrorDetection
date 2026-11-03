"""Tests for the Stage-2 student.

These pin the *frozen* architecture. The experiment matrix is only
interpretable if every run shares one student, so a change to the parameter
count, the MAC count or the spatial ladder should fail here and force a
deliberate decision rather than silently invalidating earlier results.

Run from the repo root with:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from models import (  # noqa: E402
    STUDENT_BLOCKS,
    TILE_SIZE,
    SeparableBlock,
    TileStudent,
    profile_model,
)

#: Measured at checkpoint 1 (2026-11-03). Changing these is a project decision.
EXPECTED = {
    1: {"params": 60_966, "macs": 1_329_184},
    2: {"params": 61_110, "macs": 1_559_584},
}


@pytest.mark.parametrize("in_channels", [1, 2])
def test_output_shape_is_six_logits(in_channels):
    model = TileStudent(in_channels=in_channels).eval()
    with torch.no_grad():
        out = model(torch.zeros(4, in_channels, TILE_SIZE, TILE_SIZE))
    assert out.shape == (4, 6)


@pytest.mark.parametrize("in_channels", [1, 2])
def test_frozen_parameter_and_mac_budget(in_channels):
    model = TileStudent(in_channels=in_channels)
    prof = profile_model(model, in_channels)
    assert prof.params == EXPECTED[in_channels]["params"]
    assert prof.macs == EXPECTED[in_channels]["macs"]
    # Team targets: ~63K params, ~63 KB int8 weights, ~1.5M MACs per tile.
    assert 55_000 <= prof.params <= 70_000
    assert prof.int8_weight_bytes <= 64 * 1024
    assert prof.macs <= 1_600_000


def test_head_returns_logits_not_probabilities():
    """No sigmoid in the graph: the losses and the firmware both want logits."""
    model = TileStudent(in_channels=1)
    assert not any(
        isinstance(m, (torch.nn.Sigmoid, torch.nn.Softmax)) for m in model.modules()
    )
    model.head.bias.data.fill_(50.0)
    with torch.no_grad():
        out = model(torch.zeros(1, 1, TILE_SIZE, TILE_SIZE))
    assert out.max() > 1.0, "a logit head must be able to exceed 1"


def test_spatial_ladder_matches_the_documented_geometry():
    """160 -> 80 -> 40 -> 20 -> 10 -> 5 -> 3 -> 3 -> 3, then global pooling."""
    prof = profile_model(TileStudent(in_channels=1), 1)
    shapes = {name: shape for name, shape, _ in prof.tensors}
    assert shapes["input_pool"] == (1, 80, 80)
    assert shapes["stem"] == (16, 40, 40)
    assert shapes["blocks.0.pointwise"] == (32, 20, 20)
    assert shapes["blocks.1.pointwise"] == (64, 10, 10)
    assert shapes["blocks.2.pointwise"] == (96, 5, 5)
    assert shapes["blocks.3.pointwise"] == (128, 3, 3)
    assert shapes["blocks.5.pointwise"] == (128, 3, 3)
    assert shapes["pool"] == (128, 1, 1)


def test_input_pool_precedes_the_stem_convolution():
    """The 4x reduction must happen before the stem, or MACs quadruple."""
    prof = profile_model(TileStudent(in_channels=1), 1)
    names = [name for name, _, _ in prof.tensors]
    assert names.index("input_pool") < names.index("stem")
    stem_macs = 40 * 40 * 16 * 9 * 1
    assert stem_macs == 230_400
    assert prof.macs - stem_macs < 1_100_000


def test_convolutions_have_no_bias_so_batchnorm_folds_cleanly():
    model = TileStudent(in_channels=2)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            assert module.bias is None, f"{name} should rely on its BatchNorm"


def test_depthwise_is_actually_depthwise():
    block = SeparableBlock(32, 64, stride=2)
    assert block.depthwise.groups == 32
    assert block.depthwise.in_channels == block.depthwise.out_channels == 32
    assert block.pointwise.kernel_size == (1, 1)


def test_block_channel_ladder_is_the_team_design():
    assert [c for c, _ in STUDENT_BLOCKS] == [32, 64, 96, 128, 128, 128]
    assert [s for _, s in STUDENT_BLOCKS] == [2, 2, 2, 2, 1, 1]


def test_rejects_unsupported_input_channel_counts():
    for bad in (0, 3, 6):
        with pytest.raises(ValueError):
            TileStudent(in_channels=bad)


def test_profiling_does_not_leave_the_model_in_eval_mode():
    model = TileStudent(in_channels=1)
    model.train()
    profile_model(model, 1)
    assert model.training, "profiling must not silently change training mode"
