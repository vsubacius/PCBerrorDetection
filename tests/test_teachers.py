"""Tests for the distillation teacher.

The teacher's job is to know more than the student. These check the structural
preconditions for that -- matching output contract, matching input channels,
much greater capacity, full input resolution -- since a KD result read off a
teacher that is weaker than, or differently-fed from, the student would be
meaningless.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "training"))

from models import TILE_SIZE, TileStudent, profile_model  # noqa: E402
from teachers import BasicBlock, ResNet18Teacher, build_teacher  # noqa: E402


@pytest.mark.parametrize("in_channels", [1, 2])
def test_output_contract_matches_the_student(in_channels):
    """The KD loss compares the two directly, so shapes must agree exactly."""
    teacher = build_teacher(in_channels).eval()
    student = TileStudent(in_channels=in_channels).eval()
    x = torch.zeros(3, in_channels, TILE_SIZE, TILE_SIZE)
    with torch.no_grad():
        assert teacher(x).shape == student(x).shape == (3, 6)


def test_is_a_resnet18_eight_basic_blocks_two_per_stage():
    teacher = build_teacher(1)
    blocks = [m for m in teacher.modules() if isinstance(m, BasicBlock)]
    assert len(blocks) == 8
    assert ResNet18Teacher.layer_blocks == (2, 2, 2, 2)
    assert ResNet18Teacher.layer_channels == (64, 128, 256, 512)


@pytest.mark.parametrize("in_channels", [1, 2])
def test_teacher_has_far_more_capacity_than_the_student(in_channels):
    teacher = profile_model(build_teacher(in_channels), in_channels)
    student = profile_model(TileStudent(in_channels=in_channels), in_channels)
    assert teacher.params > 50 * student.params
    assert teacher.macs > 50 * student.macs


def test_teacher_sees_the_tile_at_full_resolution():
    """Unlike the student, no 4x pooling before the first convolution.

    That extra resolution is much of why the teacher should be stronger, and
    therefore much of what there is to distil.
    """
    teacher = profile_model(build_teacher(1), 1)
    shapes = {name: shape for name, shape, _ in teacher.tensors}
    assert shapes["conv1"] == (64, 80, 80), "a stride-2 stem on the full tile"

    student = profile_model(TileStudent(in_channels=1), 1)
    student_shapes = {name: shape for name, shape, _ in student.tensors}
    assert student_shapes["input_pool"] == (1, 80, 80), "student pools first"


def test_input_channels_are_configurable_so_the_teacher_matches_the_input_mode():
    """A teacher fed differently from its student would leak information."""
    for channels in (1, 2):
        assert build_teacher(channels).conv1.in_channels == channels
    for bad in (0, 3):
        with pytest.raises(ValueError):
            build_teacher(bad)


def test_residual_branches_start_as_identity():
    """Last BN of each block is zero-initialised: standard, and it matters
    more than usual here because there is no pretraining to fall back on."""
    for block in build_teacher(1).modules():
        if isinstance(block, BasicBlock):
            assert torch.all(block.bn2.weight == 0)


def test_a_freshly_built_teacher_outputs_near_zero_logits():
    """With identity residuals and a small head init, output starts unbiased."""
    teacher = build_teacher(1).eval()
    with torch.no_grad():
        out = teacher(torch.rand(4, 1, TILE_SIZE, TILE_SIZE))
    assert out.abs().max() < 1.0


def test_downsample_appears_exactly_where_the_stage_changes():
    blocks = [m for m in build_teacher(1).modules() if isinstance(m, BasicBlock)]
    has_downsample = [b.downsample is not None for b in blocks]
    # Stage 0 keeps 64 channels at stride 1; stages 1-3 each start with one.
    assert has_downsample == [False, False, True, False, True, False, True, False]


def test_teacher_can_be_trained_a_few_steps():
    teacher = build_teacher(1)
    x = torch.rand(2, 1, TILE_SIZE, TILE_SIZE)
    y = torch.zeros(2, 6)
    y[0, 1] = 1.0
    optimiser = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    first = None
    for _ in range(3):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(teacher(x), y)
        first = first if first is not None else loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert loss.item() < first


def test_dropout_is_optional_and_off_by_default():
    assert isinstance(build_teacher(1).dropout, torch.nn.Identity)
    assert isinstance(build_teacher(1, dropout=0.3).dropout, torch.nn.Dropout)


def test_the_loss_can_consume_teacher_and_student_logits_together():
    """End-to-end wiring check for the KD runs."""
    from losses import build_loss

    teacher = build_teacher(1).eval()
    student = TileStudent(in_channels=1)
    x = torch.rand(5, 1, TILE_SIZE, TILE_SIZE)
    targets = torch.zeros(5, 6)
    targets[0, 2] = 1.0
    ignore = torch.zeros(5, 6)
    with torch.no_grad():
        teacher_logits = teacher(x)
    loss = build_loss("defect_kd", alpha=3.0)
    value = loss(student(x), targets, ignore, teacher_logits)
    assert torch.isfinite(value) and value.requires_grad
