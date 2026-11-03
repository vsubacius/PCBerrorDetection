"""Tests for the masked losses, including the defect-aware KD loss.

The reduction identities matter most. The experiment plan's whole argument is
that `defect_kd` differs from `hinton_kd` in exactly one thing -- the
defect-tile weight -- so `defect_kd` at alpha=1 must equal `hinton_kd` to
floating-point precision. If it does not, the "prior method" comparison is
measuring an implementation difference instead of a method difference.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "training"))

from losses import LOSS_KINDS, LossConfig, MaskedTileLoss, build_loss  # noqa: E402


def sample(n=32, seed=0, ignore_frac=0.2, pos_frac=0.25):
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(n, 6, generator=g)
    teacher = torch.randn(n, 6, generator=g)
    targets = (torch.rand(n, 6, generator=g) < pos_frac).float()
    ignore = ((torch.rand(n, 6, generator=g) < ignore_frac).float()
              * (1.0 - targets))        # ignore never coexists with positive
    return student, teacher, targets, ignore


WEIGHTS = torch.full((6,), 12.0)


# --------------------------------------------------------------------------- #
# the reduction identities
# --------------------------------------------------------------------------- #

def test_defect_kd_at_alpha_one_equals_hinton_kd():
    """The plan's central claim: w = 1 everywhere reduces to Hinton KD."""
    student, teacher, targets, ignore = sample()
    common = dict(lam=0.7, temperature=3.0)
    hinton = build_loss("hinton_kd", **common)
    assert hinton.config.alpha == 1.0

    # 'defect_kd' refuses alpha == 1 outright, because two rows of the run
    # matrix must not be silently identical. So drive alpha down towards 1 and
    # check it converges on the Hinton value.
    for alpha, tol in ((1.1, 2e-2), (1.01, 2e-3), (1.0001, 2e-5)):
        near = build_loss("defect_kd", alpha=alpha, **common)
        assert torch.allclose(near(student, targets, ignore, teacher),
                              hinton(student, targets, ignore, teacher), atol=tol)


def test_reweighted_at_alpha_one_equals_plain_bce():
    student, _, targets, ignore = sample()
    plain = build_loss("plain")
    cfg = LossConfig(kind="reweighted", alpha=1.0)
    reweighted = MaskedTileLoss(cfg)
    assert torch.allclose(plain(student, targets, ignore),
                          reweighted(student, targets, ignore))


def test_kd_with_lambda_zero_equals_plain_bce():
    student, teacher, targets, ignore = sample()
    plain = build_loss("plain")
    kd = build_loss("hinton_kd", lam=0.0)
    assert torch.allclose(kd(student, targets, ignore, teacher),
                          plain(student, targets, ignore))


def test_plain_matches_a_hand_written_masked_bce():
    """Independent reimplementation, so the reduction is not self-referential."""
    import torch.nn.functional as F

    student, _, targets, ignore = sample()
    keep = 1.0 - ignore
    expected = ((F.binary_cross_entropy_with_logits(student, targets,
                                                    reduction="none") * keep).sum()
                / keep.sum())
    assert torch.allclose(build_loss("plain")(student, targets, ignore), expected)


def test_defect_kd_differs_from_hinton_when_alpha_is_large():
    student, teacher, targets, ignore = sample()
    hinton = build_loss("hinton_kd", lam=1.0, temperature=2.0)
    defect = build_loss("defect_kd", alpha=5.0, lam=1.0, temperature=2.0)
    assert not torch.allclose(hinton(student, targets, ignore, teacher),
                              defect(student, targets, ignore, teacher))


# --------------------------------------------------------------------------- #
# the ignore mask
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind,kwargs", [
    ("plain", {}), ("reweighted", {"alpha": 3.0}),
    ("hinton_kd", {}), ("defect_kd", {"alpha": 3.0}),
])
def test_ignored_entries_cannot_affect_the_loss(kind, kwargs):
    """Changing the logits at ignored positions must change nothing at all."""
    student, teacher, targets, ignore = sample()
    loss = build_loss(kind, WEIGHTS, use_class_weights=True, **kwargs)
    teacher_arg = teacher if loss.config.needs_teacher else None

    before = loss(student, targets, ignore, teacher_arg)
    perturbed = student.clone()
    perturbed[ignore > 0] += 100.0
    after = loss(perturbed, targets, ignore, teacher_arg)
    assert torch.allclose(before, after), f"{kind} leaked ignored entries"


def test_ignored_entries_get_no_gradient():
    student, _, targets, ignore = sample()
    student = student.clone().requires_grad_(True)
    build_loss("plain")(student, targets, ignore).backward()
    assert torch.all(student.grad[ignore > 0] == 0)
    assert torch.any(student.grad[ignore == 0] != 0)


def test_loss_scale_does_not_depend_on_how_much_is_ignored():
    """The mean is over surviving entries, so masking must not shrink the loss."""
    student, _, targets, _ = sample(ignore_frac=0.0)
    none_ignored = torch.zeros_like(targets)
    half = torch.zeros_like(targets)
    half[: len(half) // 2] = (1.0 - targets[: len(half) // 2])

    loss = build_loss("plain")
    a = loss(student, targets, none_ignored)
    b = loss(student, targets, half)
    # Not equal (different entries), but the same order of magnitude -- a
    # sum-based reduction would roughly halve it.
    assert 0.5 < float(b / a) < 2.0


def test_all_ignored_is_finite_and_does_not_divide_by_zero():
    student, teacher, targets, _ = sample()
    ignore = torch.ones_like(targets)
    for kind, kwargs in [("plain", {}), ("defect_kd", {"alpha": 2.0})]:
        loss = build_loss(kind, **kwargs)
        value = loss(student, targets, ignore,
                     teacher if loss.config.needs_teacher else None)
        assert torch.isfinite(value) and float(value) == 0.0


def test_ignore_may_be_a_bool_tensor():
    student, _, targets, ignore = sample()
    loss = build_loss("plain")
    assert torch.allclose(loss(student, targets, ignore),
                          loss(student, targets, ignore.bool()))


# --------------------------------------------------------------------------- #
# weighting
# --------------------------------------------------------------------------- #

def test_defect_tiles_are_weighted_above_clean_tiles():
    targets = torch.zeros(4, 6)
    targets[1, 2] = 1.0                       # tile 1 holds a defect
    ignore = torch.zeros(4, 6)
    loss = build_loss("reweighted", alpha=4.0, normalize_weights=False)
    weights = loss.entry_weights(targets, 1.0 - ignore)
    assert torch.all(weights[1] == 4.0)
    assert torch.all(weights[0] == 1.0)


def test_normalisation_puts_the_mean_weight_at_one():
    """Otherwise alpha would act as a learning-rate change and confound the ablation."""
    _, _, targets, ignore = sample()
    keep = 1.0 - ignore
    for alpha in (1.0, 2.0, 10.0):
        loss = build_loss("reweighted", alpha=alpha, normalize_weights=True)
        weights = loss.entry_weights(targets, keep)
        assert torch.allclose(weights.sum() / keep.sum(), torch.tensor(1.0), atol=1e-5)


def test_without_normalisation_alpha_inflates_the_loss():
    student, _, targets, ignore = sample()
    small = build_loss("reweighted", alpha=1.0, normalize_weights=False)
    large = build_loss("reweighted", alpha=8.0, normalize_weights=False)
    assert large(student, targets, ignore) > small(student, targets, ignore)


def test_class_weights_apply_to_positive_targets_only():
    targets = torch.zeros(2, 6)
    targets[0, 0] = 1.0
    ignore = torch.zeros(2, 6)
    weights = torch.tensor([9.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    loss = build_loss("plain", weights, use_class_weights=True,
                      normalize_weights=False)
    w = loss.entry_weights(targets, 1.0 - ignore)
    assert float(w[0, 0]) == 9.0, "the positive entry takes the class weight"
    assert float(w[0, 1]) == 1.0, "negative entries do not"


def test_a_tile_counts_as_defect_even_if_other_classes_are_ignored():
    targets = torch.zeros(1, 6)
    targets[0, 1] = 1.0
    ignore = torch.zeros(1, 6)
    ignore[0, 3] = 1.0
    loss = build_loss("defect_kd", alpha=5.0, normalize_weights=False)
    w = loss.entry_weights(targets, 1.0 - ignore)
    assert float(w[0, 0]) == 5.0


# --------------------------------------------------------------------------- #
# distillation behaviour
# --------------------------------------------------------------------------- #

def test_a_student_matching_its_teacher_minimises_the_soft_term():
    _, teacher, targets, ignore = sample()
    loss = build_loss("hinton_kd", lam=1.0, temperature=2.0)
    matched = loss.components(teacher.clone(), targets, ignore, teacher)["soft"]
    for shift in (0.5, 1.0, 2.0):
        worse = loss.components(teacher + shift, targets, ignore, teacher)["soft"]
        assert worse > matched


def test_the_teacher_receives_no_gradient():
    student, teacher, targets, ignore = sample()
    student = student.clone().requires_grad_(True)
    teacher = teacher.clone().requires_grad_(True)
    build_loss("defect_kd", alpha=2.0)(student, targets, ignore, teacher).backward()
    assert teacher.grad is None or torch.all(teacher.grad == 0)
    assert student.grad is not None and torch.any(student.grad != 0)


def test_t_squared_keeps_the_distillation_gradient_scale_stable():
    """The reason T^2 is included: without it, lambda has to absorb T."""
    def soft_grad_norm(temperature, t_squared):
        student = (0.1 * torch.randn(64, 6, generator=torch.Generator().manual_seed(0))
                   ).requires_grad_(True)
        teacher = 0.1 * torch.randn(64, 6, generator=torch.Generator().manual_seed(1))
        targets = torch.zeros(64, 6)
        ignore = torch.zeros(64, 6)
        loss = build_loss("hinton_kd", lam=1.0, temperature=temperature,
                          t_squared=t_squared)
        loss.components(student, targets, ignore, teacher)["soft"].backward()
        return float(student.grad.norm())

    with_t2 = [soft_grad_norm(t, True) for t in (1.0, 2.0, 4.0, 8.0)]
    without = [soft_grad_norm(t, False) for t in (1.0, 2.0, 4.0, 8.0)]
    # With T^2 the scale is roughly flat; without it, it collapses with T.
    assert max(with_t2) / min(with_t2) < 1.5, with_t2
    assert without[0] / without[-1] > 20.0, without


def test_higher_temperature_softens_the_targets():
    _, teacher, _, _ = sample()
    sharp = torch.sigmoid(teacher / 1.0)
    soft = torch.sigmoid(teacher / 8.0)
    assert (soft - 0.5).abs().mean() < (sharp - 0.5).abs().mean()


# --------------------------------------------------------------------------- #
# learning actually happens
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind,kwargs", [
    ("plain", {}), ("reweighted", {"alpha": 3.0}),
    ("hinton_kd", {}), ("defect_kd", {"alpha": 3.0}),
])
def test_each_loss_can_be_driven_down_by_gradient_descent(kind, kwargs):
    student, teacher, targets, ignore = sample(n=16)
    logits = torch.zeros_like(student).requires_grad_(True)
    loss = build_loss(kind, WEIGHTS, use_class_weights=True, **kwargs)
    teacher_arg = teacher if loss.config.needs_teacher else None
    optimiser = torch.optim.SGD([logits], lr=1.0)

    with torch.no_grad():
        first = loss(logits, targets, ignore, teacher_arg).item()
    for _ in range(50):
        optimiser.zero_grad()
        loss(logits, targets, ignore, teacher_arg).backward()
        optimiser.step()
    with torch.no_grad():
        assert loss(logits, targets, ignore, teacher_arg).item() < first


def test_a_confident_correct_student_has_near_zero_hard_loss():
    targets = (torch.rand(32, 6, generator=torch.Generator().manual_seed(0)) < 0.3).float()
    ignore = torch.zeros_like(targets)
    logits = torch.where(targets > 0, 20.0, -20.0)
    assert float(build_loss("plain")(logits, targets, ignore)) < 1e-6


# --------------------------------------------------------------------------- #
# configuration validation
# --------------------------------------------------------------------------- #

def test_plain_and_hinton_refuse_a_defect_weight():
    """Those kinds are *defined* by alpha == 1; silently accepting it would
    make two rows of the run matrix indistinguishable."""
    for kind in ("plain", "hinton_kd"):
        with pytest.raises(ValueError, match="alpha"):
            LossConfig(kind=kind, alpha=2.0)


@pytest.mark.parametrize("kwargs", [
    {"kind": "nonsense"}, {"temperature": 0.0}, {"temperature": -1.0},
    {"alpha": 0.0, "kind": "defect_kd"}, {"lam": -1.0},
])
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs)


def test_class_weights_required_when_requested():
    with pytest.raises(ValueError, match="class_pos_weights"):
        MaskedTileLoss(LossConfig(use_class_weights=True))


@pytest.mark.parametrize("bad", [torch.ones(5), torch.ones(6, 2)])
def test_wrong_class_weight_shape_is_rejected(bad):
    with pytest.raises(ValueError, match="class_pos_weights"):
        MaskedTileLoss(LossConfig(use_class_weights=True), bad)


def test_negative_class_weights_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        MaskedTileLoss(LossConfig(use_class_weights=True),
                       torch.tensor([1.0, -2.0, 1.0, 1.0, 1.0, 1.0]))


def test_kd_kinds_demand_a_teacher():
    student, _, targets, ignore = sample()
    for kind in ("hinton_kd", "defect_kd"):
        loss = build_loss(kind, **({"alpha": 2.0} if kind == "defect_kd" else {}))
        with pytest.raises(ValueError, match="teacher"):
            loss(student, targets, ignore)


@pytest.mark.parametrize("shape", [(8, 5), (7, 6)])
def test_mismatched_shapes_are_rejected(shape):
    student, _, targets, ignore = sample(n=8)
    loss = build_loss("plain")
    with pytest.raises(ValueError, match="match"):
        loss(student, torch.zeros(shape), torch.zeros(shape))
    with pytest.raises(ValueError):
        loss(student, targets, torch.zeros(shape))


def test_every_declared_kind_is_constructible():
    for kind in LOSS_KINDS:
        kwargs = {"alpha": 2.0} if kind in ("reweighted", "defect_kd") else {}
        assert build_loss(kind, **kwargs).config.kind == kind


def test_repr_names_the_configuration():
    text = repr(build_loss("defect_kd", alpha=3.0, lam=0.5, temperature=4.0))
    assert "defect_kd" in text and "3.0" in text
