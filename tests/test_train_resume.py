"""Tests for the training loop's model selection and crash recovery.

A teacher run takes about 4.6 hours on this machine's four cores, and this
session's worker has already been restarted mid-sweep more than once, so
losing a run to a restart is a measured cost rather than a hypothetical one.
These tests check that a resumed run picks up where it stopped and reaches the
same place as an uninterrupted one.
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "training"))

from train import (  # noqa: E402
    RESUME_FILE,
    TrainConfig,
    build_model,
    load_resume_state,
    save_resume_state,
    train,
)

NPZ = _REPO / "data" / "tiles" / "deeppcb_tiles_t160_s120.npz"


def tiny_config(tmp_path, **overrides) -> TrainConfig:
    base = dict(
        run_id="t", representation="A", loss_kind="plain", seed=0, epochs=2,
        batch_images=2, limit_train_images=4, limit_val_images=2,
        num_workers=0, out_dir=str(tmp_path), npz=str(NPZ),
        early_stop_patience=99,
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_build_model_selects_student_or_teacher():
    student = build_model(TrainConfig(run_id="a", model="student"))
    teacher = build_model(TrainConfig(run_id="b", model="teacher"))
    n_student = sum(p.numel() for p in student.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    assert n_student < 100_000 < n_teacher


def test_teacher_input_channels_follow_the_input_mode():
    for representation, channels in (("A", 1), ("C", 2)):
        model = build_model(TrainConfig(run_id="x", model="teacher",
                                        representation=representation))
        assert model.conv1.in_channels == channels


def test_a_teacher_cannot_be_trained_with_a_distillation_loss():
    """It would need a teacher of its own, which is nonsense."""
    with pytest.raises(ValueError, match="teacher cannot be trained"):
        TrainConfig(run_id="x", model="teacher", loss_kind="hinton_kd")


def test_invalid_model_kind_is_rejected():
    with pytest.raises(ValueError, match="model must be"):
        TrainConfig(run_id="x", model="resnet50")


def test_snapshot_round_trips(tmp_path):
    model = build_model(TrainConfig(run_id="x"))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Take a step so the weights and optimiser state are not at their defaults.
    model(torch.zeros(1, 1, 160, 160)).sum().backward()
    optimiser.step()
    save_resume_state(tmp_path, model, optimiser, epoch=3, history=[],
                      best_ap=0.5, best_epoch=2, since_best=1)
    assert (tmp_path / RESUME_FILE).exists()

    restored = build_model(TrainConfig(run_id="x"))
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_resume_state(tmp_path, restored, restored_opt)
    assert state["epoch"] == 3 and state["best_ap"] == 0.5
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_loading_a_missing_snapshot_returns_none(tmp_path):
    model = build_model(TrainConfig(run_id="x"))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert load_resume_state(tmp_path, model, optimiser) is None


@pytest.mark.skipif(not NPZ.exists(), reason="tile dataset not built")
def test_an_interrupted_run_resumes_instead_of_restarting(tmp_path):
    """A snapshot left at epoch 0 must make a 3-epoch run train only 2 more.

    The snapshot is written by hand to stand in for a run killed after its
    first epoch, which is exactly what a restarted worker produces.
    """
    from train import EpochRecord

    config = tiny_config(tmp_path, run_id="interrupted", epochs=3)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr)
    first_epoch = EpochRecord(epoch=0, train_loss=0.9, train_hard=0.9,
                              train_soft=0.0, val_loss=0.9, val_macro_ap=0.11)
    save_resume_state(config.run_dir, model, optimiser, epoch=0,
                      history=[first_epoch], best_ap=0.11, best_epoch=0,
                      since_best=0)

    result = train(config, verbose=False)
    # One restored epoch plus the two it still had to run.
    assert result["epochs_run"] == 3
    assert [r["epoch"] for r in result["history"]] == [0, 1, 2]
    assert result["history"][0]["val_macro_ap"] == 0.11, "restored epoch kept"


@pytest.mark.skipif(not NPZ.exists(), reason="tile dataset not built")
def test_resume_can_be_turned_off(tmp_path):
    from train import EpochRecord

    config = tiny_config(tmp_path, run_id="fresh", epochs=2)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr)
    save_resume_state(config.run_dir, model, optimiser, epoch=0,
                      history=[EpochRecord(0, 0.9, 0.9, 0.0, 0.9, 0.11)],
                      best_ap=0.11, best_epoch=0, since_best=0)

    result = train(config, verbose=False, resume=False)
    assert [r["epoch"] for r in result["history"]] == [0, 1], "ignored the snapshot"


@pytest.mark.skipif(not NPZ.exists(), reason="tile dataset not built")
def test_a_finished_run_deletes_its_snapshot(tmp_path):
    config = tiny_config(tmp_path, run_id="done", epochs=1)
    train(config, verbose=False)
    assert (config.run_dir / "metrics.json").exists()
    assert not (config.run_dir / RESUME_FILE).exists()


@pytest.mark.skipif(not NPZ.exists(), reason="tile dataset not built")
def test_resuming_a_completed_epoch_count_is_a_no_op(tmp_path):
    """Re-running with the same epoch budget must not train further."""
    config = tiny_config(tmp_path, run_id="again", epochs=1)
    first = train(config, verbose=False)
    second = train(config, verbose=False)
    assert first["epochs_run"] == second["epochs_run"] == 1


def test_snapshot_is_written_atomically(tmp_path):
    """Via a temporary file, so a crash mid-write cannot corrupt it."""
    model = build_model(TrainConfig(run_id="x"))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_resume_state(tmp_path, model, optimiser, 0, [], 0.0, 0, 0)
    assert not list(tmp_path.glob("*.tmp")), "temporary file must be moved into place"


def test_state_snapshots_are_not_committed():
    """A teacher snapshot carries optimiser moments: ~134 MB."""
    ignore = (_REPO / ".gitignore").read_text()
    assert "runs/**/state.pt" in ignore
