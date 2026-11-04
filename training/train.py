"""Train loops for the Stage-2 student.

One entry point for every row of the experiment matrix. A run is fully
described by a :class:`TrainConfig`, which is written next to its checkpoint,
so any result can be regenerated from the saved JSON alone::

    python training/train.py --run-id control_s0 --loss plain --representation A
    python training/train.py --config runs/control_s0/config.json

What is held fixed across runs, per the experiment plan: the student
architecture, the split, labels, tiling, sampling and augmentation, the
optimiser, learning rate, batch size, epoch budget and early stopping, and the
metric code. Only the setting under test moves.

Checkpoint selection uses **validation macro average precision**, which is
threshold-free. Selecting on recall at a fixed threshold would entangle model
selection with the threshold choice, and thresholds are only set afterwards,
on validation int8 outputs, to hit an agreed board-level false-alarm rate.

Batching is per *image*: one item is one augmented board's tiles, because
augmentation and tiling are image-level work. Per-tile batching repeats that
work 25 times and costs 5.6 minutes an epoch instead of a few seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))
sys.path.insert(0, str(_REPO / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from losses import LOSS_KINDS, LossConfig, MaskedTileLoss  # noqa: E402
from metrics import tile_average_precision  # noqa: E402
from models import TileStudent  # noqa: E402
from teachers import build_teacher  # noqa: E402
from tile_dataset import (  # noqa: E402
    AugmentConfig,
    TileData,
    TileImageDataset,
    collate_tiles,
)

DEFAULT_NPZ = "data/tiles/deeppcb_tiles_t160_s120.npz"


@dataclass
class TrainConfig:
    """Everything needed to reproduce one run."""

    run_id: str = "control_s0"
    model: str = "student"             # "student" (Stage 2) or "teacher" (KD)

    # --- the setting under test -------------------------------------------
    representation: str = "A"          # A: test only, B: +template, C: +XOR
    loss_kind: str = "plain"
    alpha: float = 1.0                 # defect-tile weight
    lam: float = 1.0                   # distillation weight
    temperature: float = 2.0
    use_class_weights: bool = False
    teacher_ckpt: str | None = None

    # --- held fixed across runs -------------------------------------------
    seed: int = 0
    epochs: int = 30
    batch_images: int = 8              # x 25 tiles = 200 tiles per step
    tiles_per_image: int | None = None
    drop_clean_frac: float = 0.0
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    early_stop_patience: int = 8
    grad_clip: float = 5.0
    num_workers: int = 4

    # --- plumbing ----------------------------------------------------------
    npz: str = DEFAULT_NPZ
    out_dir: str = "runs"
    limit_train_images: int | None = None   # smoke tests only
    limit_val_images: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.representation not in ("A", "B", "C"):
            raise ValueError(f"representation must be A, B or C, got {self.representation!r}")
        if self.loss_kind not in LOSS_KINDS:
            raise ValueError(f"loss_kind must be one of {LOSS_KINDS}")
        if self.model not in ("student", "teacher"):
            raise ValueError(f"model must be 'student' or 'teacher', got {self.model!r}")
        if self.model == "teacher" and self.loss_config().needs_teacher:
            raise ValueError("a teacher cannot be trained with a distillation loss")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")

    @property
    def in_channels(self) -> int:
        return 1 if self.representation == "A" else 2

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_id

    def loss_config(self) -> LossConfig:
        return LossConfig(
            kind=self.loss_kind, alpha=self.alpha, lam=self.lam,
            temperature=self.temperature, use_class_weights=self.use_class_weights,
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text()))


RESUME_FILE = "state.pt"


def save_resume_state(run_dir: Path, model, optimiser, epoch: int,
                      history: list, best_ap: float, best_epoch: int,
                      since_best: int) -> None:
    """Snapshot after each epoch so a restart does not lose the run.

    A teacher run takes hours on CPU and this session's worker has already been
    restarted mid-sweep more than once, so losing a run to that is a real cost,
    not a hypothetical one. Written to a temporary file and moved into place so
    a crash during the write cannot leave a half-written snapshot.
    """
    tmp = run_dir / (RESUME_FILE + ".tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "history": [asdict(r) for r in history],
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "since_best": since_best,
        "torch_rng": torch.get_rng_state(),
    }, tmp)
    tmp.replace(run_dir / RESUME_FILE)


def load_resume_state(run_dir: Path, model, optimiser):
    """Restore a snapshot, or return None if there is none to restore."""
    path = run_dir / RESUME_FILE
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimiser.load_state_dict(state["optimiser"])
    torch.set_rng_state(state["torch_rng"])
    return state


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_hard: float
    train_soft: float
    val_loss: float
    val_macro_ap: float
    val_per_class_ap: list[float] = field(default_factory=list)
    seconds: float = 0.0
    lr: float = 0.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_loaders(config: TrainConfig):
    """Train and validation loaders, plus the class weights from the train fold."""
    train_fold = TileData.load(config.npz, "train")
    val_fold = TileData.load(config.npz, "val")

    if config.limit_train_images:
        train_fold = _head(train_fold, config.limit_train_images)
    if config.limit_val_images:
        val_fold = _head(val_fold, config.limit_val_images)

    train_ds = TileImageDataset(
        train_fold, config.representation, AugmentConfig.train_default(),
        seed=config.seed, tiles_per_image=config.tiles_per_image,
        drop_clean_frac=config.drop_clean_frac,
    )
    # No augmentation on validation: the comparison must not move between runs.
    val_ds = TileImageDataset(val_fold, config.representation, aug=None,
                              seed=config.seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_images, shuffle=True,
        num_workers=config.num_workers, collate_fn=collate_tiles,
        drop_last=False, persistent_workers=config.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_images, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_tiles,
        persistent_workers=config.num_workers > 0,
    )
    return train_ds, train_loader, val_loader, train_fold.class_pos_weights()


def _head(fold: TileData, n: int) -> TileData:
    """First ``n`` images of a fold, for smoke tests."""
    return TileData(
        test=fold.test[:n], temp=fold.temp[:n], y=fold.y[:n],
        ignore=fold.ignore[:n], boxes=fold.boxes[:n], names=fold.names[:n],
        groups=fold.groups[:n], grid=fold.grid, meta=fold.meta,
    )


def build_model(config: TrainConfig) -> torch.nn.Module:
    """The network this run trains.

    The teacher is trained by the same loop as the student, on the same data,
    split and schedule, so that "the teacher is better" is a statement about
    capacity and input resolution rather than about training setup.
    """
    if config.model == "teacher":
        return build_teacher(config.in_channels)
    return TileStudent(in_channels=config.in_channels)


def load_teacher(config: TrainConfig, device: torch.device):
    """The teacher for the KD runs.

    Kept deliberately dumb: the checkpoint must already match the frozen input
    mode, so a KD run cannot silently distil from a teacher that saw different
    inputs than the student.
    """
    if not config.loss_config().needs_teacher:
        return None
    if not config.teacher_ckpt:
        raise ValueError(f"loss_kind={config.loss_kind!r} requires --teacher-ckpt")
    teacher = build_teacher(config.in_channels)
    teacher.load_state_dict(torch.load(config.teacher_ckpt, map_location="cpu"))
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, teacher=None) -> tuple[float, np.ndarray, float]:
    """Validation loss and threshold-free average precision."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    scores, targets, ignores = [], [], []
    for x, y, ignore in loader:
        x, y, ignore = x.to(device), y.to(device), ignore.to(device)
        logits = model(x)
        teacher_logits = teacher(x) if teacher is not None else None
        total_loss += float(loss_fn(logits, y, ignore, teacher_logits))
        n_batches += 1
        scores.append(logits.cpu().numpy())
        targets.append(y.cpu().numpy())
        ignores.append(ignore.cpu().numpy())

    per_class, macro = tile_average_precision(
        np.concatenate(scores), np.concatenate(targets),
        np.concatenate(ignores) > 0.5,
    )
    return total_loss / max(n_batches, 1), per_class, macro


def train(config: TrainConfig, verbose: bool = True, resume: bool = True) -> dict:
    """Run one training job. Returns the history and the best metrics.

    With ``resume`` (the default) an interrupted run continues from its
    per-epoch snapshot rather than starting over.
    """
    set_seed(config.seed)
    device = torch.device(config.device)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.json")

    train_ds, train_loader, val_loader, class_weights = make_loaders(config)
    model = build_model(config).to(device)
    teacher = load_teacher(config, device)
    loss_fn = MaskedTileLoss(config.loss_config(),
                             torch.as_tensor(class_weights, dtype=torch.float32)
                             ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    steps_per_epoch = max(1, len(train_loader))

    def lr_at(epoch: int) -> float:
        """Linear warmup then cosine decay, so the schedule is run-independent."""
        if epoch < config.warmup_epochs:
            return config.lr * (epoch + 1) / max(1, config.warmup_epochs)
        progress = (epoch - config.warmup_epochs) / max(
            1, config.epochs - config.warmup_epochs)
        return config.lr * 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    history: list[EpochRecord] = []
    best_ap, best_epoch, since_best = -np.inf, -1, 0
    start_epoch = 0

    if resume:
        state = load_resume_state(run_dir, model, optimiser)
        if state is not None:
            start_epoch = state["epoch"] + 1
            history = [EpochRecord(**r) for r in state["history"]]
            best_ap, best_epoch = state["best_ap"], state["best_epoch"]
            since_best = state["since_best"]
            if verbose:
                print(f"resuming {config.run_id} at epoch {start_epoch} "
                      f"(best val_mAP {best_ap:.4f} at epoch {best_epoch})")

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"run {config.run_id}: {config.model}, {config.loss_kind} loss, input "
              f"{config.representation} ({config.in_channels}ch), seed {config.seed}")
        print(f"  {n_params:,} parameters, {len(train_ds)} train images x "
              f"{train_ds.data.n_tiles} tiles, {steps_per_epoch} steps/epoch, "
              f"device {config.device}")
        print(f"  class_pos_weights {np.array2string(class_weights, precision=1)}")

    for epoch in range(start_epoch, config.epochs):
        started = time.time()
        lr = lr_at(epoch)
        for group in optimiser.param_groups:
            group["lr"] = lr

        train_ds.set_epoch(epoch)
        model.train()
        sums = {"total": 0.0, "hard": 0.0, "soft": 0.0}
        n_steps = 0
        for x, y, ignore in train_loader:
            x, y, ignore = x.to(device), y.to(device), ignore.to(device)
            with torch.no_grad():
                teacher_logits = teacher(x) if teacher is not None else None
            parts = loss_fn.components(model(x), y, ignore, teacher_logits)
            optimiser.zero_grad(set_to_none=True)
            parts["total"].backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            for key in sums:
                sums[key] += float(parts[key].detach())
            n_steps += 1

        val_loss, per_class_ap, macro_ap = evaluate(
            model, val_loader, loss_fn, device, teacher)
        record = EpochRecord(
            epoch=epoch,
            train_loss=sums["total"] / max(n_steps, 1),
            train_hard=sums["hard"] / max(n_steps, 1),
            train_soft=sums["soft"] / max(n_steps, 1),
            val_loss=val_loss, val_macro_ap=macro_ap,
            val_per_class_ap=[None if np.isnan(v) else round(float(v), 5)
                              for v in per_class_ap],
            seconds=time.time() - started, lr=lr,
        )
        history.append(record)

        if macro_ap > best_ap:
            best_ap, best_epoch, since_best = macro_ap, epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            since_best += 1

        if verbose:
            mark = " *" if best_epoch == epoch else ""
            print(f"  epoch {epoch:>3}  lr {lr:.2e}  train {record.train_loss:.4f}  "
                  f"val {val_loss:.4f}  val_mAP {macro_ap:.4f}  "
                  f"{record.seconds:.1f}s{mark}")

        save_resume_state(run_dir, model, optimiser, epoch, history,
                          best_ap, best_epoch, since_best)

        if since_best >= config.early_stop_patience:
            if verbose:
                print(f"  early stop: no improvement for "
                      f"{config.early_stop_patience} epochs")
            break

    torch.save(model.state_dict(), run_dir / "last.pt")
    result = {
        "run_id": config.run_id,
        "best_epoch": best_epoch,
        "best_val_macro_ap": None if not np.isfinite(best_ap) else float(best_ap),
        "epochs_run": len(history),
        "history": [asdict(r) for r in history],
        "class_pos_weights": [float(v) for v in class_weights],
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    # The run is complete; the snapshot is only needed to recover an unfinished one.
    (run_dir / RESUME_FILE).unlink(missing_ok=True)
    if verbose:
        print(f"  best val_mAP {best_ap:.4f} at epoch {best_epoch}; "
              f"wrote {run_dir}/best.pt")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="reproduce a run from its config.json")
    ap.add_argument("--run-id")
    ap.add_argument("--model", choices=["student", "teacher"])
    ap.add_argument("--representation", choices=["A", "B", "C"])
    ap.add_argument("--loss", dest="loss_kind", choices=list(LOSS_KINDS))
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--class-weights", dest="use_class_weights",
                    action="store_true", default=None)
    ap.add_argument("--teacher-ckpt")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-images", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--npz")
    ap.add_argument("--out-dir")
    ap.add_argument("--device")
    ap.add_argument("--limit-train-images", type=int)
    ap.add_argument("--limit-val-images", type=int)
    args = ap.parse_args()

    if args.config:
        config = TrainConfig.load(args.config)
        overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k != "config"}
        if overrides:
            config = TrainConfig(**{**asdict(config), **overrides})
    else:
        provided = {k: v for k, v in vars(args).items()
                    if v is not None and k != "config"}
        config = TrainConfig(**provided)
    train(config)


if __name__ == "__main__":
    main()
