"""Run a set of training jobs sequentially, skipping ones already finished.

The experiment matrix is dozens of trainings (four loss configurations and
three input modes, each over three seeds), and a job that dies -- a restarted
worker, a full disk, a cancelled session -- should not cost the whole sweep.
A run is considered finished when its ``metrics.json`` exists, so re-invoking
this driver resumes rather than restarting.

Runs go one at a time on purpose. The box has four cores; two concurrent jobs
each asking for worker processes thrash and make every run slower, and -- worse
for the results -- make wall-clock timings incomparable between runs.

Usage::

    python training/run_matrix.py --set input_ablation
    python training/run_matrix.py --set input_ablation --dry-run
    python training/run_matrix.py --set control --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from train import TrainConfig, train  # noqa: E402

SEEDS = (0, 1, 2)

#: Input mode frozen at the step-8 checkpoint (2026-11-04) from the input
#: ablation: modes B and C tied within seed noise (macro AP 0.7994 +/-0.0221 vs
#: 0.8050 +/-0.0333), and C was chosen because Stage 1 already thresholds the
#: eroded XOR, so both stages see the same difference signal.
REFERENCE_MODE = "C"

#: Starting points for the defect-aware KD hyperparameters. The plan requires
#: these to be tuned on validation and the search logged; these are the values
#: the first pass uses, not a tuned result.
DEFAULT_ALPHA = 3.0
DEFAULT_LAMBDA = 1.0
DEFAULT_TEMPERATURE = 2.0


def control_runs() -> list[TrainConfig]:
    """The naive baseline: all 25 tiles, test-only input, plain masked BCE."""
    return [TrainConfig(run_id=f"control_s{s}", representation="A",
                        loss_kind="plain", seed=s) for s in SEEDS]


def input_ablation_runs() -> list[TrainConfig]:
    """Ablation 1: input mode A / B / C, with nothing else changed.

    Mode A is the control, so those runs are shared with the ``control`` set
    and will be skipped if already done.
    """
    runs = list(control_runs())
    for mode in ("B", "C"):
        runs += [TrainConfig(run_id=f"input_{mode}_s{s}", representation=mode,
                             loss_kind="plain", seed=s) for s in SEEDS]
    return runs


def teacher_runs() -> list[TrainConfig]:
    """The distillation teachers, one per seed, on the frozen input mode.

    Trained by the same loop, data, split and schedule as the students, so
    "the teacher is better" is a statement about capacity and input resolution
    rather than about training setup.
    """
    return [TrainConfig(run_id=f"teacher_s{s}", model="teacher",
                        representation=REFERENCE_MODE, loss_kind="plain",
                        seed=s) for s in SEEDS]


def reweight_only_runs() -> list[TrainConfig]:
    """Weighted BCE with no teacher: does weighting alone explain a gain?"""
    return [TrainConfig(run_id=f"reweight_only_s{s}", representation=REFERENCE_MODE,
                        loss_kind="reweighted", alpha=DEFAULT_ALPHA,
                        use_class_weights=True, seed=s) for s in SEEDS]


def reweight_only_nocw_runs() -> list[TrainConfig]:
    """`reweight_only` testing the defect-tile weight *on its own*.

    The plan bundles the defect-tile weight with per-class inverse-frequency
    weights, so the original `reweight_only` moved both at once. The
    reweight_sweep showed the class weights are what does the damage -- alone
    they cost 0.108 macro AP at seed 0, three times the seed spread -- while
    alpha = 3 on its own costs 0.030, which is inside the noise.

    Since step 6 established that class balancing is *not* needed for training
    to work, it is optional here, and the validation evidence says to leave it
    off. That makes this row test what it is named for.
    """
    return [TrainConfig(run_id=f"reweight_only_nocw_s{s}",
                        representation=REFERENCE_MODE, loss_kind="reweighted",
                        alpha=DEFAULT_ALPHA, use_class_weights=False, seed=s)
            for s in SEEDS]


def kd_runs() -> list[TrainConfig]:
    """The prior-method comparison and the novel contribution.

    Both use the *same* teacher for a given seed, so the only difference
    between them is the defect-tile weight.

    Per-class inverse-frequency weights are **off** in both. The plan's `w`
    bundles them with alpha, but measured at three seeds on the frozen input
    mode they cost 0.046 macro AP (0.8028 +/- 0.0207 without them against
    0.7564 +/- 0.0253 with), while alpha = 3 on its own is worth -0.002, a
    tenth of the seed spread. Step 6 established that class balancing is not
    needed for training to work, so it is optional rather than shared setup,
    and this is a validation-driven configuration choice -- logged here, and
    approved at the step-9 checkpoint on 2026-11-04 -- not a change to what
    the rows mean. Both KD rows carry the same setting, so the
    hinton-versus-defect comparison still isolates alpha exactly.
    """
    runs = []
    for seed in SEEDS:
        teacher = f"runs/teacher_s{seed}/best.pt"
        runs.append(TrainConfig(
            run_id=f"hinton_kd_s{seed}", representation=REFERENCE_MODE,
            loss_kind="hinton_kd", lam=DEFAULT_LAMBDA,
            temperature=DEFAULT_TEMPERATURE, use_class_weights=False,
            teacher_ckpt=teacher, seed=seed))
        runs.append(TrainConfig(
            run_id=f"defect_kd_s{seed}", representation=REFERENCE_MODE,
            loss_kind="defect_kd", alpha=DEFAULT_ALPHA, lam=DEFAULT_LAMBDA,
            temperature=DEFAULT_TEMPERATURE, use_class_weights=False,
            teacher_ckpt=teacher, seed=seed))
    return runs


def reweight_sweep_runs() -> list[TrainConfig]:
    """Disentangle the two things `reweight_only` changes at once.

    The plan defines the reweighted loss as the defect-tile weight `alpha`
    *times* per-class inverse-frequency weights, so `reweight_only` moves both
    away from `reference` together. It measured worse than `reference` on every
    class, but that result cannot say which half is responsible, and the plan
    also requires `alpha` to be tuned on validation rather than assumed.

    One seed each: this is a direction-finding sweep, not a reported result.
    """
    variants = [
        ("a1_cw",    1.0, True),    # class weights only
        ("a3_nocw",  3.0, False),   # defect-tile weight only
        ("a15_cw",   1.5, True),    # a gentler version of the full loss
        ("a5_cw",    5.0, True),    # a stronger one
    ]
    return [TrainConfig(run_id=f"rw_{name}_s0", representation=REFERENCE_MODE,
                        loss_kind="reweighted", alpha=alpha,
                        use_class_weights=class_weights, seed=0)
            for name, alpha, class_weights in variants]


RUN_SETS = {
    "control": control_runs,
    "reweight_sweep": reweight_sweep_runs,
    "input_ablation": input_ablation_runs,
    "teacher": teacher_runs,
    "reweight_only": reweight_only_runs,
    "reweight_only_nocw": reweight_only_nocw_runs,
    "kd": kd_runs,
}


def is_finished(config: TrainConfig) -> bool:
    return (config.run_dir / "metrics.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="run_set", required=True, choices=list(RUN_SETS))
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="rerun finished jobs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = RUN_SETS[args.run_set]()
    for config in runs:
        config.out_dir = args.out_dir
        config.num_workers = args.num_workers
        config.epochs = args.epochs

    missing_teachers = sorted({
        c.teacher_ckpt for c in runs
        if c.teacher_ckpt and not Path(c.teacher_ckpt).exists()
    })
    if missing_teachers:
        print("\nMissing teacher checkpoints -- train them first with "
              "`--set teacher`:")
        for path in missing_teachers:
            print(f"  {path}")
        raise SystemExit(1)

    pending = [c for c in runs if args.force or not is_finished(c)]
    print(f"run set {args.run_set!r}: {len(runs)} jobs, {len(pending)} to run")
    for config in runs:
        state = "RUN " if config in pending else "skip"
        print(f"  {state} {config.run_id:<18} {config.model:<8} input "
              f"{config.representation}  {config.loss_kind:<10} "
              f"alpha {config.alpha:<4g} seed {config.seed}")
    if args.dry_run:
        return

    started = time.time()
    for index, config in enumerate(pending, start=1):
        print(f"\n===== [{index}/{len(pending)}] {config.run_id} "
              f"({time.time() - started:.0f}s elapsed) =====", flush=True)
        try:
            train(config)
        except Exception as exc:                      # keep the sweep going
            print(f"  FAILED {config.run_id}: {type(exc).__name__}: {exc}",
                  flush=True)
            (config.run_dir).mkdir(parents=True, exist_ok=True)
            (config.run_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    print(f"\nrun set {args.run_set!r} done in {time.time() - started:.0f}s")
    summary = {}
    for config in runs:
        path = config.run_dir / "metrics.json"
        if path.exists():
            data = json.loads(path.read_text())
            summary[config.run_id] = {
                "model": config.model,
                "representation": config.representation,
                "loss_kind": config.loss_kind,
                "alpha": config.alpha,
                "lam": config.lam,
                "temperature": config.temperature,
                "use_class_weights": config.use_class_weights,
                "seed": config.seed,
                "best_val_macro_ap": data.get("best_val_macro_ap"),
                "best_epoch": data.get("best_epoch"),
                "epochs_run": data.get("epochs_run"),
            }
    out = Path(args.out_dir) / f"summary_{args.run_set}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")
    for run_id, row in summary.items():
        ap_value = row["best_val_macro_ap"]
        print(f"  {run_id:<16} input {row['representation']}  "
              f"val_mAP {ap_value:.4f}" if ap_value is not None
              else f"  {run_id:<16} (no result)")


if __name__ == "__main__":
    main()
