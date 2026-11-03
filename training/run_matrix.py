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


RUN_SETS = {
    "control": control_runs,
    "input_ablation": input_ablation_runs,
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

    pending = [c for c in runs if args.force or not is_finished(c)]
    print(f"run set {args.run_set!r}: {len(runs)} jobs, {len(pending)} to run")
    for config in runs:
        state = "RUN " if config in pending else "skip"
        print(f"  {state} {config.run_id:<16} input {config.representation}  "
              f"{config.loss_kind:<10} seed {config.seed}")
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
                "representation": config.representation,
                "loss_kind": config.loss_kind,
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
