"""Aggregate several runs into one comparison table, with mean and spread.

The experiment plan requires every training to be repeated over three seeds and
reported as mean and spread, because a difference smaller than the seed spread
is not a result. This script evaluates a group of runs (reusing a cached
``eval_<fold>.json`` when present), groups them by whatever is being varied,
and prints both the aggregate and the per-class breakdown.

Usage::

    python eval/compare_runs.py --runs runs/control_s* runs/input_B_s* runs/input_C_s*
    python eval/compare_runs.py --runs runs/* --group-by representation --force
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_run import evaluate_run  # noqa: E402
from metrics import CLASS_ORDER  # noqa: E402
from train import TrainConfig  # noqa: E402


def load_or_evaluate(run_dir: Path, fold: str, target_fa: float,
                     force: bool = False) -> dict:
    cached = run_dir / f"eval_{fold}.json"
    if cached.exists() and not force:
        payload = json.loads(cached.read_text())
        if payload.get("target_board_false_alarm_rate") == target_fa:
            return payload
    result = evaluate_run(run_dir, fold, target_fa)
    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    cached.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return payload


def summarise(values: list[float]) -> tuple[float, float]:
    """``(mean, sample standard deviation)``; sd is 0 for a single value."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--fold", default="val", choices=["train", "val", "test"])
    ap.add_argument("--target-fa", type=float, default=0.05)
    ap.add_argument("--group-by", default="representation",
                    help="TrainConfig field whose value names each group")
    ap.add_argument("--force", action="store_true", help="re-evaluate, ignore cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    for pattern in args.runs:
        run_dir = Path(pattern)
        if not (run_dir / "config.json").exists():
            print(f"  skip {run_dir} (no config.json)")
            continue
        config = TrainConfig.load(run_dir / "config.json")
        key = str(getattr(config, args.group_by))
        print(f"  evaluating {run_dir.name} ({args.group_by}={key})", flush=True)
        groups[key].append(load_or_evaluate(run_dir, args.fold, args.target_fa,
                                            args.force))

    print(f"\n=== {args.fold} fold, float32, grouped by {args.group_by} ===")
    print(f"thresholds set per run to a {args.target_fa:.0%} board false-alarm "
          f"target on the same fold's templates\n")
    head = (f"{args.group_by:>14}{'n':>4}{'macro AP':>18}{'defect recall':>20}"
            f"{'precision':>18}{'clean-board FA':>17}")
    print(head)
    print("-" * len(head))
    aggregate = {}
    for key in sorted(groups):
        rows = groups[key]
        ap_mean, ap_sd = summarise([r["macro_ap"] for r in rows])
        rc_mean, rc_sd = summarise([r["defect_recall"] for r in rows])
        pr_mean, pr_sd = summarise([r["defect_precision"] for r in rows])
        fa_mean, fa_sd = summarise([r["clean_board_false_alarm_rate"] for r in rows])
        aggregate[key] = {
            "n_seeds": len(rows),
            "macro_ap": [ap_mean, ap_sd],
            "defect_recall": [rc_mean, rc_sd],
            "defect_precision": [pr_mean, pr_sd],
            "clean_board_fa": [fa_mean, fa_sd],
            "runs": [r["run_id"] for r in rows],
        }
        print(f"{key:>14}{len(rows):>4}"
              f"{ap_mean:>11.4f} ±{ap_sd:.4f}"
              f"{rc_mean * 100:>13.2f}% ±{rc_sd * 100:.2f}"
              f"{pr_mean * 100:>11.1f}% ±{pr_sd * 100:.1f}"
              f"{fa_mean * 100:>11.1f}% ±{fa_sd * 100:.1f}")

    print(f"\nper-class average precision (mean ± sd over seeds)")
    print(f"{args.group_by:>14}" + "".join(f"{n:>18}" for n in CLASS_ORDER))
    print("-" * (14 + 18 * len(CLASS_ORDER)))
    per_class = {}
    for key in sorted(groups):
        cells, store = [], {}
        for name in CLASS_ORDER:
            mean, sd = summarise([r["per_class_ap"].get(name) for r in groups[key]])
            store[name] = [mean, sd]
            cells.append(f"{mean:>11.3f} ±{sd:.3f}")
        per_class[key] = store
        print(f"{key:>14}" + "".join(cells))

    keys = sorted(groups)
    if len(keys) > 1:
        base = keys[0]
        print(f"\nchange in per-class AP relative to {args.group_by}={base}:")
        print(f"{args.group_by:>14}" + "".join(f"{n:>13}" for n in CLASS_ORDER))
        for key in keys[1:]:
            deltas = [per_class[key][n][0] - per_class[base][n][0] for n in CLASS_ORDER]
            print(f"{key:>14}" + "".join(f"{d:>+13.3f}" for d in deltas))
        print("\nA difference smaller than the seed spread is not a result.")

    payload = {"fold": args.fold, "target_fa": args.target_fa,
               "group_by": args.group_by, "aggregate": aggregate,
               "per_class_ap": per_class}
    out = Path(args.out) if args.out else Path("runs") / f"compare_{args.group_by}_{args.fold}.json"
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
