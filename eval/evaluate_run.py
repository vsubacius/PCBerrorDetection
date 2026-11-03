"""Evaluate a trained run on the PC, in float32.

Produces the numbers the experiment plan asks for, on whichever fold is named:

* per-class and macro average precision (threshold-free);
* defect-level recall and precision at chosen per-class thresholds;
* false alarms on the same fold's **templates run as defect-free boards**,
  which is where the gate's value shows and where thresholds are set;
* the diagnostics behind the class-imbalance decision -- how often the model
  fires at all, against how often it should.

Thresholds are chosen on the *validation* fold to hit a target board-level
false-alarm rate and then reused unchanged. Never pass ``--fold test`` while
tuning anything; it exists so the frozen configuration can be scored once.

Usage::

    python eval/evaluate_run.py --run runs/control_s0 --fold val
    python eval/evaluate_run.py --run runs/control_s0 --fold val --target-fa 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))
sys.path.insert(0, str(_REPO / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (  # noqa: E402
    CLASS_ORDER,
    BoardPrediction,
    choose_thresholds_for_false_alarm_rate,
    evaluate_clean_boards,
    evaluate_defects,
    reachability_ceiling,
    tile_average_precision,
)
from models import TileStudent  # noqa: E402
from tile_dataset import TileData  # noqa: E402
from train import TrainConfig  # noqa: E402


@torch.no_grad()
def score_fold(model, data: TileData, representation: str, device, clean: bool = False):
    """Per-board logits for every tile of a fold.

    With ``clean=True`` the *template* is used as the test image, so the pair
    matches everywhere and every tile is genuinely defect-free. That is how the
    500 test templates become defect-free boards.
    """
    model.eval()
    out = []
    for i in range(len(data)):
        if clean:
            saved = data.test[i]
            data.test[i] = data.temp[i]
        x, _, _ = data.tiles(i, representation, None)
        if clean:
            data.test[i] = saved
        logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
        out.append(logits)
    return out


def imbalance_diagnostics(scores, data: TileData, threshold: float = 0.0) -> dict:
    """Is the model simply predicting negatives?

    ``threshold`` is in logit space; 0.0 corresponds to a probability of 0.5.
    """
    all_scores = np.concatenate(scores)
    y = data.y.reshape(-1, len(CLASS_ORDER))
    ignore = data.ignore.reshape(-1, len(CLASS_ORDER)).astype(bool)
    keep = ~ignore

    fired = (all_scores > threshold) & keep
    actual = (y > 0) & keep
    probs = 1.0 / (1.0 + np.exp(-all_scores[keep]))

    return {
        "threshold_logit": threshold,
        "entries_scored": int(keep.sum()),
        "true_positive_rate": float(actual.sum() / keep.sum()),
        "predicted_positive_rate": float(fired.sum() / keep.sum()),
        "n_predicted_positive": int(fired.sum()),
        "n_actually_positive": int(actual.sum()),
        "recall_at_threshold": float((fired & actual).sum() / max(actual.sum(), 1)),
        "max_probability": float(probs.max()),
        "mean_probability": float(probs.mean()),
        "p99_probability": float(np.quantile(probs, 0.99)),
        "share_of_positives_scoring_above_half": float(
            (all_scores[actual] > 0.0).mean()) if actual.sum() else float("nan"),
    }


def evaluate_run(run_dir: Path, fold: str, target_fa: float,
                 thresholds: np.ndarray | None = None,
                 device: str = "cpu") -> dict:
    config = TrainConfig.load(run_dir / "config.json")
    data = TileData.load(config.npz, fold)
    model = TileStudent(in_channels=config.in_channels)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.to(torch.device(device))

    defective = score_fold(model, data, config.representation, device)
    clean = score_fold(model, data, config.representation, device, clean=True)

    per_class_ap, macro_ap = tile_average_precision(
        np.concatenate(defective),
        data.y.reshape(-1, len(CLASS_ORDER)),
        data.ignore.reshape(-1, len(CLASS_ORDER)).astype(bool),
    )

    clean_preds = [BoardPrediction(scores=s) for s in clean]
    if thresholds is None:
        thresholds = choose_thresholds_for_false_alarm_rate(
            clean_preds, data.grid, target_fa)

    defect_res = evaluate_defects(
        data.boxes, [BoardPrediction(scores=s) for s in defective],
        data.grid, thresholds,
        labels_per_board=[(data.y[i], data.ignore[i]) for i in range(len(data))],
    )
    clean_res = evaluate_clean_boards(clean_preds, data.grid, thresholds)
    ceiling, reachable, total = reachability_ceiling(data.boxes, data.grid)

    return {
        "run_id": config.run_id,
        "fold": fold,
        "precision_mode": "float32 (PC)",
        "target_board_false_alarm_rate": target_fa,
        "thresholds_logit": [float(t) for t in thresholds],
        "macro_ap": macro_ap,
        "per_class_ap": {n: (None if np.isnan(v) else float(v))
                         for n, v in zip(CLASS_ORDER, per_class_ap)},
        "reachability_ceiling": ceiling,
        "defects_unreachable": total - reachable,
        "defect_recall": defect_res.recall,
        "defect_precision": defect_res.precision,
        "per_class_recall": {n: (None if np.isnan(v) else float(v))
                             for n, v in zip(CLASS_ORDER, defect_res.per_class_recall)},
        "n_defects": defect_res.n_defects,
        "n_found": defect_res.n_found,
        "true_positives": defect_res.true_positives,
        "false_positives": defect_res.false_positives,
        "ignored_firings": defect_res.ignored_firings,
        "clean_board_false_alarm_rate": clean_res.board_false_alarm_rate,
        "clean_tile_false_alarm_rate": clean_res.tile_false_alarm_rate,
        "imbalance": imbalance_diagnostics(defective, data),
        "_defect_report": defect_res.describe(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", default="val", choices=["train", "val", "test"])
    ap.add_argument("--target-fa", type=float, default=0.05,
                    help="target board-level false-alarm rate on clean boards")
    ap.add_argument("--thresholds", default=None,
                    help="JSON list of 6 logit thresholds to reuse instead")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    thresholds = (np.array(json.loads(args.thresholds), dtype=float)
                  if args.thresholds else None)
    result = evaluate_run(Path(args.run), args.fold, args.target_fa,
                          thresholds, args.device)

    print(f"=== {result['run_id']}  fold={result['fold']}  "
          f"{result['precision_mode']} ===\n")
    print(f"macro AP {result['macro_ap']:.4f}")
    for name, value in result["per_class_ap"].items():
        print(f"  AP {name:<12}{value:.4f}" if value is not None
              else f"  AP {name:<12}  n/a")
    print()
    print(result["_defect_report"])
    print(f"\nreachability ceiling on this fold: "
          f"{result['reachability_ceiling'] * 100:.2f}% "
          f"({result['defects_unreachable']} unreachable)")
    print(f"\nthresholds (logit): "
          f"{np.array2string(np.array(result['thresholds_logit']), precision=3)}")
    print(f"clean-board false alarms: "
          f"{result['clean_board_false_alarm_rate'] * 100:.2f}% of boards "
          f"(target {args.target_fa * 100:.0f}%), "
          f"{result['clean_tile_false_alarm_rate'] * 100:.3f}% of tiles")

    imb = result["imbalance"]
    print(f"\n--- class-imbalance diagnostics (threshold p=0.5) ---")
    print(f"  true positive rate      {imb['true_positive_rate'] * 100:>7.3f}%  "
          f"({imb['n_actually_positive']} of {imb['entries_scored']} entries)")
    print(f"  predicted positive rate {imb['predicted_positive_rate'] * 100:>7.3f}%  "
          f"({imb['n_predicted_positive']} entries)")
    print(f"  recall at p=0.5         {imb['recall_at_threshold'] * 100:>7.2f}%")
    print(f"  positives scoring p>0.5 {imb['share_of_positives_scoring_above_half'] * 100:>7.2f}%")
    print(f"  probabilities: mean {imb['mean_probability']:.4f}  "
          f"p99 {imb['p99_probability']:.4f}  max {imb['max_probability']:.4f}")

    out = Path(args.out) if args.out else Path(args.run) / f"eval_{args.fold}.json"
    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
