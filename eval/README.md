# eval/

| Module | Purpose |
| --- | --- |
| `metrics.py` | **the one shared metric module** — every run in the matrix is scored here |
| `gate_analysis.py` | Stage-1 classical gate separability vs misalignment |

## `metrics.py`

### The defect-level rule

A defect counts as **found** if a tile holding at least `pos_thresh` (0.70) of its area
predicts that defect's class above threshold — deliberately the *same* rule the tile labels
use, so the metric measures the model rather than the grid.

The older rule in `CLAUDE.md` (a tile *fully containing* the defect) caps recall at 95.17% on
trainval and 95.89% on test for purely geometric reasons, and is not used.

Measured ceilings under the 0.70 rule:

| split | defects | unreachable | recall ceiling |
| --- | --- | --- | --- |
| trainval | 6,873 | 5 | 99.93% |
| **test (official)** | 3,140 | **0** | **100.00%** |

All five unreachable defects (best containment 0.669–0.699) are in trainval, **none in the
official test split** — so the reported test-set ceiling is 100%, and the plan's instruction
to count those five as misses applies to train/val only. Two of them land in the seed-0 val
fold. The 0.70 threshold is still **pending team confirmation**; note that `pos_thresh=0.66`
would make all 10,013 defects reachable, since four of the five sit within 0.011 of the line.

### Overlapping tiles

Tiles overlap by 40 px, so more than one tile can hold ≥70% of the same small defect.

* `match="any"` (default) — the defect is found if *any* qualifying tile fires. This is what
  a detector is actually asked to do.
* `match="best"` — only the single best-containment tile may be credited; the stricter
  reading of the plan's singular wording.

Both are available and should be reported if they differ.

### Precision, and the honest denominator

Recall counts **defects**; precision counts **(tile, class) firings**, because a false
positive is not attached to any defect. The two have different denominators, and every report
this module prints says so rather than hiding it.

* A firing on a class the tile marks **ignore** is neither a true nor a false positive — it is
  excluded entirely (the tile holds a fragment too small to name, so neither answer is wrong).
* A firing on a non-ignored negative class is a false positive.

### The gate cannot cheat

A tile the gate skips produces **no** predictions. Any defect it held is lost and stays lost:
`gate_fired` masks predictions off and recall drops accordingly, with `n_lost_to_gate`
recording how many defects were lost that way. Verified on the real test split: a gate that
skips everything yields 0% recall with all 3,140 defects charged as lost.

### Thresholds

`choose_thresholds_for_false_alarm_rate` targets a **board-level** false-alarm rate — the
share of defect-free boards raising at least one defect, which is what an operator
experiences. A per-class *tile* rate is not interpretable as one number: six classes at 2% of
tiles each is 12% of tiles overall.

It sweeps one shared per-class tile rate, places every class at its own quantile of that rate
(so a class with a wider score spread gets a correspondingly higher threshold), and keeps the
loosest setting that still meets the target. Board rate is monotone in that sweep, so a
bisection is exact. Thresholds are selected on **validation** only and reused unchanged on
test.

Scores are **logits** by default — the student has no sigmoid — so thresholds live in logit
space too. A probability `p` corresponds to `log(p / (1 - p))`.

### Validation

`tests/test_metrics.py` covers the cases the plan lists — hits, misses, ignored classes, gate
skips, clean-board false alarms — plus per-class thresholds, the two match modes, and the real
split ceilings. End-to-end on the official 500-board test split:

| predictor | recall | precision | false positives |
| --- | --- | --- | --- |
| oracle (fires on every labelled-positive tile/class) | 100.00% | 100.00% | 0 |
| silent | 0.00% | — | 0 |
| oracle with a gate that skips every tile | 0.00% | — | 0 |

The oracle's recall equalling the geometric ceiling exactly is the check that the metric and
the labelling rule agree.

## `gate_analysis.py`

See the project README for the measured table. Two corrections to apply to that script before
its numbers go in the paper:

* **It reports a requested recall it never verifies.** `thr@98` is the 2% quantile of the
  defect-tile score distribution, but when that quantile is 0 — which it is for a 5×5 erosion
  — the gate cannot actually achieve 98% recall, because tiles scoring 0 can never fire. The
  table needs an *achieved* `recall@thr` column.
* Consequently the published `erode2` rows are internally inconsistent: `skip@98 = 98.4%` and
  `recall@skip50 = 86.9%` describe the **same** operating point (threshold 0). Directly
  measured: **15.4% of defect tiles score exactly 0 under a 5×5 erosion**, versus 0.0% under
  3×3. `Findings.md`'s claim that 5×5 reaches 98% recall is wrong, and 3×3 is the right
  classical gate.
