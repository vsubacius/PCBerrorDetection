# paper/

4–6 page workshop-style write-up. Nothing here yet.

Results that already exist and belong in it:

* The tile-grid justification: defect-size histogram per class, and reachability (the recall
  ceiling) as a function of tile/stride — `data/grid_sweep.py`.
* The Stage-1 gate characterisation: raw XOR counting separates clean from defective tiles at
  only AUC 0.78 because re-binarized JPEG leaves a 1-px fringe along every copper edge; eroding
  the mask first takes it to 0.95 — `eval/gate_analysis.py`.

Both are in the README with the exact numbers and the commands that reproduce them.
