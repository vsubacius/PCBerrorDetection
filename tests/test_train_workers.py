"""DataLoader workers must not be used when the dataset cannot reach them.

On a spawn platform each worker is a fresh process and the dataset is pickled
into it. The PKU training fold is 3.1 GB unpacked, and the transfer fails with
`OSError: [Errno 22]` in the parent and a truncated-pickle error in the child
-- neither of which names the cause. The whole 9-run sweep died that way in
223 seconds, every run reporting the same opaque error, so this guard exists
to turn a silent sweep-wide failure into a printed line and a working run.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "training"))
sys.path.insert(0, str(_REPO / "data"))

from train import (  # noqa: E402
    MAX_WORKER_PICKLE_BYTES,
    fold_bytes,
    safe_num_workers,
)


def fake_fold(n_images: int, size: int = 640):
    """Something with .test and .temp of a known size, without the memory."""
    array = types.SimpleNamespace(nbytes=n_images * size * size)
    return types.SimpleNamespace(test=array, temp=array)


def test_fold_bytes_counts_both_images():
    fold = fake_fold(10)
    assert fold_bytes(fold) == 2 * 10 * 640 * 640
    assert fold_bytes(fold, fold) == 4 * 10 * 640 * 640


def test_small_dataset_keeps_its_workers(monkeypatch):
    monkeypatch.setattr("train.multiprocessing.get_start_method", lambda allow_none: "spawn")
    # DeepPCB: 800 train + 200 val images, which has always worked with workers.
    assert safe_num_workers(3, fake_fold(800), fake_fold(200), verbose=False) == 3


def test_large_dataset_drops_to_zero_workers(monkeypatch, capsys):
    monkeypatch.setattr("train.multiprocessing.get_start_method", lambda allow_none: "spawn")
    # PKU: 3,827 train + 581 val crops = 3.6 GB.
    assert safe_num_workers(3, fake_fold(3827), fake_fold(581)) == 0
    assert "num_workers=0" in capsys.readouterr().out


def test_fork_platforms_are_left_alone(monkeypatch):
    """On fork, children inherit the arrays and nothing is pickled."""
    monkeypatch.setattr("train.multiprocessing.get_start_method", lambda allow_none: "fork")
    assert safe_num_workers(3, fake_fold(10_000), fake_fold(2_000), verbose=False) == 3


def test_zero_stays_zero(monkeypatch):
    monkeypatch.setattr("train.multiprocessing.get_start_method", lambda allow_none: "spawn")
    assert safe_num_workers(0, fake_fold(3827), verbose=False) == 0


def test_threshold_sits_between_the_two_datasets():
    """The limit must separate the dataset that works from the one that does not."""
    deeppcb = fold_bytes(fake_fold(800), fake_fold(200))
    pku = fold_bytes(fake_fold(3827), fake_fold(581))
    assert deeppcb < MAX_WORKER_PICKLE_BYTES < pku
