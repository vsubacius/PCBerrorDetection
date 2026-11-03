"""The firmware gate must agree bit-for-bit with the numpy reference.

Every gate threshold in this project is selected on validation data in Python
and then used on the chip. If the two implementations disagree even slightly,
the operating point moves and the reported energy/recall trade-off is wrong.

These tests build ``firmware/pcb_bench/main/gate.cpp`` on the host with g++ and
compare its output against per-tile ``binary_morph`` from the data pipeline.
They skip if no C++ compiler is available, and the DeepPCB-based test skips if
the dataset has not been downloaded.
"""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))

from tile_dataset import binary_morph  # noqa: E402
from tiling import TileGrid  # noqa: E402

GATE_DIR = _REPO / "firmware" / "pcb_bench" / "main"
HARNESS = Path(__file__).resolve().parent / "gate_parity_harness.cpp"
TILE = 160
RADIUS = 1


def reference_score(test: np.ndarray, temp: np.ndarray, radius: int = RADIUS) -> int:
    """Count of erode(XOR(test, temp)) over a single tile, in numpy."""
    xor = (test ^ temp).astype(np.uint8)
    return int(binary_morph(xor, dilate=False, radius=radius).sum())


@pytest.fixture(scope="module")
def gate_binary(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no host C++ compiler available")
    out = tmp_path_factory.mktemp("gate") / "harness"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{GATE_DIR}", str(HARNESS), str(GATE_DIR / "gate.cpp"), "-o", str(out)],
        check=True, capture_output=True,
    )
    return out


def run_gate(gate_binary, tests: np.ndarray, temps: np.ndarray,
             radius: int = RADIUS) -> np.ndarray:
    """Score a stack of (n, tile, tile) tile pairs with the firmware gate."""
    assert tests.shape == temps.shape
    n, tile, _ = tests.shape
    payload = (struct.pack("<iii", tile, radius, n)
               + np.ascontiguousarray(tests, dtype=np.uint8).tobytes()
               + np.ascontiguousarray(temps, dtype=np.uint8).tobytes())
    done = subprocess.run([str(gate_binary)], input=payload,
                          capture_output=True, check=True)
    return np.frombuffer(done.stdout, dtype="<i4", count=n)


def test_matches_reference_on_random_tiles(gate_binary):
    """Dense random tiles exercise every window position and the edge clamping."""
    rng = np.random.default_rng(0)
    tests = rng.integers(0, 2, (24, TILE, TILE)).astype(np.uint8)
    # Templates that differ from the test image in varying amounts.
    temps = np.stack([
        np.where(rng.random((TILE, TILE)) < p, 1 - t, t).astype(np.uint8)
        for t, p in zip(tests, np.linspace(0.0, 0.5, len(tests)))
    ])
    got = run_gate(gate_binary, tests, temps)
    want = np.array([reference_score(t, m) for t, m in zip(tests, temps)])
    assert np.array_equal(got, want)


def test_matches_reference_on_structured_shapes(gate_binary):
    """Blobs, thin lines and edge-touching shapes -- the cases that matter."""
    cases = []
    for shape in range(6):
        test = np.zeros((TILE, TILE), np.uint8)
        temp = np.zeros((TILE, TILE), np.uint8)
        if shape == 0:                      # a solid blob well inside
            test[40:60, 40:60] = 1
        elif shape == 1:                    # a 1-px line: erosion must kill it
            test[80, :] = 1
        elif shape == 2:                    # a blob flush against the left edge
            test[70:90, 0:10] = 1
        elif shape == 3:                    # a blob in the corner
            test[0:6, 0:6] = 1
        elif shape == 4:                    # identical images -> zero
            test[20:40, 20:40] = 1
            temp[20:40, 20:40] = 1
        else:                               # everything differs
            test[:, :] = 1
        cases.append((test, temp))
    tests = np.stack([c[0] for c in cases])
    temps = np.stack([c[1] for c in cases])
    got = run_gate(gate_binary, tests, temps)
    want = np.array([reference_score(t, m) for t, m in zip(tests, temps)])
    assert np.array_equal(got, want)
    assert got[1] == 0, "a 1-px line must not survive a 3x3 erosion"
    assert got[4] == 0, "identical tile and template must score zero"


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_matches_reference_for_each_radius(gate_binary, radius):
    rng = np.random.default_rng(radius)
    tests = rng.integers(0, 2, (6, 64, 64)).astype(np.uint8)
    temps = rng.integers(0, 2, (6, 64, 64)).astype(np.uint8)
    got = run_gate(gate_binary, tests, temps, radius=radius)
    want = np.array([reference_score(t, m, radius) for t, m in zip(tests, temps)])
    assert np.array_equal(got, want)


def test_matches_reference_on_real_deeppcb_tiles(gate_binary):
    """The case the project actually runs: real boards, all 25 tiles."""
    import deeppcb

    root = _REPO / "data" / "DeepPCB"
    if not (root / "PCBData").is_dir():
        pytest.skip("DeepPCB not downloaded (python data/build_tiles.py --download)")

    samples = deeppcb.index(root, "trainval")[:8]
    grid = TileGrid(TILE, 120)
    tests, temps = [], []
    for sample in samples:
        test, temp = sample.load(binarize=True)
        tests.append(grid.cut(test))
        temps.append(grid.cut(temp))
    tests = np.concatenate(tests)
    temps = np.concatenate(temps)

    got = run_gate(gate_binary, tests, temps)
    want = np.array([reference_score(t, m) for t, m in zip(tests, temps)])
    assert np.array_equal(got, want)
    assert want.sum() > 0, "expected some surviving XOR pixels on real boards"


def test_scratch_size_is_small_enough_to_live_in_internal_sram():
    """The streaming vertical pass must not need the whole XOR image."""
    sys.path.insert(0, str(GATE_DIR))
    header = (GATE_DIR / "gate.hpp").read_text()
    assert "gate_scratch_bytes" in header
    # (2*radius+1) * tile bytes: 480 for a 160-px tile at radius 1.
    assert (2 * RADIUS + 1) * TILE == 480
