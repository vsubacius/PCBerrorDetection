"""The firmware's binarization must agree bit-for-bit with data/binarize.py.

Same contract as tests/test_gate_parity.py, and the same reason: the model is
trained on binarized dataset photos and runs on binarized camera frames, so a
one-level difference in the threshold or a single different pixel on an edge
means the chip is classifying a slightly different image than anything the
network ever saw.

Builds firmware/pcb_bench/main/pcb_binarize.cpp on the host with g++ and
compares. Skips when no host C++ compiler is available, and the PKU-based test
skips when the dataset has not been downloaded.
"""

import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))

from binarize import binarize, binarize_board, to_channel  # noqa: E402

FW_DIR = _REPO / "firmware" / "pcb_bench" / "main"
HARNESS = Path(__file__).resolve().parent / "binarize_parity_harness.cpp"
PKU_ROOT = _REPO / "data" / "PKU-DATASET"


@pytest.fixture(scope="module")
def binarize_binary(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no host C++ compiler available")
    out = tmp_path_factory.mktemp("binarize") / "harness"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{FW_DIR}", str(HARNESS), str(FW_DIR / "pcb_binarize.cpp"), "-o", str(out)],
        check=True, capture_output=True,
    )
    return out


def run_firmware(binarize_binary, frames: np.ndarray, flat: bool = False,
                 invert: bool = False, threshold: int = -1):
    """Binarize a stack of (n, h, w) single-channel frames with the C++ code."""
    n, h, w = frames.shape
    payload = (struct.pack("<iiiiiiii", 0, w, h, n, int(flat), int(invert),
                           threshold, -1)
               + np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
    done = subprocess.run([str(binarize_binary)], input=payload,
                          capture_output=True, check=True)
    thresholds, masks = [], []
    plane = h * w
    for i in range(n):
        at = i * (4 + plane)
        thresholds.append(struct.unpack_from("<i", done.stdout, at)[0])
        masks.append(np.frombuffer(done.stdout, np.uint8, plane, at + 4).reshape(h, w))
    return np.array(thresholds), np.stack(masks)


@pytest.mark.parametrize("invert", [False, True])
def test_matches_on_random_frames(binarize_binary, invert):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (6, 48, 64)).astype(np.uint8)
    got_t, got = run_firmware(binarize_binary, frames, invert=invert)
    for i, frame in enumerate(frames):
        want, want_t = binarize(frame, invert=invert)
        assert got_t[i] == want_t, f"threshold differs on frame {i}"
        assert np.array_equal(got[i], want), f"mask differs on frame {i}"


def test_matches_on_bimodal_frames(binarize_binary):
    """Realistic histograms: two modes, which is where Otsu's argmax matters."""
    rng = np.random.default_rng(1)
    frames = []
    for split in (0.2, 0.5, 0.8):
        f = rng.normal(70, 18, (40, 40))
        mask = rng.random((40, 40)) < split
        f[mask] = rng.normal(185, 22, int(mask.sum()))
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    frames = np.stack(frames)
    got_t, got = run_firmware(binarize_binary, frames)
    for i, frame in enumerate(frames):
        want, want_t = binarize(frame)
        assert got_t[i] == want_t
        assert np.array_equal(got[i], want)


def test_matches_with_flat_field(binarize_binary):
    """The box blur's edge clamping is the easy thing to get subtly wrong."""
    rng = np.random.default_rng(2)
    h, w = 96, 96
    ramp = np.linspace(30, 220, w)[None, :].repeat(h, axis=0)
    frames = np.stack([
        np.clip(ramp + rng.normal(0, 25, (h, w)), 0, 255).astype(np.uint8)
        for _ in range(3)
    ])
    got_t, got = run_firmware(binarize_binary, frames, flat=True)
    for i, frame in enumerate(frames):
        want, want_t = binarize(frame, use_flat_field=True)
        assert got_t[i] == want_t
        assert np.array_equal(got[i], want)


def test_matches_with_an_explicit_threshold(binarize_binary):
    """The live path thresholds a frame with the stored template's level."""
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, (4, 32, 32)).astype(np.uint8)
    got_t, got = run_firmware(binarize_binary, frames, threshold=137)
    assert (got_t == 137).all()
    for i, frame in enumerate(frames):
        want, _ = binarize(frame, threshold=137)
        assert np.array_equal(got[i], want)


def run_firmware_board(binarize_binary, frames: np.ndarray, flat: bool = False,
                       thresholds: tuple[int, int] = (-1, -1)):
    """Run the two-threshold board path on a stack of (n, h, w, 3) RGB frames."""
    n, h, w = frames.shape[:3]
    planes = np.stack([frames[..., 0], frames[..., 1]], axis=1)   # (n, 2, h, w)
    payload = (struct.pack("<iiiiiiii", 1, w, h, n, int(flat), 0,
                           int(thresholds[0]), int(thresholds[1]))
               + np.ascontiguousarray(planes, dtype=np.uint8).tobytes())
    done = subprocess.run([str(binarize_binary)], input=payload,
                          capture_output=True, check=True)
    used, masks = [], []
    plane = h * w
    for i in range(n):
        at = i * (8 + plane)
        used.append(struct.unpack_from("<ii", done.stdout, at))
        masks.append(np.frombuffer(done.stdout, np.uint8, plane, at + 8).reshape(h, w))
    return used, np.stack(masks)


def test_board_path_matches_on_random_rgb(binarize_binary):
    rng = np.random.default_rng(10)
    frames = rng.integers(0, 256, (5, 40, 56, 3)).astype(np.uint8)
    used, got = run_firmware_board(binarize_binary, frames)
    for i, frame in enumerate(frames):
        want, want_t = binarize_board(frame)
        assert tuple(used[i]) == want_t, f"thresholds differ on frame {i}"
        assert np.array_equal(got[i], want), f"mask differs on frame {i}"


def test_board_path_matches_with_given_thresholds(binarize_binary):
    """What the live path does: both levels come from the stored template."""
    rng = np.random.default_rng(11)
    frames = rng.integers(0, 256, (3, 32, 32, 3)).astype(np.uint8)
    used, got = run_firmware_board(binarize_binary, frames, thresholds=(100, 90))
    assert all(tuple(u) == (100, 90) for u in used)
    for i, frame in enumerate(frames):
        want, _ = binarize_board(frame, thresholds=(100, 90))
        assert np.array_equal(got[i], want)


def test_board_path_matches_on_real_pku_photographs(binarize_binary):
    """Real boards, including 08 -- the one a single green Otsu gets wrong."""
    if not (PKU_ROOT / "PCB_USED").is_dir():
        pytest.skip("PKU-Market-PCB not downloaded (python data/pku.py --download)")

    frames = []
    for board in ("01", "08", "11"):
        path = PKU_ROOT / "PCB_USED" / f"{board}.JPG"
        if not path.exists():
            continue
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]
        frames.append(arr[h // 2:h // 2 + 240, w // 2:w // 2 + 240])
    if not frames:
        pytest.skip("no PKU templates found")

    frames = np.stack(frames)
    used, got = run_firmware_board(binarize_binary, frames)
    for i, frame in enumerate(frames):
        want, want_t = binarize_board(frame)
        assert tuple(used[i]) == want_t, f"thresholds differ on board crop {i}"
        assert np.array_equal(got[i], want), f"mask differs on board crop {i}"


def test_matches_on_real_pku_crops(binarize_binary):
    """The case the project actually runs: green channel, real photographs."""
    if not (PKU_ROOT / "PCB_USED").is_dir():
        pytest.skip("PKU-Market-PCB not downloaded (see data/pku.py --download)")

    crops = []
    for template in sorted((PKU_ROOT / "PCB_USED").glob("*.JPG"))[:3]:
        with Image.open(template) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        g = to_channel(arr, "g")
        h, w = g.shape
        crops.append(g[h // 2:h // 2 + 160, w // 2:w // 2 + 160])

        board = re.match(r"(\d+)", template.stem).group(1)
        tests = sorted((PKU_ROOT / "images" / "Open_circuit").glob(f"{board}_*.jpg"))
        if tests:
            with Image.open(tests[0]) as im:
                t_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            tg = to_channel(t_arr, "g")
            crops.append(tg[h // 2:h // 2 + 160, w // 2:w // 2 + 160])

    frames = np.stack(crops)
    for invert in (False, True):
        got_t, got = run_firmware(binarize_binary, frames, invert=invert)
        for i, frame in enumerate(frames):
            want, want_t = binarize(frame, invert=invert)
            assert got_t[i] == want_t, f"threshold differs on crop {i}"
            assert np.array_equal(got[i], want), f"mask differs on crop {i}"
    assert got.sum() > 0, "expected some copper in real crops"
