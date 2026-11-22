"""Pad-grid pitch, excluding the hatched-dot texture, plus a DeepPCB ruler crop.

The first autocorrelation attempt locked onto the ~12 px ground-plane dot
pattern. Through-hole pads sit on a 2.54 mm grid, so search 30..200 px instead
and report the three strongest peaks per board for a sanity check by eye.

Usage:
    python -I scratchpad/pitch.py --pku <clone> --deeppcb <PCBData> --out-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    return int(np.nan_to_num(sigma_b).argmax())


def peaks(prof: np.ndarray, lo: int, hi: int, k: int = 3) -> list[tuple[int, float]]:
    seg = prof[lo:hi]
    idx = []
    for i in range(1, len(seg) - 1):
        if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
            idx.append((i + lo, float(seg[i])))
    idx.sort(key=lambda t: -t[1])
    return idx[:k]


def autocorr_profiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = mask.astype(np.float32)
    m -= m.mean()
    f = np.fft.rfft2(m)
    ac = np.fft.irfft2(f * np.conj(f), s=m.shape)
    ac /= ac.flat[0]
    return ac[0, :len(ac[0]) // 2], ac[:len(ac) // 2, 0]


def ruler(img: np.ndarray, every: int = 50) -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    out = img.copy()
    out[::every, :] = np.array([255, 0, 0], dtype=np.uint8)
    out[:, ::every] = np.array([255, 0, 0], dtype=np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pku", required=True)
    ap.add_argument("--deeppcb", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PKU pad pitch, peaks in the 30..200 px band (x / y):")
    for tp in sorted(Path(args.pku, "PCB_USED").glob("*.JPG")):
        with Image.open(tp) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        g = arr[..., 0]                      # R separates pads from board best
        mask = (g > otsu(g)).astype(np.float32)
        px, py = autocorr_profiles(mask)
        tx, ty = peaks(px, 30, 200), peaks(py, 30, 200)
        print(f"  {tp.stem}: x " + ", ".join(f"{d}px({v:.3f})" for d, v in tx)
              + "   y " + ", ".join(f"{d}px({v:.3f})" for d, v in ty))

    # DeepPCB: a template crop with a 50 px ruler, to read its pad pitch by eye
    pcb = Path(args.deeppcb)
    temps = sorted(pcb.glob("group*/*/*_temp.jpg"))
    for tp in temps[:2]:
        with Image.open(tp) as im:
            arr = np.asarray(im.convert("L"), dtype=np.uint8)
        Image.fromarray(ruler(arr)).save(out_dir / f"deeppcb_{tp.stem}_ruler.png")
        mask = (arr > 127).astype(np.float32)
        px, py = autocorr_profiles(mask)
        print(f"DeepPCB {tp.stem}: x "
              + ", ".join(f"{d}px({v:.3f})" for d, v in peaks(px, 20, 300))
              + "   y " + ", ".join(f"{d}px({v:.3f})" for d, v in peaks(py, 20, 300)))
        print(f"  wrote {out_dir / f'deeppcb_{tp.stem}_ruler.png'}")


if __name__ == "__main__":
    main()
