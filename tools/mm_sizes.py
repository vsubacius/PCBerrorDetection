"""Defect box sizes in millimetres, using the per-board pad pitch as the scale."""
import re, sys, xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
PITCH = {"01":57,"04":57,"05":57,"06":57,"07":69,"08":69,"09":69,"10":103,"11":103,"12":103}
PXMM = {b: p/2.54 for b, p in PITCH.items()}
NAME = re.compile(r"^(\d+)_")
by_class = defaultdict(list); by_board = defaultdict(list)
for ann in root.joinpath("Annotations").rglob("*.xml"):
    board = NAME.match(ann.stem).group(1)
    for obj in ET.parse(ann).getroot().findall("object"):
        b = obj.find("bndbox")
        x1,y1,x2,y2 = (int(float(b.findtext(k))) for k in ("xmin","ymin","xmax","ymax"))
        name = obj.findtext("name").strip()
        short_px = min(x2-x1, y2-y1); long_px = max(x2-x1, y2-y1)
        by_class[name].append((short_px/PXMM[board], long_px/PXMM[board], short_px))
        by_board[board].append(short_px/PXMM[board])
print(f"px/mm per board: " + ", ".join(f"{b}:{v:.1f}" for b,v in sorted(PXMM.items())))
print(f"\n{'class':<18}{'n':>6}{'short mm':>10}{'long mm':>9}{'short p90 mm':>14}{'short px (native)':>19}")
for name in sorted(by_class):
    a = np.asarray(by_class[name])
    print(f"{name:<18}{len(a):>6}{np.median(a[:,0]):>10.2f}{np.median(a[:,1]):>9.2f}"
          f"{np.percentile(a[:,0],90):>14.2f}{np.median(a[:,2]):>19.0f}")
allmm = np.concatenate([np.asarray(v)[:,0] for v in by_class.values()])
print(f"{'ALL':<18}{len(allmm):>6}{np.median(allmm):>10.2f}")
print("\nDeepPCB for comparison: short side median 31 px at ~48 px/mm = 0.65 mm")
print("\nshort side (mm) median per board:")
for b in sorted(by_board):
    print(f"  {b}: {np.median(by_board[b]):.2f} mm  (n={len(by_board[b])})")
