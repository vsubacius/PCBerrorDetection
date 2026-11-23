"""Serial monitor for the PCB firmware on the XIAO ESP32-S3.

* Reconnects by itself when the board resets (its USB port drops for ~1 s).
* Type a command and press Enter to send it to the board (e.g. 3, then c).
* Images the board sends (@@IMG blocks) are saved as PNG files in captures/.
  Type v for a live view window (needs: pip install opencv-python), with a
  check that all four corner squares are visible and a focus score.
  In the live view: type c + Enter to capture and test without leaving the
  view, a board number + Enter to pick the board, Enter alone to stop.
* Test results go to results/*.csv (one row per test; see save_result).
  After a live test it also saves boardN_compare_*.png:
  live | stored | difference (red = live black where stored is white,
  blue = live white where stored is black).
* Everything else is shown and saved to the log file.

Usage:
    python monitor.py              # finds the Espressif port by itself
    python monitor.py COM5         # or give the port
    python monitor.py COM5 my.txt  # and a log file name (default pcb_log.txt)

Needs: pip install pyserial pillow numpy   (pillow/numpy only for PNG output)
Stop with Ctrl+C.
"""
import base64
import csv
import json
import re
import sys
import zlib
import threading
import time
from pathlib import Path

import serial
from serial.tools import list_ports

try:
    import numpy as np
    from PIL import Image
except ImportError:          # images are then saved as PGM files
    np = None
    Image = None
try:
    import cv2               # live view window (pip install opencv-python)
except ImportError:
    cv2 = None

ESPRESSIF_VID = 0x303A
# The camera driver prints warnings straight to the USB port from its own task,
# so they can appear in the middle of an image line. Remove them from the byte
# stream (and show them) before reading lines.
BASE64_LINE = re.compile(r"[A-Za-z0-9+/]*={0,2}")
DRIVER_MSG = re.compile(rb"(?:\x1b\[[0-9;]*m)?(?:[IWEDV] \(\d+\) )?cam_hal: [^\n]*\n")
CAPTURES = Path("captures")
RESULTS = Path("results")
last_saved = {"compare": ""}
current = {"ser": None}
last_boards = {}


def find_port():
    for p in list_ports.comports():
        if p.vid == ESPRESSIF_VID:
            return p.device
    return None


def keyboard():
    """Send each typed line to the board."""
    while True:
        try:
            line = input()
        except EOFError:
            return
        ser = current["ser"]
        if ser is None:
            print("[monitor] not connected yet")
            continue
        try:
            ser.write((line.strip() + "\n").encode())
        except serial.SerialException:
            print("[monitor] could not send (board resetting?)")


def check_image(header, end_line, data):
    """Return an error message, or None if the byte count and CRC32 match."""
    expected = int(header.split()[5])
    if len(data) != expected:
        return f"{len(data)} bytes received, {expected} expected"
    parts = end_line.split()
    if len(parts) > 1 and int(parts[1], 16) != zlib.crc32(data):
        return "checksum does not match"
    return None


# ------------------------------------------------------------------ results
# The firmware prints "@@RESULT {json}" after each test. Each record is added
# to results/results.jsonl and as a row to results/<kind>.csv:
#   session        device, model size, input mode (once per boot)
#   live_capture   one camera test: alignment, accuracy vs stored, timing, memory
#   bench_board    stored-image benchmark, one row per board
#   bench_summary  stored-image benchmark totals: recall, precision, latency,
#                  gate speedup, memory, model size
#   energy_run     the 'e' test: tiles per second (add your meter readings)
def save_result(line, log):
    RESULTS.mkdir(exist_ok=True)
    try:
        rec = json.loads(line[len("@@RESULT "):])
    except json.JSONDecodeError:
        with open(RESULTS / "unreadable_results.txt", "a") as f:
            f.write(line + "\n")
        print("[monitor] a result line was damaged; saved to results/unreadable_results.txt")
        return
    rec = {"pc_time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    if rec.get("kind") == "live_capture" and last_saved["compare"]:
        rec["compare_image"] = last_saved["compare"]
        last_saved["compare"] = ""
    with open(RESULTS / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    kind = rec.get("kind", "other")
    path = RESULTS / f"{kind}.csv"
    if kind == "live_capture":
        rec.setdefault("compare_image", "")
    fields = list(rec.keys())
    if path.exists():
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if set(fields) <= set(header):
            fields = header          # same columns (missing ones stay empty)
        else:                        # firmware added columns: start a new file
            path = RESULTS / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        w.writerow(rec)
    note = f"[monitor] result saved: {path}"
    if kind == "live_capture":
        note += (f"  (board {rec.get('board_index')}: live found {rec.get('live_found')}/"
                 f"{rec.get('defects')}, stored {rec.get('stored_found')}/{rec.get('defects')}, "
                 f"pixel error {rec.get('pixel_error_pct', 0):.2f}%)")
    print(note)
    log.write(note + "\n")


# ---------------------------------------------------------------- live view
# Same corner-square check as the firmware (main/live.cpp), as a framing guide.
FID_RATIO = 56.0 / (640 + 2 * 60.0)      # square side / distance between square centres
view_state = {"window": False, "warned": False, "board": "?"}


def find_corner_squares(gray):
    """Return (list of 4 (x, y, w, h) boxes or None, message)."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.dilate(small, np.ones((29, 29), np.uint8))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    norm = np.clip(gray.astype(np.float32) * 230.0 / np.maximum(bg, 1), 0, 255).astype(np.uint8)
    _, dark = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        if area < 0.00015 * w * h or area > 0.03 * w * h:
            continue
        if area / (bw * bh) < 0.5 or not 0.5 <= bw / bh <= 2.0:
            continue
        cand.append((cents[i][0], cents[i][1], area, (x, y, bw, bh)))
    if len(cand) < 4:
        return None, f"only {len(cand)} square-ish dark blobs (need the 4 corner squares)"
    picks = []
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        picks.append(max(cand, key=lambda c: dx * c[0] + dy * c[1]))
    if len({id(p) for p in picks}) < 4:
        return None, "corner squares not found (sheet rotated a lot?)"
    pts = np.array([(p[0], p[1]) for p in picks])
    sides = [np.hypot(*(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
    areas = [p[2] for p in picks]
    ratio = np.sqrt((min(areas) + max(areas)) / 2) / np.mean(sides)
    if (max(sides) > 1.5 * min(sides) or max(areas) > 2.5 * min(areas)
            or not 0.5 * FID_RATIO <= ratio <= 2 * FID_RATIO):
        return [p[3] for p in picks], "4 blobs found but the shape is wrong (not the corner squares?)"
    board_px = np.mean(sides) * 640 / 760
    return [p[3] for p in picks], f"OK: all 4 corner squares found, board about {board_px:.0f} px wide"


def show_live(jpeg_bytes):
    CAPTURES.mkdir(exist_ok=True)
    (CAPTURES / "live_last.jpg").write_bytes(jpeg_bytes)
    if cv2 is None or np is None:
        if not view_state["warned"]:
            print("[monitor] for a live window run: pip install opencv-python numpy  "
                  "(meanwhile the newest frame is in captures/live_last.jpg)")
            view_state["warned"] = True
        return
    gray = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return
    boxes, message = find_corner_squares(gray)
    ok = message.startswith("OK")
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colour = (0, 200, 0) if ok else (0, 0, 255)
    if boxes:
        for x, y, bw, bh in boxes:
            cv2.rectangle(view, (int(x) - 3, int(y) - 3), (int(x + bw) + 3, int(y + bh) + 3), colour, 2)
    # Focus score: variance of the Laplacian in the middle of the frame
    # (higher = sharper; compare while turning the lens or moving the sheet).
    h, w = gray.shape
    mid = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    focus = cv2.Laplacian(mid, cv2.CV_64F).var()
    if w > 1000:   # fit large frames on the screen
        view = cv2.resize(view, (1000, int(h * 1000 / w)), interpolation=cv2.INTER_AREA)
    # Text goes in a bar above the picture, so it never hides a corner square.
    view = cv2.copyMakeBorder(view, 86, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    for k, text in enumerate((message,
                              f"focus {focus:.0f} (higher = sharper)    board {view_state['board']} selected",
                              "monitor: c+Enter = capture, number+Enter = board, Enter = stop")):
        y = 24 + 26 * k
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    colour if k == 0 else (255, 255, 255), 2)
    cv2.imshow("PCB live view", view)
    cv2.waitKey(1)
    view_state["window"] = True


def save_image(header, data):
    _, name, w, h, kind, _ = header.split()[:6]
    for extra in header.split()[6:]:
        if extra.startswith("board="):
            view_state["board"] = extra.split("=", 1)[1]
    w, h = int(w), int(h)
    if kind == "jpeg":
        if name == "live":
            show_live(bytes(data))
            return
        CAPTURES.mkdir(exist_ok=True)
        path = CAPTURES / f"{name}_{time.strftime('%H%M%S')}.jpg"
        path.write_bytes(bytes(data))
        print(f"[monitor] saved {path}")
        return
    CAPTURES.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    if kind == "bit":
        raw = bytes(data)
        pixels = bytearray()
        for byte in raw:
            for bit in range(7, -1, -1):
                pixels.append(255 if (byte >> bit) & 1 else 0)
    else:
        pixels = bytearray(data)
    if Image is None:
        path = CAPTURES / f"{name}_{stamp}.pgm"
        path.write_bytes(f"P5 {w} {h} 255\n".encode() + bytes(pixels))
        print(f"[monitor] saved {path}")
        return
    img = np.frombuffer(bytes(pixels), np.uint8).reshape(h, w)
    path = CAPTURES / f"{name}_{stamp}.png"
    Image.fromarray(img).save(path)
    print(f"[monitor] saved {path}")
    if name.endswith("_live") or name.endswith("_stored"):
        board, which = name.rsplit("_", 1)
        last_boards[(board, which)] = img
        live = last_boards.get((board, "live"))
        stored = last_boards.get((board, "stored"))
        if which == "stored" and live is not None and stored is not None:
            diff = np.stack([stored] * 3, -1).copy()
            diff[(live == 0) & (stored == 255)] = (255, 0, 0)
            diff[(live == 255) & (stored == 0)] = (0, 80, 255)
            gap = np.full((h, 8, 3), 128, np.uint8)
            comp = np.concatenate([np.stack([live] * 3, -1), gap,
                                   np.stack([stored] * 3, -1), gap, diff], 1)
            cpath = CAPTURES / f"{board}_compare_{stamp}.png"
            Image.fromarray(comp).save(cpath)
            print(f"[monitor] saved {cpath}  (live | stored | difference)")
            last_saved["compare"] = str(cpath)
            del last_boards[(board, "live")]
            del last_boards[(board, "stored")]


def main():
    port_arg = sys.argv[1] if len(sys.argv) > 1 else None
    log_name = sys.argv[2] if len(sys.argv) > 2 else "pcb_log.txt"
    threading.Thread(target=keyboard, daemon=True).start()
    with open(log_name, "a", encoding="utf-8", errors="replace") as log:
        waiting_said = False
        while True:
            port = port_arg or find_port()
            if port is None:
                if not waiting_said:
                    print("[monitor] waiting for the ESP32-S3 to appear...")
                    waiting_said = True
                time.sleep(0.2)
                continue
            try:
                with serial.Serial(port, 115200, timeout=0.2) as ser:
                    current["ser"] = ser
                    print(f"[monitor] connected to {port}, logging to {log_name}. "
                          f"Type a command and press Enter (h = help).")
                    waiting_said = False
                    pending = b""
                    img_header, img_data = None, []
                    while True:
                        chunk = ser.read(ser.in_waiting or 1)
                        if view_state["window"]:
                            cv2.waitKey(1)   # keep the live window responsive
                        if not chunk:
                            continue
                        pending += chunk
                        for msg in DRIVER_MSG.findall(pending):
                            text = msg.decode("utf-8", errors="replace").strip()
                            print(text)
                            log.write(text + "\n")
                        pending = DRIVER_MSG.sub(b"", pending)
                        while b"\n" in pending:
                            raw, pending = pending.split(b"\n", 1)
                            line = raw.decode("utf-8", errors="replace").rstrip("\r")
                            if img_header is not None:
                                if line.startswith("@@END"):
                                    raw_block = list(img_data)
                                    try:
                                        data = base64.b64decode("".join(img_data), validate=True)
                                        problem = check_image(img_header, line, data)
                                        if problem:
                                            CAPTURES.mkdir(exist_ok=True)
                                            bad = CAPTURES / f"damaged_{time.strftime('%H%M%S')}.txt"
                                            bad.write_text(img_header + "\n" + "\n".join(raw_block) + "\n" + line + "\n")
                                            print(f"[monitor] image damaged in transfer ({problem}); "
                                                  f"not saved -- send the command again. Raw data: {bad}")
                                        else:
                                            save_image(img_header, data)
                                    except Exception as e:  # keep monitoring
                                        print(f"[monitor] image not saved: {e}")
                                    img_header, img_data = None, []
                                elif BASE64_LINE.fullmatch(line.strip()):
                                    img_data.append(line.strip())
                                else:
                                    # A whole log line from another task, between image lines.
                                    print(line)
                                    log.write(line + "\n")
                                    log.flush()
                                continue
                            if line.startswith("@@RESULT "):
                                save_result(line, log)
                                continue
                            if line.startswith("@@IMG"):
                                img_header, img_data = line, []
                                if line.split()[1] != "live":
                                    print(f"[monitor] receiving image {line.split()[1]} ...")
                                continue
                            print(line)
                            log.write(line + "\n")
                            log.flush()
            except (serial.SerialException, OSError):
                current["ser"] = None
                print("\n[monitor] port lost (board reset?), reconnecting...")
                time.sleep(0.3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[monitor] stopped")
