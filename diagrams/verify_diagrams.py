"""
verify_diagrams.py - REVIEW ALL GENERATED DIAGRAMS
===================================================
Opens every PNG/JPG under diagrams/, verifies it decodes, records format,
size, dimensions, and a blank-detection check (near-zero pixel variance
means the panel is empty/broken).

Run:   python diagrams/verify_diagrams.py
Out:   prints a per-folder summary and writes diagrams/gallery/REVIEW.txt
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "gallery")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["llm", "agents", "ml", "math", "dl", "backend", "nlp",
              "rag", "transformers", "graphs", "graph", "recommenders", "rl"]

EXTENSIONS = (".png", ".jpg", ".jpeg")


def analyze(path):
    """Return (ok, info-dict) for one image."""
    size_bytes = os.path.getsize(path)
    try:
        img = Image.open(path)
        img.load()
    except Exception as e:  # noqa: BLE001
        return False, {"path": path, "error": str(e), "bytes": size_bytes}
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    # Near-zero std => blank / solid-color panel.
    blank = std < 1.0
    return True, {
        "path": path,
        "fmt": (img.format or "?").upper(),
        "mode": img.mode,
        "w": img.width,
        "h": img.height,
        "bytes": size_bytes,
        "mean": round(mean, 1),
        "std": round(std, 1),
        "blank": blank,
    }


def main():
    results = []
    problems = []
    for cat in CATEGORIES:
        folder = os.path.join(ROOT, cat)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(EXTENSIONS):
                continue
            path = os.path.join(folder, fn)
            ok, info = analyze(path)
            info["cat"] = cat
            info["file"] = fn
            if ok:
                results.append(info)
                if info["blank"]:
                    problems.append(f"BLANK  {cat}/{fn}  (std={info['std']})")
            else:
                problems.append(f"UNREADABLE {cat}/{fn}: {info['error']}")

    total = len(results)
    total_bytes = sum(r["bytes"] for r in results)
    blank_count = sum(1 for r in results if r["blank"])

    lines = []
    lines.append("=" * 78)
    lines.append("DIAGRAM REVIEW - verify_diagrams.py")
    lines.append("=" * 78)
    lines.append(f"images verified : {total}")
    lines.append(f"total size      : {total_bytes/1e6:.1f} MB")
    lines.append(f"blank panels    : {blank_count}")
    lines.append("")
    lines.append(f"{'FOLDER':<14}{'COUNT':>6}{'TOTAL MB':>10}{'WxH RANGE':>26}")
    lines.append("-" * 58)
    for cat in CATEGORIES:
        cat_res = [r for r in results if r["cat"] == cat]
        if not cat_res:
            continue
        ws = [r["w"] for r in cat_res]
        hs = [r["h"] for r in cat_res]
        mb = sum(r["bytes"] for r in cat_res) / 1e6
        lines.append(f"{cat:<14}{len(cat_res):>6}{mb:>10.1f}"
                     f"{f'{min(ws)}x{min(hs)} - {max(ws)}x{max(hs)}':>26}")
    lines.append("-" * 58)
    lines.append("")
    lines.append("PER-IMAGE TABLE")
    lines.append(f"{'FILE':<56}{'FMT':<5}{'SIZE':>10}{'DIM':>12}{'STD':>7}  NOTE")
    lines.append("-" * 100)
    for r in sorted(results, key=lambda x: (x["cat"], x["file"])):
        dim = f"{r['w']}x{r['h']}"
        note = "BLANK?" if r["blank"] else ""
        lines.append(f"{r['cat']+'/'+r['file']:<56}{r['fmt']:<5}"
                     f"{r['bytes']/1e3:>8.1f}KB{dim:>12}{r['std']:>7.1f}  {note}")
    lines.append("")
    if problems:
        lines.append("PROBLEMS FOUND:")
        lines.extend("  " + p for p in problems)
        lines.append("")
        verdict = "PROBLEMS FOUND - inspect flagged panels"
    else:
        verdict = "ALL IMAGES OK - no unreadable or blank panels"
    lines.append("VERDICT: " + verdict)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, "REVIEW.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nreview saved to diagrams/gallery/REVIEW.txt")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())