#!/usr/bin/env python3
"""Evaluate the production pipeline on Google's Monk Skin Tone Examples (MST-E).

MST-E (https://skintone.google/mste-dataset) is 1,515 consented photographs of
19 people, each with one expert MST label assigned by Dr. Ellis Monk, shot
across poses and lighting conditions. It is the closest thing to ground truth
that exists for this scale.

Terms that shape this script
----------------------------
* Research, model evaluation and annotator training only. Never training. This
  script only evaluates; nothing here learns from the images.
* Downloading needs a signed-in Google account, so the images are fetched by a
  person, not by this script. See eval/README.md.
* The images never enter the repository. Point --root at wherever they live.
  The report holds aggregates only: no file names, no subject names (subjects
  become S01..S19, ordered by label).

Honest N
--------
1,515 images sounds like a lot, but they come from 19 people, and photos of
the same person are not independent. The report therefore also gives
per-person results and confidence intervals from a bootstrap that resamples
*people*, not images.

The pipeline is exactly what /analyze runs: decode and white balance, face
landmarks, the quality gate, skin patch sampling, the Monk classifier. A photo
the app would refuse counts as refused here too.

Usage
-----
    python eval/mste_eval.py --root ~/datasets/mste --out mste-report.json
    python eval/mste_eval.py --root ~/datasets/mste --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

LEVELS = list(range(1, 11))
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
BOOTSTRAP_SEED = 20261104


# ── Finding the data ──────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def find_metadata(root: Path) -> Path:
    """The per-image CSV: the one with an image column and an MST column."""
    for path in sorted(root.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as fh:
            header = {_norm(h) for h in next(csv.reader(fh), [])}
        if "mst" in header and ({"image_name", "image"} & header):
            return path
    raise SystemExit(
        f"No metadata CSV under {root} with image_name and MST columns.\n"
        "Download the MST-E metadata file alongside the images (eval/README.md)."
    )


def index_images(root: Path) -> dict[str, list[Path]]:
    """Every image under root, keyed by lower-case file name and by stem."""
    index: dict[str, list[Path]] = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            if p.suffix.lower() in IMAGE_EXT:
                index[f.lower()].append(p)
                index[p.stem.lower()].append(p)
    return index


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "masked", "mask"}


def load_rows(meta: Path, index: dict[str, list[Path]], include_masked: bool) -> tuple[list[dict], Counter]:
    rows, skipped = [], Counter()
    with meta.open(newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            r = {_norm(k): (v or "").strip() for k, v in raw.items() if k}
            name = r.get("image_name") or r.get("image") or ""
            subject = r.get("subject_name") or r.get("subject") or ""
            try:
                mst = int(float(r.get("mst", "")))
            except ValueError:
                skipped["no label"] += 1
                continue
            if not 1 <= mst <= 10:
                skipped["no label"] += 1
                continue
            if _truthy(r.get("mask")) and not include_masked:
                skipped["face mask"] += 1
                continue

            key = Path(name).name.lower()
            candidates = index.get(key) or index.get(Path(key).stem) or []
            if len(candidates) > 1 and subject:
                in_subject = [c for c in candidates if subject.lower() in str(c).lower()]
                candidates = in_subject or candidates
            if not candidates:
                skipped["image not found"] += 1
                continue

            rows.append({
                "path": candidates[0],
                "subject": subject or candidates[0].parent.name,
                "mst": mst,
                "lighting": r.get("lighting") or "unknown",
                "pose": r.get("pose") or "unknown",
            })
    return rows, skipped


# ── The pipeline ──────────────────────────────────────────────────────────────

def analyse(image_bytes: bytes) -> dict:
    """What /analyze does, without HTTP. Never raises."""
    from detection.face_detection import detect_face, extract_skin_sample_from_image
    from detection.monk_classifier import classify_monk
    from quality_gate import run_quality_gate

    try:
        img_bgr, landmarks = detect_face(image_bytes)
    except ValueError:
        return {"status": "refused", "reason": "unreadable"}
    if landmarks is None:
        return {"status": "refused", "reason": "no face found"}
    gate = run_quality_gate(img_bgr, landmarks)
    if not gate["passed"]:
        return {"status": "refused", "reason": "quality gate"}
    try:
        sample = extract_skin_sample_from_image(img_bgr, landmarks)
    except ValueError:
        return {"status": "refused", "reason": "too few skin patches"}
    result = classify_monk(sample, image_bytes)
    return {"status": "ok", "predicted": int(result["monk_scale"].split("-")[1])}


# ── Scoring ───────────────────────────────────────────────────────────────────

def _rate(num: float, den: float):
    return round(num / den, 4) if den else None


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    exact = sum(r["predicted"] == r["mst"] for r in ok)
    within = sum(abs(r["predicted"] - r["mst"]) <= 1 for r in ok)
    confusion = [[0] * 10 for _ in LEVELS]
    for r in ok:
        confusion[r["mst"] - 1][r["predicted"] - 1] += 1
    return {
        "n": n,
        "measured": len(ok),
        "rejected": n - len(ok),
        "exact": _rate(exact, len(ok)),
        "within1": _rate(within, len(ok)),
        "exact_of_all": _rate(exact, n),
        "within1_of_all": _rate(within, n),
        "rejection_rate": _rate(n - len(ok), n),
        "mean_bias": _rate(sum(r["predicted"] - r["mst"] for r in ok), len(ok)),
        "confusion": confusion,
        "refusal_reasons": dict(Counter(r["reason"] for r in rows if r["status"] != "ok")),
    }


def bootstrap(rows: list[dict], key: str, draws: int = 2000) -> list[float] | None:
    """95% interval for a rate, resampling people rather than photos."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    subjects = list(by_subject)
    if len(subjects) < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stats = []
    for _ in range(draws):
        pick = rng.integers(len(subjects), size=len(subjects))
        sample = [r for i in pick for r in by_subject[subjects[i]]]
        value = summarise(sample)[key]
        if value is not None:
            stats.append(value)
    if not stats:
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def per_person(rows: list[dict]) -> list[dict]:
    """One row per subject, anonymised and ordered by label."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    ordered = sorted(by_subject.items(), key=lambda kv: (kv[1][0]["mst"], kv[0]))
    out = []
    for i, (_, items) in enumerate(ordered, start=1):
        s = summarise(items)
        preds = [r["predicted"] for r in items if r["status"] == "ok"]
        out.append({
            "id": f"S{i:02d}",
            "mst": items[0]["mst"],
            "photos": s["n"],
            "exact": s["exact"],
            "within1": s["within1"],
            "rejection_rate": s["rejection_rate"],
            "typical_prediction": Counter(preds).most_common(1)[0][0] if preds else None,
        })
    return out


def build_report(rows: list[dict], skipped: Counter) -> dict:
    lightings = sorted({r["lighting"] for r in rows})
    poses = sorted({r["pose"] for r in rows})
    people = per_person(rows)
    overall = summarise(rows)
    return {
        "meta": {
            "kind": "photos",
            "dataset": "Monk Skin Tone Examples (MST-E), Google, images courtesy of TONL",
            "terms": "Research and model evaluation only. Not used for training.",
            "photos": len(rows),
            "people": len(people),
            "skipped": dict(skipped),
            "generated": time.strftime("%Y-%m-%d"),
        },
        "overall": overall,
        "intervals": {
            "exact_of_all": bootstrap(rows, "exact_of_all"),
            "within1_of_all": bootstrap(rows, "within1_of_all"),
            "within1": bootstrap(rows, "within1"),
        },
        "people_within1": _rate(
            sum(1 for p in people if p["typical_prediction"] is not None and abs(p["typical_prediction"] - p["mst"]) <= 1),
            len(people),
        ),
        "levels_covered": sorted({r["mst"] for r in rows}),
        "by_lighting": {
            l: {**summarise([r for r in rows if r["lighting"] == l]),
                "label": l.replace("_", " ").capitalize()}
            for l in lightings
        },
        "by_pose": {p: summarise([r for r in rows if r["pose"] == p]) for p in poses},
        "by_level": {
            str(lvl): summarise([r for r in rows if r["mst"] == lvl])
            for lvl in LEVELS if any(r["mst"] == lvl for r in rows)
        },
        "people": people,
    }


def _print(report: dict) -> None:
    m, o, ci = report["meta"], report["overall"], report["intervals"]
    fmt = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
    band = lambda c: "" if not c else f"  (95% CI {c[0]:.0%} to {c[1]:.0%}, by person)"  # noqa: E731
    print(f"\n{m['photos']} photos of {m['people']} people. Skipped: {m['skipped'] or 'none'}")
    print(f"Refused {fmt(o['rejection_rate'])}: {o['refusal_reasons'] or 'none'}")
    print(f"Exact, of all photos      {fmt(o['exact_of_all'])}{band(ci['exact_of_all'])}")
    print(f"Within 1, of all photos   {fmt(o['within1_of_all'])}{band(ci['within1_of_all'])}")
    print(f"Within 1, of answered     {fmt(o['within1'])}{band(ci['within1'])}")
    print(f"People whose typical answer is within 1: {fmt(report['people_within1'])}")
    print(f"Levels covered: {report['levels_covered']}")
    print("\nBy lighting")
    for k, s in report["by_lighting"].items():
        print(f"  {k:<16} n={s['n']:<5} within1 {fmt(s['within1_of_all'])}  refused {fmt(s['rejection_rate'])}  bias {s['mean_bias']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="folder holding the MST-E images and metadata CSV")
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    ap.add_argument("--limit", type=int, help="only the first N photos (smoke test)")
    ap.add_argument("--include-masked", action="store_true", help="keep photos where the subject wears a face mask")
    args = ap.parse_args()

    root = args.root.expanduser()
    meta = find_metadata(root)
    rows, skipped = load_rows(meta, index_images(root), args.include_masked)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"No usable photos. Skipped: {dict(skipped)}")
        return 1

    print(f"Metadata: {meta.name}. Running {len(rows)} photos through the pipeline...")
    start = time.time()
    for i, r in enumerate(rows, start=1):
        r.update(analyse(r["path"].read_bytes()))
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}  ({time.time() - start:.0f}s)")

    report = build_report(rows, skipped)
    _print(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
