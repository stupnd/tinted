"""MST-E harness: metadata matching, filtering, scoring. The pipeline is stubbed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import mste_eval as me  # noqa: E402


@pytest.fixture
def dataset(tmp_path):
    rows = ["image_name,pose,lighting,mask,subject_name,MST"]
    for s, mst in (("alice", 2), ("bo", 6), ("cy", 9)):
        (tmp_path / s).mkdir()
        for i, light in enumerate(["well_lit", "well_lit", "dim"]):
            name = f"{s}_{i}.jpg"
            (tmp_path / s / name).write_bytes(b"x")
            rows.append(f"{name},frontal,{light},no,{s},{mst}")
    rows.append("masked.jpg,frontal,well_lit,yes,alice,2")
    (tmp_path / "alice" / "masked.jpg").write_bytes(b"x")
    rows.append("ghost.jpg,frontal,well_lit,no,cy,9")
    # An annotations CSV that must not be mistaken for metadata.
    (tmp_path / "annotations.csv").write_text("subject,image,region,annotation\n")
    (tmp_path / "meta.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_finds_metadata_not_annotations(dataset):
    assert me.find_metadata(dataset).name == "meta.csv"


def test_loads_rows_and_counts_skips(dataset):
    rows, skipped = me.load_rows(me.find_metadata(dataset), me.index_images(dataset), include_masked=False)
    assert len(rows) == 9
    assert skipped == {"face mask": 1, "image not found": 1}
    assert {r["lighting"] for r in rows} == {"well_lit", "dim"}


def test_report_scores_and_anonymises(dataset):
    rows, skipped = me.load_rows(me.find_metadata(dataset), me.index_images(dataset), include_masked=False)
    for r in rows:
        if r["lighting"] == "dim" and r["subject"] == "cy":
            r.update({"status": "refused", "reason": "quality gate"})
        else:
            r.update({"status": "ok", "predicted": r["mst"] + (1 if r["subject"] == "bo" else 0)})
    report = me.build_report(rows, skipped)

    o = report["overall"]
    assert o["n"] == 9 and o["rejected"] == 1
    assert o["within1"] == 1.0
    assert o["exact"] == pytest.approx(5 / 8, abs=1e-4)
    assert o["refusal_reasons"] == {"quality gate": 1}
    assert report["by_lighting"]["dim"]["rejection_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert [p["id"] for p in report["people"]] == ["S01", "S02", "S03"]
    assert [p["mst"] for p in report["people"]] == [2, 6, 9]
    assert "alice" not in str(report) and ".jpg" not in str(report)
    lo, hi = report["intervals"]["exact_of_all"]
    assert 0 <= lo <= o["exact_of_all"] <= hi <= 1
