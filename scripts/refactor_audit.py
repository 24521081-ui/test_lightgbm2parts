"""Snapshot/check existing artifacts without training or rewriting results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "refactor_audit"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def availability(files):
    """Describe saved evidence, without inventing missing component values."""
    component_files = [name for name, record in files.items()
                       if {"p_raw", "p_platt", "p_C7", "mu"}.issubset(record.get("columns", []))]
    return {
        "forecast_metric_config_model_hashes": "Xem danh mục files, columns, metadata và summaries.",
        "C5_C6_daily_components": "Đã có probability_raw/probability_positive/conditional_units trong CSV chi tiết.",
        "C7_daily_components": {
            "status": "đã có dữ liệu" if component_files else "chưa có dữ liệu",
            "files": component_files,
            "note": "Không thay bằng xác suất trung bình hoặc thành phần trên đường dự báo C6.",
        },
    }


def snapshot():
    files = {}
    for path in sorted((ROOT / "results").rglob("*")):
        if not path.is_file():
            continue
        record = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                record["columns"] = next(csv.reader(stream), [])
        if path.suffix == ".json":
            record["metadata"] = json.loads(path.read_text(encoding="utf-8-sig"))
        files[path.relative_to(ROOT).as_posix()] = record
    summaries = {}
    for path in sorted((ROOT / "results" / "analysis").rglob("01_by_origin.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            summaries[path.relative_to(ROOT).as_posix()] = list(csv.DictReader(stream))
    return {"files": files, "summaries": summaries, "availability": availability(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["before", "check"])
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    baseline = AUDIT / "baseline_before.json"
    if args.action == "before" and baseline.exists():
        raise FileExistsError("Baseline already exists; refusing to overwrite it.")
    current = snapshot()
    if args.action == "before":
        current["git_status_before"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, encoding="utf-8"
        )
        current["scope"] = "Existing results only; no pipeline execution. Missing components are not invented."
        baseline.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Baseline: {len(current['files'])} artifacts")
    else:
        old = json.loads(baseline.read_text(encoding="utf-8"))
        changed = [name for name, info in old["files"].items() if current["files"].get(name) != info]
        result = {"status": "FAIL" if changed else "PASS", "checked": len(old["files"]),
                  "changed_or_missing": changed, "new": sorted(set(current["files"]) - set(old["files"])),
                  "scope": "Artifact integrity only; this is not a prediction replay."}
        (AUDIT / "artifact_integrity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        if changed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
