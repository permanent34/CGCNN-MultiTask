#!/usr/bin/env python
"""Summarize CGCNN test_results.csv files without changing the source data."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [[cell.strip() for cell in row] for row in csv.reader(handle) if row]


def metric_dict(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if actual.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}

    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return {"n": int(actual.size), "mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task-names", nargs="*", default=[])
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    rows = parse_rows(args.input)
    if not rows:
        raise SystemExit("No rows found in input CSV.")

    ncol = len(rows[0])
    if ncol == 3:
        num_tasks = 1
    elif ncol >= 5 and ncol % 2 == 1:
        num_tasks = (ncol - 1) // 2
    else:
        raise SystemExit(
            "Expected 3 columns (id,target,pred) or 1+2K columns; "
            f"got {ncol}."
        )

    names = list(args.task_names)
    if len(names) < num_tasks:
        names.extend(f"task_{i + 1}" for i in range(len(names), num_tasks))
    names = names[:num_tasks]

    tasks = {}
    for index, name in enumerate(names):
        actual = [row[1 + index] for row in rows]
        predicted = [row[1 + num_tasks + index] for row in rows]
        tasks[name] = metric_dict(actual, predicted)

    valid_maes = [item["mae"] for item in tasks.values() if item["mae"] is not None]
    report = {
        "dataset": args.dataset,
        "source_csv": str(args.input),
        "source_sha256": sha256_file(args.input),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "num_tasks": num_tasks,
        "mean_task_mae": float(np.mean(valid_maes)) if valid_maes else None,
        "tasks": tasks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        f"# {args.dataset} test metrics",
        "",
        f"- Source CSV: `{args.input.as_posix()}`",
        f"- SHA-256: `{report['source_sha256']}`",
        f"- Rows: {report['rows']}",
        f"- Mean task MAE: {report['mean_task_mae']:.6f}",
        "",
        "| Task | Valid values | MAE | RMSE | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in tasks.items():
        r2 = "N/A" if item["r2"] is None else f"{item['r2']:.6f}"
        lines.append(
            f"| {name} | {item['n']} | {item['mae']:.6f} | "
            f"{item['rmse']:.6f} | {r2} |"
        )
    (args.output_dir / "metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
