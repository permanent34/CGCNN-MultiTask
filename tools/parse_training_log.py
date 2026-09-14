#!/usr/bin/env python
"""Parse CGCNN console logs and export real training metrics.

CGCNN prints:
  * MAE value   after validation
  ** MAE value  after the final test

Those lines are authoritative. Batch-level Test lines are only guaranteed to
represent the complete validation set when the final batch was printed.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"Epoch:\s*\[(?P<epoch>\d+)\]\[(?P<batch>\d+)/(?P<batches>\d+)\].*?"
    r"Loss\s+(?P<loss>[-+0-9.eE]+)\s+\((?P<loss_avg>[-+0-9.eE]+)\).*?"
    r"MAE\s+(?P<mae>[-+0-9.eE]+)\s+\((?P<mae_avg>[-+0-9.eE]+)\)"
)
TEST_RE = re.compile(
    r"Test:\s*\[(?P<batch>\d+)/(?P<batches>\d+)\].*?"
    r"Loss\s+(?P<loss>[-+0-9.eE]+)\s+\((?P<loss_avg>[-+0-9.eE]+)\).*?"
    r"MAE\s+(?P<mae>[-+0-9.eE]+)\s+\((?P<mae_avg>[-+0-9.eE]+)\)"
)
VAL_MAE_RE = re.compile(r"^\s*\*\s+MAE\s+(?P<mae>[-+0-9.eE]+)\s*$")
TEST_MAE_RE = re.compile(r"^\s*\*\*\s+MAE\s+(?P<mae>[-+0-9.eE]+)\s*$")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path):
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    train_by_epoch = {}
    validation = []
    final_test = None
    in_final_test = False

    for line in lines:
        if "Evaluate Model on Test Set" in line:
            in_final_test = True
            continue

        match = EPOCH_RE.search(line)
        if match:
            epoch = int(match.group("epoch"))
            batch = int(match.group("batch"))
            batches = int(match.group("batches"))
            train_by_epoch[epoch] = {
                "epoch": epoch,
                "train_loss": float(match.group("loss_avg")),
                "train_mae": float(match.group("mae_avg")),
                "train_loss_complete": batch + 1 == batches,
            }
            continue

        match = TEST_RE.search(line)
        if match:
            batch = int(match.group("batch"))
            batches = int(match.group("batches"))
            item = {
                "loss": float(match.group("loss_avg")),
                "loss_complete": batch + 1 == batches,
                "mae": float(match.group("mae_avg")),
                "mae_exact": False,
            }
            if in_final_test:
                final_test = item
            else:
                validation.append(item)
            continue

        match = VAL_MAE_RE.match(line)
        if match and not in_final_test and validation:
            validation[-1]["mae"] = float(match.group("mae"))
            validation[-1]["mae_exact"] = True
            continue

        match = TEST_MAE_RE.match(line)
        if match and final_test is not None:
            final_test["mae"] = float(match.group("mae"))
            final_test["mae_exact"] = True

    train = [train_by_epoch[key] for key in sorted(train_by_epoch)]
    if not train:
        raise SystemExit("No Epoch records found in log.")
    if len(validation) != len(train):
        raise SystemExit(
            f"Expected one validation record per epoch, got "
            f"{len(validation)} validation and {len(train)} training records."
        )
    if final_test is None:
        raise SystemExit("No final test record found after the test marker.")

    rows = []
    for train_item, val_item in zip(train, validation):
        rows.append({
            "epoch": train_item["epoch"],
            "train_loss": train_item["train_loss"],
            "train_mae": train_item["train_mae"],
            "val_loss": val_item["loss"],
            "val_mae": val_item["mae"],
            "train_loss_complete": train_item["train_loss_complete"],
            "val_loss_complete": val_item["loss_complete"],
        })

    best_validation = min(rows, key=lambda row: row["val_mae"])
    report = {
        "source_log": str(path),
        "source_sha256": sha256_file(path),
        "epochs": len(rows),
        "best_validation": best_validation,
        "final_test": {
            "loss": final_test["loss"],
            "loss_complete": final_test["loss_complete"],
            "mae": final_test["mae"],
            "mae_exact": final_test["mae_exact"],
        },
        "notes": {
            "train_loss": "Reported running average from the last printed batch.",
            "val_loss": "Only complete when val_loss_complete is true.",
            "val_mae": "Exact when the * MAE marker was found.",
        },
    }
    return rows, report


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch", "train_loss", "train_mae", "val_loss", "val_mae",
        "train_loss_complete", "val_loss_complete",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(prefix, rows):
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in rows], label="Train loss (reported average)")
    if all(row["val_loss_complete"] for row in rows):
        ax.plot(epochs, [row["val_loss"] for row in rows], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Real CGCNN training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_loss.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_mae"] for row in rows], label="Train MAE (reported average)")
    ax.plot(epochs, [row["val_mae"] for row in rows], label="Validation MAE (reported)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Real CGCNN training MAE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_mae.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--figure-prefix", type=Path)
    args = parser.parse_args()

    rows, report = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{args.prefix}_metrics.csv", rows)
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.figure_prefix:
        args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
        make_figures(args.figure_prefix, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
