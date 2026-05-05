from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PTQ grid results")

    parser.add_argument(
        "--input",
        type=str,
        default="results/ptq_grid.json",
        help="Path to ptq_grid.json",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save plots",
    )

    return parser.parse_args()


def load_runs(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline = payload.get("baseline", {})
    runs = payload.get("runs", [])

    return baseline, runs


def group_by_weight_mode(runs):
    grouped = {}

    for row in runs:
        mode = row["weight_mode"]
        grouped.setdefault(mode, []).append(row)

    for mode in grouped:
        grouped[mode] = sorted(
            grouped[mode],
            key=lambda r: r["calibration_batches"],
        )

    return grouped


def plot_top1_vs_calibration(grouped, baseline, output_dir: Path):
    plt.figure()

    for mode, rows in grouped.items():
        x = [r["calibration_batches"] for r in rows]
        y = [r["top1"] for r in rows]
        plt.plot(x, y, marker="o", label=mode)

    if baseline and "top1" in baseline:
        plt.axhline(
            y=baseline["top1"],
            linestyle="--",
            label="FP32 baseline",
        )

    plt.xlabel("Calibration batches")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Top-1 Accuracy vs Calibration Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ptq_top1_vs_calibration.png", dpi=200)
    plt.close()


def plot_top5_vs_calibration(grouped, baseline, output_dir: Path):
    plt.figure()

    for mode, rows in grouped.items():
        x = [r["calibration_batches"] for r in rows]
        y = [r["top5"] for r in rows]
        plt.plot(x, y, marker="o", label=mode)

    if baseline and "top5" in baseline:
        plt.axhline(
            y=baseline["top5"],
            linestyle="--",
            label="FP32 baseline",
        )

    plt.xlabel("Calibration batches")
    plt.ylabel("Top-5 Accuracy")
    plt.title("Top-5 Accuracy vs Calibration Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ptq_top5_vs_calibration.png", dpi=200)
    plt.close()


def plot_latency_vs_calibration(grouped, output_dir: Path):
    plt.figure()

    for mode, rows in grouped.items():
        x = [r["calibration_batches"] for r in rows]
        y = [r["avg_latency_ms_per_image"] for r in rows]
        plt.plot(x, y, marker="o", label=mode)

    plt.xlabel("Calibration batches")
    plt.ylabel("Latency (ms/image)")
    plt.title("Latency vs Calibration Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ptq_latency_vs_calibration.png", dpi=200)
    plt.close()


def plot_speedup_vs_calibration(grouped, output_dir: Path):
    plt.figure()

    for mode, rows in grouped.items():
        x = [r["calibration_batches"] for r in rows]
        y = [r["speedup_vs_fp32"] for r in rows]
        plt.plot(x, y, marker="o", label=mode)

    plt.xlabel("Calibration batches")
    plt.ylabel("Speedup vs FP32")
    plt.title("Speedup vs Calibration Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ptq_speedup_vs_calibration.png", dpi=200)
    plt.close()


def plot_accuracy_loss_vs_calibration(grouped, output_dir: Path):
    plt.figure()

    for mode, rows in grouped.items():
        x = [r["calibration_batches"] for r in rows]
        y = [r["accuracy_loss_top1_pp"] for r in rows]
        plt.plot(x, y, marker="o", label=mode)

    plt.xlabel("Calibration batches")
    plt.ylabel("Top-1 Accuracy Loss (percentage points)")
    plt.title("Top-1 Accuracy Loss vs Calibration Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ptq_accuracy_loss_vs_calibration.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline, runs = load_runs(input_path)

    if not runs:
        raise ValueError(f"No runs found in {input_path}")

    grouped = group_by_weight_mode(runs)

    plot_top1_vs_calibration(grouped, baseline, output_dir)
    plot_top5_vs_calibration(grouped, baseline, output_dir)
    plot_latency_vs_calibration(grouped, output_dir)
    plot_speedup_vs_calibration(grouped, output_dir)
    plot_accuracy_loss_vs_calibration(grouped, output_dir)

    print(f"Saved PTQ plots to {output_dir}")


if __name__ == "__main__":
    main()