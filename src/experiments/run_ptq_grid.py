from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.config import ProjectConfig, ensure_results_dir
from src.data.places365 import (
    build_eval_dataset,
    build_fake_places365,
    has_real_places365_data,
    make_loader,
)
from src.eval.metrics import evaluate_model, serialized_model_size_mb
from src.models.places365_resnet50 import load_resnet50
from src.quant.ptq_static import (
    build_qconfig,
    calibrate,
    choose_backend,
    convert_static_ptq,
    prepare_static_ptq,
)

# Default paths (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHTS = str(_PROJECT_ROOT / "resnet50_places365.pth.tar")
_DEFAULT_DATA_ROOT = os.getenv("PLACES365_ROOT", str(_PROJECT_ROOT / "places365_data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static PTQ ablations")
    parser.add_argument("--smoke", action="store_true", help="Use small subsets for quick validation")
    parser.add_argument("--smoke-calib-samples", type=int, default=512)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument("--weights-source", choices=["torchvision", "local"], default="local")
    parser.add_argument("--local-weights", type=str, default=_DEFAULT_WEIGHTS)
    parser.add_argument("--num-classes", type=int, default=365, help="Number of output classes (365 for Places365)")
    parser.add_argument(
        "--data-root",
        type=str,
        default=_DEFAULT_DATA_ROOT,
        help="Root directory for Places365 dataset (torchvision format)",
    )
    parser.add_argument("--results-dir", type=str, default=None, help="Directory for output artifacts")
    parser.add_argument(
        "--calibration-batches",
        type=int,
        nargs="+",
        default=None,
        help="Calibration batch counts to test (default: project config grid)",
    )
    parser.add_argument(
        "--weight-modes",
        choices=["per_tensor", "per_channel"],
        nargs="+",
        default=["per_tensor", "per_channel"],
        help="Weight quantization modes to test",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override configured DataLoader workers")
    return parser.parse_args()


def load_baseline_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_is_compatible(
    baseline: Dict[str, Any],
    mode: str,
    expected_samples: int,
    dataset_source: str,
) -> bool:
    return (
        baseline.get("mode") == mode
        and int(baseline.get("num_samples", -1)) == expected_samples
        and baseline.get("dataset_source", dataset_source) == dataset_source
    )


def add_comparison_fields(row: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    delta_top1 = float(baseline["top1"]) - float(row["top1"])
    delta_top5 = float(baseline["top5"]) - float(row["top5"])

    row["delta_top1_vs_fp32"] = delta_top1
    row["delta_top5_vs_fp32"] = delta_top5
    row["accuracy_loss_top1_pp"] = delta_top1 * 100.0
    row["accuracy_loss_top5_pp"] = delta_top5 * 100.0
    row["size_ratio_vs_fp32"] = baseline["model_size_mb"] / max(row["model_size_mb"], 1e-9)
    row["speedup_vs_fp32"] = baseline["avg_latency_ms_per_image"] / max(
        row["avg_latency_ms_per_image"], 1e-9
    )


def save_results_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    """Save PTQ grid results in a structured CSV table."""
    fieldnames = [
        "weight_mode",
        "calibration_batches",
        "calibration_samples",
        "calibration_observed_batches",
        "calibration_seconds",
        "top1",
        "top5",
        "delta_top1_vs_fp32",
        "delta_top5_vs_fp32",
        "accuracy_loss_top1_pp",
        "accuracy_loss_top5_pp",
        "model_size_mb",
        "avg_latency_ms_per_image",
        "size_ratio_vs_fp32",
        "speedup_vs_fp32",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_report(baseline: Dict[str, Any] | None, results: List[Dict[str, Any]], report_path: Path) -> None:
    lines = [
        "# PTQ Places365 Sprint 4 Report",
        "",
        "This report summarizes the Sprint 4 experiments for static Post-Training Quantization of ResNet50-Places365.",
        "The experiments compare per-tensor and per-channel INT8 quantization under different calibration sizes.",
        "",
    ]

    if baseline:
        lines.extend(
            [
                "## Baseline FP32",
                "",
                f"- Mode: {baseline.get('mode', 'unknown')}",
                f"- Dataset: {baseline.get('dataset_source', 'unknown')}",
                f"- Samples: {int(baseline.get('num_samples', 0))}",
                f"- Top-1 Accuracy: {baseline.get('top1', 0.0):.4f}",
                f"- Top-5 Accuracy: {baseline.get('top5', 0.0):.4f}",
                f"- Average latency: {baseline.get('avg_latency_ms_per_image', 0.0):.4f} ms/image",
                f"- Model size: {baseline.get('model_size_mb', 0.0):.2f} MB",
                "",
            ]
        )

    lines.extend(
        [
            "## Structured PTQ Results",
            "",
            "| Weight mode | Calibration batches | Calibration samples | Top-1 | Top-5 | Top-1 loss (pp) | Top-5 loss (pp) | Size (MB) | Latency (ms/img) | Speedup |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in results:
        lines.append(
            "| "
            f"{row['weight_mode']} | "
            f"{row['calibration_batches']} | "
            f"{row['calibration_samples']} | "
            f"{row['top1']:.4f} | "
            f"{row['top5']:.4f} | "
            f"{row['accuracy_loss_top1_pp']:.2f} | "
            f"{row['accuracy_loss_top5_pp']:.2f} | "
            f"{row['model_size_mb']:.2f} | "
            f"{row['avg_latency_ms_per_image']:.2f} | "
            f"{row['speedup_vs_fp32']:.2f}x |"
        )

    lines.append("")

    if results:
        best_accuracy = min(
            results,
            key=lambda r: (r["accuracy_loss_top1_pp"], -r["speedup_vs_fp32"]),
        )
        best_speed = max(results, key=lambda r: r["speedup_vs_fp32"])

        lines.extend(
            [
                "## Preliminary Analysis",
                "",
                (
                    f"The configuration with the best Top-1 preservation was "
                    f"`{best_accuracy['weight_mode']}` using "
                    f"{best_accuracy['calibration_batches']} calibration batches "
                    f"({best_accuracy['calibration_samples']} samples). "
                    f"It achieved Top-1 accuracy of {best_accuracy['top1']:.4f}, "
                    f"with a Top-1 loss of {best_accuracy['accuracy_loss_top1_pp']:.2f} percentage points."
                ),
                "",
                (
                    f"The fastest configuration was `{best_speed['weight_mode']}` using "
                    f"{best_speed['calibration_batches']} calibration batches. "
                    f"It achieved a speedup of {best_speed['speedup_vs_fp32']:.2f}x over FP32, "
                    f"with latency of {best_speed['avg_latency_ms_per_image']:.2f} ms/image."
                ),
                "",
                "Overall, these experiments allow a direct comparison between per-tensor and per-channel quantization, "
                "showing how calibration size affects accuracy, latency, model size, and the trade-off between efficiency and predictive performance.",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    if args.results_dir:
        cfg.results_dir = Path(args.results_dir)
    ensure_results_dir(cfg)

    backend = choose_backend()
    torch.backends.quantized.engine = backend
    print(f"Using quantization backend: {backend}")

    fp32_model = load_resnet50(
        weights_source=args.weights_source,
        local_weights=args.local_weights,
        num_classes=args.num_classes,
    )

    data_root = Path(args.data_root)
    use_real_data = data_root.exists() and has_real_places365_data(str(data_root), split="val")

    if use_real_data:
        print(f"[info] Using real Places365 data from {data_root}")
        # Use val split for both calibration and evaluation
        val_dataset, dataset_source = build_eval_dataset(
            root=str(data_root),
            split="val",
            image_size=cfg.image_size,
            crop_size=cfg.crop_size,
        )
        test_dataset = val_dataset
    elif args.smoke:
        print("[info] Places365 data not found, using FakeData in smoke mode.")
        val_dataset = build_fake_places365(
            num_samples=args.smoke_calib_samples,
            image_size=cfg.image_size,
            crop_size=cfg.crop_size,
            num_classes=args.num_classes,
        )
        test_dataset = build_fake_places365(
            num_samples=args.smoke_test_samples,
            image_size=cfg.image_size,
            crop_size=cfg.crop_size,
            num_classes=args.num_classes,
        )
        dataset_source = "fake-data"
    else:
        raise FileNotFoundError(
            f"Places365 data not found at {data_root}. "
            "Use --smoke for synthetic data or --data-root to specify the data directory."
        )

    mode = "smoke" if args.smoke else "full"
    batch_size = args.batch_size or cfg.batch_size
    num_workers = cfg.num_workers if args.num_workers is None else args.num_workers
    test_samples = min(args.smoke_test_samples, len(test_dataset)) if args.smoke else len(test_dataset)

    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=args.smoke_calib_samples if args.smoke else None,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=args.smoke_test_samples if args.smoke else None,
    )

    baseline_path = cfg.results_dir / "baseline_fp32.json"
    baseline = load_baseline_if_exists(baseline_path)
    baseline_source = str(baseline_path)

    if baseline is not None and not baseline_is_compatible(baseline, mode, test_samples, dataset_source):
        print("[warn] baseline_fp32.json is incompatible with this run; computing baseline inline.")
        baseline = None

    if baseline is None:
        print("[warn] baseline_fp32.json not found or not reusable; computing baseline inline.")
        baseline = evaluate_model(fp32_model, test_loader, device=cfg.device, desc="fp32-test-inline")
        fp32_size_path = cfg.results_dir / "tmp_fp32_state_dict.pt"
        baseline["model_size_mb"] = serialized_model_size_mb(fp32_model, fp32_size_path)
        fp32_size_path.unlink(missing_ok=True)
        baseline["mode"] = mode
        baseline["dataset_source"] = dataset_source
        baseline_source = "inline"

    rows: List[Dict[str, Any]] = []
    calibration_batches = args.calibration_batches or cfg.calibration_batches

    for weight_mode in args.weight_modes:
        for calib_batches in calibration_batches:
            qconfig = build_qconfig(backend=backend, weight_mode=weight_mode)
            prepared = prepare_static_ptq(fp32_model, qconfig=qconfig)
            calibration_stats = calibrate(prepared, val_loader, max_batches=calib_batches)
            int8_model = convert_static_ptq(prepared)

            quant_metrics = evaluate_model(
                int8_model,
                test_loader,
                device=cfg.device,
                desc=f"ptq-{weight_mode}-b{calib_batches}",
            )
            int8_size_path = cfg.results_dir / f"tmp_int8_{weight_mode}_{calib_batches}.pt"
            int8_size_mb = serialized_model_size_mb(int8_model, int8_size_path)
            int8_size_path.unlink(missing_ok=True)

            row = {
                "weight_mode": weight_mode,
                "calibration_batches": calib_batches,
                "top1": quant_metrics["top1"],
                "top5": quant_metrics["top5"],
                "avg_latency_ms_per_image": quant_metrics["avg_latency_ms_per_image"],
                "model_size_mb": int8_size_mb,
                "calibration_samples": calibration_stats.observed_samples,
                "calibration_observed_batches": calibration_stats.observed_batches,
                "calibration_seconds": calibration_stats.elapsed_seconds,
            }
            add_comparison_fields(row, baseline)
            rows.append(row)

    payload = {
        "mode": mode,
        "quant_backend": backend,
        "dataset_source": dataset_source,
        "baseline_source": baseline_source,
        "baseline": baseline,
        "runs": rows,
    }

    grid_path = cfg.results_dir / "ptq_grid.json"
    grid_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = cfg.results_dir / "ptq_grid.csv"
    save_results_csv(rows, csv_path)

    report_path = cfg.results_dir / "report.md"
    build_report(baseline, rows, report_path)

    print(f"Saved PTQ grid to {grid_path}")
    print(f"Saved CSV table to {csv_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
