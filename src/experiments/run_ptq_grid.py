from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.config import ProjectConfig, ensure_results_dir
from src.data.places365 import build_fake_places365, build_imagefolder, make_loader
from src.eval.metrics import evaluate_model, serialized_model_size_mb
from src.models.places365_resnet50 import load_resnet50
from src.quant.ptq_static import (
    build_qconfig,
    calibrate,
    choose_backend,
    convert_static_ptq,
    prepare_static_ptq,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static PTQ ablations")
    parser.add_argument("--smoke", action="store_true", help="Use small subsets for quick validation")
    parser.add_argument("--smoke-calib-samples", type=int, default=512)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument("--weights-source", choices=["torchvision", "local"], default="torchvision")
    parser.add_argument("--local-weights", type=str, default=None)
    return parser.parse_args()


def load_baseline_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(baseline: Dict[str, Any] | None, results: List[Dict[str, Any]], report_path: Path) -> None:
    lines = ["# PTQ Places365 Report", ""]
    if baseline:
        lines.extend(
            [
                "## Baseline FP32",
                f"- Top-1: {baseline.get('top1', 0.0):.4f}",
                f"- Top-5: {baseline.get('top5', 0.0):.4f}",
                f"- Avg latency (ms/img): {baseline.get('avg_latency_ms_per_image', 0.0):.4f}",
                f"- Model size (MB): {baseline.get('model_size_mb', 0.0):.2f}",
                "",
            ]
        )
    lines.append("## PTQ Runs")
    for row in results:
        lines.extend(
            [
                (
                    f"- weight_mode={row['weight_mode']}, calib_batches={row['calibration_batches']}, "
                    f"top1={row['top1']:.4f}, top5={row['top5']:.4f}, "
                    f"delta_top1={row['delta_top1_vs_fp32']:.4f}, size_ratio={row['size_ratio_vs_fp32']:.2f}x, "
                    f"speedup={row['speedup_vs_fp32']:.2f}x"
                )
            ]
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    ensure_results_dir(cfg)

    backend = choose_backend()
    torch.backends.quantized.engine = backend
    print(f"Using quantization backend: {backend}")

    fp32_model = load_resnet50(weights_source=args.weights_source, local_weights=args.local_weights)

    if args.smoke and not cfg.val_dir.exists():
        print("[info] Validation split not found, using FakeData in smoke mode.")
        val_dataset = build_fake_places365(
            num_samples=args.smoke_calib_samples,
            image_size=cfg.image_size,
            crop_size=cfg.crop_size,
        )
    else:
        val_dataset = build_imagefolder(cfg.val_dir, image_size=cfg.image_size, crop_size=cfg.crop_size)

    if args.smoke and not cfg.test_dir.exists():
        print("[info] Test split not found, using FakeData in smoke mode.")
        test_dataset = build_fake_places365(
            num_samples=args.smoke_test_samples,
            image_size=cfg.image_size,
            crop_size=cfg.crop_size,
        )
    else:
        test_dataset = build_imagefolder(cfg.test_dir, image_size=cfg.image_size, crop_size=cfg.crop_size)

    val_loader = make_loader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_samples=args.smoke_calib_samples if args.smoke else None,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_samples=args.smoke_test_samples if args.smoke else None,
    )

    baseline_path = cfg.results_dir / "baseline_fp32.json"
    baseline = load_baseline_if_exists(baseline_path)
    if baseline is None:
        print("[warn] baseline_fp32.json not found; computing baseline inline.")
        baseline = evaluate_model(fp32_model, test_loader, device=cfg.device, desc="fp32-test-inline")
        fp32_size_path = cfg.results_dir / "tmp_fp32_state_dict.pt"
        baseline["model_size_mb"] = serialized_model_size_mb(fp32_model, fp32_size_path)
        fp32_size_path.unlink(missing_ok=True)

    rows: List[Dict[str, Any]] = []
    for weight_mode in ("per_tensor", "per_channel"):
        for calib_batches in cfg.calibration_batches:
            qconfig = build_qconfig(backend=backend, weight_mode=weight_mode)
            prepared = prepare_static_ptq(fp32_model, qconfig=qconfig)
            calibrate(prepared, val_loader, max_batches=calib_batches)
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
                "delta_top1_vs_fp32": baseline["top1"] - quant_metrics["top1"],
                "size_ratio_vs_fp32": baseline["model_size_mb"] / max(int8_size_mb, 1e-9),
                "speedup_vs_fp32": baseline["avg_latency_ms_per_image"] / max(
                    quant_metrics["avg_latency_ms_per_image"], 1e-9
                ),
            }
            rows.append(row)

    payload = {
        "mode": "smoke" if args.smoke else "full",
        "quant_backend": backend,
        "baseline": baseline,
        "runs": rows,
    }

    grid_path = cfg.results_dir / "ptq_grid.json"
    grid_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = cfg.results_dir / "report.md"
    build_report(baseline, rows, report_path)

    print(f"Saved PTQ grid to {grid_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
