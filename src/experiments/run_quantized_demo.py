from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config import ProjectConfig, ensure_results_dir
from src.data.places365 import build_eval_dataset, build_eval_transform, has_real_places365_data, make_loader
from src.models.places365_resnet50 import load_resnet50
from src.quant.ptq_static import (
    build_qconfig,
    calibrate,
    choose_backend,
    convert_static_ptq,
    prepare_static_ptq,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHTS = str(_PROJECT_ROOT / "resnet50_places365.pth.tar")
_DEFAULT_DATA_ROOT = os.getenv("PLACES365_ROOT", str(_PROJECT_ROOT / "places365_data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run a tangible INT8 Places365 demo")
    parser.add_argument("--data-root", type=str, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--local-weights", type=str, default=_DEFAULT_WEIGHTS)
    parser.add_argument("--results-dir", type=str, default="results/quantized_demo")
    parser.add_argument("--num-classes", type=int, default=365)
    parser.add_argument("--weight-mode", choices=["per_tensor", "per_channel"], default="per_channel")
    parser.add_argument("--calibration-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--image-path", action="append", default=[], help="Optional image(s) to classify")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild INT8 model even if an artifact exists")
    return parser.parse_args()


def load_category_names(data_root: Path, num_classes: int) -> list[str]:
    categories_path = data_root / "categories_places365.txt"
    names = [f"class_{idx}" for idx in range(num_classes)]
    if not categories_path.exists():
        return names

    for line in categories_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        raw_name, raw_index = parts[0], parts[-1]
        try:
            index = int(raw_index)
        except ValueError:
            continue
        if 0 <= index < len(names):
            names[index] = raw_name.strip("/").replace("/", " / ").replace("_", " ")
    return names


def dataset_image_path(dataset: Any, index: int) -> str | None:
    if hasattr(dataset, "imgs") and index < len(dataset.imgs):
        return str(dataset.imgs[index][0])
    if hasattr(dataset, "samples") and index < len(dataset.samples):
        return str(dataset.samples[index][0])
    return None


def classify_tensor(model: torch.nn.Module, image: torch.Tensor, topk: int, category_names: list[str]) -> list[dict[str, Any]]:
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).cpu())
        probabilities = torch.softmax(logits, dim=1)
        scores, indices = probabilities.topk(topk, dim=1)

    predictions = []
    for score, index in zip(scores[0].tolist(), indices[0].tolist()):
        predictions.append(
            {
                "class_index": int(index),
                "class_name": category_names[int(index)],
                "confidence": float(score),
            }
        )
    return predictions


def classify_dataset_samples(
    model: torch.nn.Module,
    dataset: Any,
    category_names: list[str],
    sample_count: int,
    topk: int,
) -> list[dict[str, Any]]:
    rows = []
    top1_hits = 0
    topk_hits = 0
    for index in range(min(sample_count, len(dataset))):
        image, label = dataset[index]
        predictions = classify_tensor(model, image, topk=topk, category_names=category_names)
        predicted_indices = [row["class_index"] for row in predictions]
        top1_hits += int(predicted_indices[0] == int(label))
        topk_hits += int(int(label) in predicted_indices)
        rows.append(
            {
                "sample_index": index,
                "image_path": dataset_image_path(dataset, index),
                "true_class_index": int(label),
                "true_class_name": category_names[int(label)],
                "top1_correct": predicted_indices[0] == int(label),
                f"top{topk}_correct": int(label) in predicted_indices,
                "predictions": predictions,
            }
        )

    if rows:
        rows.append(
            {
                "summary": {
                    "sample_count": len(rows),
                    "top1_on_demo_samples": top1_hits / len(rows),
                    f"top{topk}_on_demo_samples": topk_hits / len(rows),
                }
            }
        )
    return rows


def classify_external_images(
    model: torch.nn.Module,
    image_paths: list[str],
    category_names: list[str],
    image_size: int,
    crop_size: int,
    topk: int,
) -> list[dict[str, Any]]:
    transform = build_eval_transform(image_size=image_size, crop_size=crop_size)
    rows = []
    for image_path in image_paths:
        with Image.open(image_path).convert("RGB") as image:
            image_tensor = transform(image)
        rows.append(
            {
                "image_path": image_path,
                "predictions": classify_tensor(model, image_tensor, topk=topk, category_names=category_names),
            }
        )
    return rows


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Quantized Places365 Demo",
        "",
        "## Artifact",
        f"- Quantized model: `{payload['quantized_model_path']}`",
        f"- Backend: `{payload['quant_backend']}`",
        f"- Weight mode: `{payload['weight_mode']}`",
        f"- Calibration batches: `{payload['calibration']['observed_batches']}`",
        f"- Calibration samples: `{payload['calibration']['observed_samples']}`",
        "",
        "## Predictions",
    ]

    for row in payload["predictions"]:
        if "summary" in row:
            summary = row["summary"]
            topk_key = f"top{payload['topk']}_on_demo_samples"
            lines.extend(
                [
                    "",
                    "## Demo Sample Accuracy",
                    f"- Top-1 on shown samples: {summary['top1_on_demo_samples']:.2%}",
                    f"- Top-{payload['topk']} on shown samples: {summary[topk_key]:.2%}",
                ]
            )
            continue

        label = row.get("true_class_name", "unknown")
        image_path = row.get("image_path", "external image")
        best = row["predictions"][0]
        lines.extend(
            [
                "",
                f"### {Path(image_path).name}",
                f"- Image: `{image_path}`",
                f"- True label: `{label}`",
                f"- Top-1 prediction: `{best['class_name']}` ({best['confidence']:.2%})",
                "- Top predictions:",
            ]
        )
        for prediction in row["predictions"]:
            lines.append(f"  - `{prediction['class_name']}`: {prediction['confidence']:.2%}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    cfg.results_dir = Path(args.results_dir)
    ensure_results_dir(cfg)

    data_root = Path(args.data_root)
    if not data_root.exists() or not has_real_places365_data(str(data_root), split="val"):
        raise FileNotFoundError(
            f"Places365 data not found at {data_root}. Set PLACES365_ROOT or pass --data-root."
        )

    category_names = load_category_names(data_root, num_classes=args.num_classes)
    quantized_model_path = cfg.results_dir / "places365_resnet50_int8_torchscript.pt"
    metadata_path = cfg.results_dir / "quantized_model_metadata.json"

    supported = torch.backends.quantized.supported_engines
    print(f"[info] Supported quantized engines: {supported}")

    if "fbgemm" in supported:
        backend = "fbgemm"
    elif "x86" in supported:
        backend = "x86"
    elif "onednn" in supported:
        backend = "onednn"
    else:
        raise RuntimeError(f"No supported quantized backend found. Available: {supported}")

    torch.backends.quantized.engine = backend
    print(f"[info] Using quantized backend: {backend}")

    int8_model = None
    if quantized_model_path.exists() and not args.rebuild:
        print(f"[info] Loading existing TorchScript INT8 model: {quantized_model_path}")
        try:
            int8_model = torch.jit.load(str(quantized_model_path), map_location="cpu")
        except Exception as exc:
            print(f"[warn] Could not load existing INT8 artifact ({exc}); rebuilding it.")
            int8_model = None

    if int8_model is not None:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            calibration_payload = metadata.get("calibration", {})
        else:
            calibration_payload = {
                "requested_batches": args.calibration_batches,
                "observed_batches": "loaded_existing_model",
                "observed_samples": "loaded_existing_model",
                "elapsed_seconds": 0.0,
            }
    else:
        print("[info] Building INT8 model with static PTQ calibration.")
        fp32_model = load_resnet50(
            weights_source="local",
            local_weights=args.local_weights,
            num_classes=args.num_classes,
        )
        val_dataset, dataset_source = build_eval_dataset(
            root=str(data_root),
            split="val",
            image_size=cfg.image_size,
            crop_size=cfg.crop_size,
        )
        calibration_loader = make_loader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.batch_size * args.calibration_batches,
        )

        qconfig = build_qconfig(backend=backend, weight_mode=args.weight_mode)
        prepared = prepare_static_ptq(fp32_model, qconfig=qconfig)
        calibration_stats = calibrate(prepared, calibration_loader, max_batches=args.calibration_batches)
        int8_model = convert_static_ptq(prepared)
        scripted_model = torch.jit.script(int8_model)
        scripted_model.save(str(quantized_model_path))
        int8_model = torch.jit.load(str(quantized_model_path), map_location="cpu")
        calibration_payload = calibration_stats.to_dict()
        metadata_path.write_text(
            json.dumps(
                {
                    "quantized_model_path": str(quantized_model_path),
                    "quant_backend": backend,
                    "weight_mode": args.weight_mode,
                    "data_root": str(data_root),
                    "calibration": calibration_payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[info] Saved TorchScript INT8 model: {quantized_model_path}")
        print(f"[info] Saved metadata: {metadata_path}")
        print(f"[info] Dataset source: {dataset_source}")

    int8_model.eval()
    val_dataset, _ = build_eval_dataset(
        root=str(data_root),
        split="val",
        image_size=cfg.image_size,
        crop_size=cfg.crop_size,
    )

    if args.image_path:
        predictions = classify_external_images(
            int8_model,
            image_paths=args.image_path,
            category_names=category_names,
            image_size=cfg.image_size,
            crop_size=cfg.crop_size,
            topk=args.topk,
        )
    else:
        predictions = classify_dataset_samples(
            int8_model,
            dataset=val_dataset,
            category_names=category_names,
            sample_count=args.sample_count,
            topk=args.topk,
        )

    payload = {
        "quantized_model_path": str(quantized_model_path),
        "quant_backend": backend,
        "weight_mode": args.weight_mode,
        "calibration": calibration_payload,
        "topk": args.topk,
        "predictions": predictions,
    }
    predictions_path = cfg.results_dir / "predictions.json"
    report_path = cfg.results_dir / "demo_report.md"
    predictions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(report_path, payload)

    print(f"[done] Predictions saved to: {predictions_path}")
    print(f"[done] Demo report saved to: {report_path}")


if __name__ == "__main__":
    main()
