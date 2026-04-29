from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.config import ProjectConfig, ensure_results_dir
from src.data.places365 import (
    build_eval_dataset,
    build_fake_places365,
    has_real_places365_data,
    make_loader,
)
from src.models.places365_resnet50 import load_resnet50

from src.eval.metrics import (
    evaluate_model_with_confusion_matrix,
    plot_binary_confusion_matrix_from_multiclass,
    plot_confusion_matrix,
    save_confusion_matrix_csv,
    serialized_model_size_mb,
)

# Default checkpoint path (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHTS = str(_PROJECT_ROOT / "resnet50_places365.pth.tar")
_DEFAULT_DATA_ROOT = os.getenv("PLACES365_ROOT", str(_PROJECT_ROOT / "places365_data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FP32 baseline evaluation")

    parser.add_argument("--smoke", action="store_true", help="Use a small subset for quick validation")
    parser.add_argument("--smoke-samples", type=int, default=512)

    parser.add_argument("--weights-source", choices=["torchvision", "local"], default="local")
    parser.add_argument("--local-weights", type=str, default=_DEFAULT_WEIGHTS)

    parser.add_argument("--num-classes", type=int, default=365, help="Number of output classes")
    parser.add_argument(
        "--target-class",
        type=int,
        default=0,
        help="Class index used for the binary one-vs-rest confusion matrix",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=_DEFAULT_DATA_ROOT,
        help="Root directory for Places365 dataset",
    )

    parser.add_argument("--results-dir", type=str, default=None, help="Directory for output artifacts")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override configured DataLoader workers")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ProjectConfig()

    if args.results_dir:
        cfg.results_dir = Path(args.results_dir)

    ensure_results_dir(cfg)

    model = load_resnet50(
        weights_source=args.weights_source,
        local_weights=args.local_weights,
        num_classes=args.num_classes,
    )

    data_root = Path(args.data_root)
    use_real_data = data_root.exists() and has_real_places365_data(str(data_root), split="val")

    if use_real_data:
        print(f"[info] Using real Places365 data from {data_root}")

        test_dataset, dataset_source = build_eval_dataset(
            root=str(data_root),
            split="val",
            image_size=cfg.image_size,
            crop_size=cfg.crop_size,
        )

    elif args.smoke:
        print("[info] Places365 data not found, using FakeData in smoke mode.")

        test_dataset = build_fake_places365(
            num_samples=args.smoke_samples,
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

    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size or cfg.batch_size,
        num_workers=cfg.num_workers if args.num_workers is None else args.num_workers,
        max_samples=args.smoke_samples if args.smoke else None,
    )

    metrics, confusion = evaluate_model_with_confusion_matrix(
        model,
        test_loader,
        num_classes=args.num_classes,
        device=cfg.device,
        desc="fp32-test",
    )

    size_path = cfg.results_dir / "tmp_fp32_state_dict.pt"
    metrics["model_size_mb"] = serialized_model_size_mb(model, size_path)
    size_path.unlink(missing_ok=True)

    metrics["mode"] = "smoke" if args.smoke else "full"
    metrics["dataset_source"] = dataset_source
    metrics["target_class_for_binary_confusion_matrix"] = args.target_class

    output_path = cfg.results_dir / "baseline_fp32.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    conf_csv_path = cfg.results_dir / "baseline_confusion_matrix.csv"
    conf_multiclass_png_path = cfg.results_dir / "baseline_confusion_matrix_multiclass.png"
    conf_binary_png_path = cfg.results_dir / "baseline_confusion_matrix.png"

    save_confusion_matrix_csv(confusion, conf_csv_path)

    # Matriz completa 365x365
    plot_confusion_matrix(confusion, conf_multiclass_png_path)

    # Matriz 2x2 one-vs-rest para uma classe específica
    plot_binary_confusion_matrix_from_multiclass(
        confusion,
        conf_binary_png_path,
        target_class=args.target_class,
    )

    print(f"Saved baseline metrics to {output_path}")
    print(f"Saved confusion matrix CSV to {conf_csv_path}")
    print(f"Saved multiclass confusion matrix image to {conf_multiclass_png_path}")
    print(f"Saved binary confusion matrix image to {conf_binary_png_path}")


if __name__ == "__main__":
    main()