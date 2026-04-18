from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ProjectConfig, ensure_results_dir
from src.data.places365 import build_fake_places365, build_places365, make_loader
from src.eval.metrics import evaluate_model, serialized_model_size_mb
from src.models.places365_resnet50 import load_resnet50

# Default checkpoint path (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHTS = str(_PROJECT_ROOT / "resnet50_places365.pth.tar")
_DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "places365_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FP32 baseline evaluation")
    parser.add_argument("--smoke", action="store_true", help="Use a small subset for quick validation")
    parser.add_argument("--smoke-samples", type=int, default=512)
    parser.add_argument("--weights-source", choices=["torchvision", "local"], default="local")
    parser.add_argument("--local-weights", type=str, default=_DEFAULT_WEIGHTS)
    parser.add_argument("--num-classes", type=int, default=365, help="Number of output classes (365 for Places365)")
    parser.add_argument("--data-root", type=str, default=_DEFAULT_DATA_ROOT,
                        help="Root directory for Places365 dataset (torchvision format)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    ensure_results_dir(cfg)

    model = load_resnet50(
        weights_source=args.weights_source,
        local_weights=args.local_weights,
        num_classes=args.num_classes,
    )

    data_root = Path(args.data_root)
    use_real_data = data_root.exists() and (data_root / "val_256").exists()

    if use_real_data:
        print(f"[info] Using real Places365 data from {data_root}")
        test_dataset = build_places365(
            root=str(data_root),
            split="val",
            small=True,
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
    else:
        raise FileNotFoundError(
            f"Places365 data not found at {data_root}. "
            "Use --smoke for synthetic data or --data-root to specify the data directory."
        )

    test_loader = make_loader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_samples=args.smoke_samples if args.smoke else None,
    )

    metrics = evaluate_model(model, test_loader, device=cfg.device, desc="fp32-test")
    size_path = cfg.results_dir / "tmp_fp32_state_dict.pt"
    metrics["model_size_mb"] = serialized_model_size_mb(model, size_path)
    size_path.unlink(missing_ok=True)
    metrics["mode"] = "smoke" if args.smoke else "full"

    output_path = cfg.results_dir / "baseline_fp32.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
