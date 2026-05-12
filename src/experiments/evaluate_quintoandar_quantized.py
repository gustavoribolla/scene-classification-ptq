from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config import ProjectConfig, ensure_results_dir
from src.data.places365 import build_eval_transform
from src.experiments.run_quantized_demo import load_category_names


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "places365_data"
_DEFAULT_MANIFEST = _PROJECT_ROOT / "data" / "external" / "quintoandar" / "manifest.csv"
_DEFAULT_MODEL = _PROJECT_ROOT / "results" / "quantized_demo" / "places365_resnet50_int8_torchscript.pt"
_DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "results" / "quintoandar_quantized_eval"


BASE_LABEL_TO_PLACES365_RAW: dict[str, tuple[str, ...]] = {
    "sala": (
        "/l/living_room",
        "/d/dining_room",
        "/t/television_room",
    ),
    "quarto": (
        "/b/bedroom",
        "/b/bedchamber",
        "/d/dorm_room",
    ),
    "cozinha": (
        "/k/kitchen",
        "/p/pantry",
        "/r/restaurant_kitchen",
    ),
    "banheiro": (
        "/b/bathroom",
        "/s/shower",
    ),
    "varanda": (
        "/b/balcony/interior",
        "/b/balcony/exterior",
        "/p/patio",
        "/p/porch",
    ),
    "area_servico": (
        "/u/utility_room",
        "/s/storage_room",
        "/l/laundromat",
    ),
    "escritorio": (
        "/h/home_office",
        "/o/office",
    ),
    "corredor": (
        "/c/corridor",
    ),
    "closet": (
        "/c/closet",
        "/d/dressing_room",
    ),
    "garagem": (
        "/g/garage/indoor",
        "/g/garage/outdoor",
        "/p/parking_garage/indoor",
        "/p/parking_garage/outdoor",
        "/d/driveway",
    ),
    "jardim": (
        "/y/yard",
        "/l/lawn",
        "/c/courtyard",
        "/f/formal_garden",
        "/t/topiary_garden",
    ),
    "area_externa": (
        "/p/patio",
        "/p/porch",
        "/c/courtyard",
        "/y/yard",
        "/l/lawn",
        "/d/driveway",
        "/h/house",
        "/b/building_facade",
        "/a/apartment_building/outdoor",
        "/b/balcony/exterior",
    ),
    "piscina": (
        "/s/swimming_pool/indoor",
        "/s/swimming_pool/outdoor",
    ),
    "academia": (
        "/g/gymnasium/indoor",
        "/m/martial_arts_gym",
    ),
    "churrasqueira": (
        "/p/patio",
        "/c/courtyard",
        "/d/dining_room",
        "/k/kitchen",
    ),
}


@dataclass(frozen=True)
class ManifestRow:
    image_path: Path
    label: str
    listing_id: str
    subtitle: str
    detail_url: str
    area_m2: str
    is_furnished: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate QuintoAndar external room photos with the quantized "
            "Places365 TorchScript model."
        )
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=365)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    return parser.parse_args()


def normalize_raw_category(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("/"):
        return raw
    return "/" + raw.strip("/")


def display_category(raw: str) -> str:
    return normalize_raw_category(raw).strip("/").replace("/", " / ").replace("_", " ")


def load_raw_category_names(categories_path: Path, num_classes: int) -> list[str]:
    names = [f"/class_{idx}" for idx in range(num_classes)]
    if not categories_path.exists():
        return names

    for line in categories_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            index = int(parts[-1])
        except ValueError:
            continue
        if 0 <= index < len(names):
            names[index] = normalize_raw_category(parts[0])
    return names


def load_manifest(path: Path, limit: int | None) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.exists():
                print(f"[warn] missing image, skipping: {image_path}")
                continue
            rows.append(
                ManifestRow(
                    image_path=image_path,
                    label=row["label"],
                    listing_id=row["listing_id"],
                    subtitle=row["subtitle"],
                    detail_url=row["detail_url"],
                    area_m2=row["area_m2"],
                    is_furnished=row["is_furnished"],
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def accepted_raw_categories(label: str) -> set[str]:
    accepted: set[str] = set()
    for base_label, raw_categories in BASE_LABEL_TO_PLACES365_RAW.items():
        if base_label in label:
            accepted.update(normalize_raw_category(category) for category in raw_categories)
    return accepted


def classify_image(
    model: torch.nn.Module,
    image_path: Path,
    transform: Any,
    topk: int,
    category_names: list[str],
    raw_category_names: list[str],
) -> list[dict[str, Any]]:
    with Image.open(image_path).convert("RGB") as image:
        image_tensor = transform(image)

    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).cpu())
        probabilities = torch.softmax(logits, dim=1)
        scores, indices = probabilities.topk(topk, dim=1)

    return [
        {
            "class_index": int(index),
            "class_name": category_names[int(index)],
            "raw_class_name": raw_category_names[int(index)],
            "confidence": float(score),
        }
        for score, index in zip(scores[0].tolist(), indices[0].tolist())
    ]


def summarize(rows: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    total = len(rows)
    top1_hits = sum(1 for row in rows if row["top1_match"])
    topk_hits = sum(1 for row in rows if row["topk_match"])

    by_label: dict[str, dict[str, Any]] = {}
    label_counts = Counter(row["label"] for row in rows)
    label_top1 = Counter(row["label"] for row in rows if row["top1_match"])
    label_topk = Counter(row["label"] for row in rows if row["topk_match"])
    label_pred = defaultdict(Counter)
    for row in rows:
        label_pred[row["label"]][row["predictions"][0]["class_name"]] += 1

    for label in sorted(label_counts):
        count = label_counts[label]
        by_label[label] = {
            "count": count,
            "top1_match": label_top1[label],
            f"top{topk}_match": label_topk[label],
            "top1_accuracy": label_top1[label] / count,
            f"top{topk}_accuracy": label_topk[label] / count,
            "most_common_top1_predictions": label_pred[label].most_common(10),
        }

    return {
        "sample_count": total,
        "top1_match": top1_hits,
        f"top{topk}_match": topk_hits,
        "top1_accuracy": top1_hits / total if total else 0.0,
        f"top{topk}_accuracy": topk_hits / total if total else 0.0,
        "by_label": by_label,
    }


def write_predictions_csv(path: Path, rows: list[dict[str, Any]], topk: int) -> None:
    fieldnames = [
        "image_path",
        "label",
        "subtitle",
        "listing_id",
        "area_m2",
        "is_furnished",
        "accepted_places365",
        "top1_class",
        "top1_confidence",
        "top1_match",
        f"top{topk}_match",
        "topk_predictions",
        "detail_url",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            predictions = row["predictions"]
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "subtitle": row["subtitle"],
                    "listing_id": row["listing_id"],
                    "area_m2": row["area_m2"],
                    "is_furnished": row["is_furnished"],
                    "accepted_places365": "; ".join(row["accepted_class_names"]),
                    "top1_class": predictions[0]["class_name"],
                    "top1_confidence": predictions[0]["confidence"],
                    "top1_match": row["top1_match"],
                    f"top{topk}_match": row["topk_match"],
                    "topk_predictions": "; ".join(
                        f"{item['class_name']}={item['confidence']:.4f}" for item in predictions
                    ),
                    "detail_url": row["detail_url"],
                }
            )


def write_report(path: Path, metrics: dict[str, Any], args: argparse.Namespace) -> None:
    topk_key = f"top{args.topk}_accuracy"
    lines = [
        "# QuintoAndar External Evaluation",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Quantized model: `{args.model_path}`",
        f"- Samples: {metrics['sample_count']}",
        f"- Top-1 mapped accuracy: {metrics['top1_accuracy']:.2%}",
        f"- Top-{args.topk} mapped accuracy: {metrics[topk_key]:.2%}",
        "",
        "## Label Mapping",
    ]
    for label in sorted(BASE_LABEL_TO_PLACES365_RAW):
        accepted = ", ".join(display_category(raw) for raw in BASE_LABEL_TO_PLACES365_RAW[label])
        lines.append(f"- `{label}` -> {accepted}")

    lines.extend(["", "## Per Label"])
    for label, row in metrics["by_label"].items():
        lines.extend(
            [
                "",
                f"### {label}",
                f"- Count: {row['count']}",
                f"- Top-1 mapped accuracy: {row['top1_accuracy']:.2%}",
                f"- Top-{args.topk} mapped accuracy: {row[f'top{args.topk}_accuracy']:.2%}",
                "- Most common top-1 predictions:",
            ]
        )
        for class_name, count in row["most_common_top1_predictions"][:5]:
            lines.append(f"  - `{class_name}`: {count}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_quantized_backend() -> str:
    supported = torch.backends.quantized.supported_engines
    for backend in ("fbgemm", "x86", "onednn", "qnnpack"):
        if backend in supported:
            torch.backends.quantized.engine = backend
            return backend
    raise RuntimeError(f"No supported quantized backend found. Available: {supported}")


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    cfg.results_dir = args.results_dir
    ensure_results_dir(cfg)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Quantized model not found: {args.model_path}")

    category_names = load_category_names(args.data_root, args.num_classes)
    raw_category_names = load_raw_category_names(
        args.data_root / "categories_places365.txt",
        args.num_classes,
    )
    transform = build_eval_transform(image_size=args.image_size, crop_size=args.crop_size)
    quantized_backend = choose_quantized_backend()
    print(f"[info] using quantized backend: {quantized_backend}")
    model = torch.jit.load(str(args.model_path), map_location="cpu")
    model.eval()

    manifest_rows = load_manifest(args.manifest, args.limit)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows, start=1):
        accepted_raw = accepted_raw_categories(row.label)
        if not accepted_raw:
            print(f"[warn] no Places365 mapping for label={row.label}; skipping {row.image_path}")
            continue
        predictions = classify_image(
            model=model,
            image_path=row.image_path,
            transform=transform,
            topk=args.topk,
            category_names=category_names,
            raw_category_names=raw_category_names,
        )
        predicted_raw = [item["raw_class_name"] for item in predictions]
        results.append(
            {
                "image_path": str(row.image_path),
                "label": row.label,
                "subtitle": row.subtitle,
                "listing_id": row.listing_id,
                "area_m2": row.area_m2,
                "is_furnished": row.is_furnished,
                "detail_url": row.detail_url,
                "accepted_raw_classes": sorted(accepted_raw),
                "accepted_class_names": [display_category(raw) for raw in sorted(accepted_raw)],
                "top1_match": predicted_raw[0] in accepted_raw,
                "topk_match": any(raw in accepted_raw for raw in predicted_raw),
                "predictions": predictions,
            }
        )
        if index % 100 == 0:
            print(f"[info] evaluated {index}/{len(manifest_rows)} images")

    metrics = summarize(results, args.topk)
    payload = {
        "model_path": str(args.model_path),
        "manifest": str(args.manifest),
        "quantized_backend": quantized_backend,
        "topk": args.topk,
        "metrics": metrics,
    }

    metrics_path = cfg.results_dir / "metrics.json"
    predictions_jsonl_path = cfg.results_dir / "predictions.jsonl"
    predictions_csv_path = cfg.results_dir / "predictions.csv"
    report_path = cfg.results_dir / "report.md"

    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with predictions_jsonl_path.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_predictions_csv(predictions_csv_path, results, args.topk)
    write_report(report_path, metrics, args)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] Metrics saved to: {metrics_path}")
    print(f"[done] Predictions CSV saved to: {predictions_csv_path}")
    print(f"[done] Predictions JSONL saved to: {predictions_jsonl_path}")
    print(f"[done] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
