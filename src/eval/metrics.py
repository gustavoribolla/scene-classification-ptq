from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        scores = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            scores.append(float(correct_k))
        return scores


def evaluate_model(model: nn.Module, loader: DataLoader, device: str = "cpu", desc: str = "eval") -> Dict[str, float]:
    model.to(device)
    model.eval()

    n_samples = 0
    top1 = 0.0
    top5 = 0.0
    elapsed = 0.0

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)

            start = time.perf_counter()
            logits = model(images)
            elapsed += time.perf_counter() - start

            b_top1, b_top5 = _accuracy_topk(logits, labels, topk=(1, 5))
            bs = labels.size(0)

            n_samples += bs
            top1 += b_top1
            top5 += b_top5

    avg_latency_ms = (elapsed / max(n_samples, 1)) * 1000.0

    return {
        "num_samples": float(n_samples),
        "top1": top1 / max(n_samples, 1),
        "top5": top5 / max(n_samples, 1),
        "avg_latency_ms_per_image": avg_latency_ms,
        "total_eval_seconds": elapsed,
    }


def evaluate_model_with_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str = "cpu",
    desc: str = "eval",
):
    model.to(device)
    model.eval()

    n_samples = 0
    top1 = 0.0
    top5 = 0.0
    elapsed = 0.0

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)

            start = time.perf_counter()
            logits = model(images)
            elapsed += time.perf_counter() - start

            preds = torch.argmax(logits, dim=1)

            b_top1, b_top5 = _accuracy_topk(logits, labels, topk=(1, 5))
            bs = labels.size(0)

            n_samples += bs
            top1 += b_top1
            top5 += b_top5

            for true_label, pred_label in zip(labels.view(-1), preds.view(-1)):
                confusion[true_label.long(), pred_label.long()] += 1

    avg_latency_ms = (elapsed / max(n_samples, 1)) * 1000.0

    metrics = {
        "num_samples": float(n_samples),
        "top1": top1 / max(n_samples, 1),
        "top5": top5 / max(n_samples, 1),
        "avg_latency_ms_per_image": avg_latency_ms,
        "total_eval_seconds": elapsed,
    }

    return metrics, confusion


def save_confusion_matrix_csv(confusion: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(confusion.cpu().numpy().tolist())


def plot_confusion_matrix(confusion: torch.Tensor, path: Path) -> None:
    """
    Matriz multiclasses completa.
    Útil para Places365, mas visualmente densa por ter 365 classes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion.cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title("Confusion Matrix - Multiclass")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_binary_confusion_matrix_from_multiclass(
    confusion: torch.Tensor,
    path: Path,
    target_class: int = 0,
) -> None:
    """
    Gera uma matriz 2x2 no formato one-vs-rest.

    Para um problema com 365 classes, não existe uma única matriz 2x2 global.
    Então esta função compara uma classe específica contra todas as outras.

    Layout:
        [[TP, FP],
         [FN, TN]]
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion.cpu().numpy()

    tp = cm[target_class, target_class]
    fp = cm[:, target_class].sum() - tp
    fn = cm[target_class, :].sum() - tp
    tn = cm.sum() - tp - fp - fn

    binary_cm = [[tp, fp], [fn, tn]]
    labels = [["TP", "FP"], ["FN", "TN"]]

    plt.figure(figsize=(6, 5))
    plt.imshow(binary_cm, interpolation="nearest")

    plt.title(f"Binary Confusion Matrix - Class {target_class} vs Rest")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks([0, 1], [f"Class {target_class}", "Rest"])
    plt.yticks([0, 1], [f"Class {target_class}", "Rest"])

    for i in range(2):
        for j in range(2):
            value = int(binary_cm[i][j])
            plt.text(
                j,
                i,
                f"{labels[i][j]}\n{value}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def serialized_model_size_mb(model: nn.Module, path: Path) -> float:
    torch.save(model.state_dict(), path)
    size_bytes = path.stat().st_size
    return size_bytes / (1024 * 1024)