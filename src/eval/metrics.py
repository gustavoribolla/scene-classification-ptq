from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

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


def serialized_model_size_mb(model: nn.Module, path: Path) -> float:
    torch.save(model.state_dict(), path)
    size_bytes = path.stat().st_size
    return size_bytes / (1024 * 1024)
