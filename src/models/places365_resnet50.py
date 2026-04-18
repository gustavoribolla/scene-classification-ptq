from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def _load_local_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    # Common checkpoints prefix params with "module."
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys while loading local checkpoint: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys while loading local checkpoint: {len(unexpected)}")
    return model


def load_resnet50(
    weights_source: Literal["torchvision", "local"] = "torchvision",
    local_weights: Optional[str] = None,
    num_classes: int = 365,
) -> nn.Module:
    """Load a ResNet-50 model with Places365 or ImageNet weights.

    When ``weights_source='local'``, the architecture is created with
    ``num_classes`` output units (default 365 for Places365) so that the
    checkpoint's FC layer matches.
    """
    if weights_source == "torchvision":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = resnet50(weights=None, num_classes=num_classes)
        if not local_weights:
            raise ValueError("local_weights path is required when weights_source='local'")
        path = Path(local_weights)
        if not path.exists():
            raise FileNotFoundError(f"Local weights not found: {path}")
        model = _load_local_checkpoint(model, str(path))
    model.eval()
    return model
