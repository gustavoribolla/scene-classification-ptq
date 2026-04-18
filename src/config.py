from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ProjectConfig:
    places365_root: Path = Path(os.getenv("PLACES365_ROOT", ""))
    val_dir_override: str = os.getenv("PLACES365_VAL_DIR", "")
    test_dir_override: str = os.getenv("PLACES365_TEST_DIR", "")
    results_dir: Path = Path("results")

    image_size: int = 256
    crop_size: int = 224
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))

    calibration_batches: List[int] = field(default_factory=lambda: [10, 50, 100, 500])
    device: str = "cpu"

    @property
    def val_dir(self) -> Path:
        if self.val_dir_override:
            return Path(self.val_dir_override)
        return self.places365_root / "val"

    @property
    def test_dir(self) -> Path:
        if self.test_dir_override:
            return Path(self.test_dir_override)
        return self.places365_root / "test"


def ensure_results_dir(cfg: ProjectConfig) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
