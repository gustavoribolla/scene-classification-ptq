from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(image_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_imagefolder(split_dir: Path, image_size: int = 256, crop_size: int = 224) -> datasets.ImageFolder:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    return datasets.ImageFolder(root=str(split_dir), transform=build_eval_transform(image_size, crop_size))


def build_fake_places365(
    num_samples: int,
    image_size: int = 256,
    crop_size: int = 224,
    num_classes: int = 365,
) -> datasets.FakeData:
    return datasets.FakeData(
        size=num_samples,
        image_size=(3, crop_size, crop_size),
        num_classes=num_classes,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    )


def make_loader(
    dataset: datasets.ImageFolder,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int] = None,
) -> DataLoader:
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
