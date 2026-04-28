from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.ao.quantization import (
    QConfig,
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx


@dataclass(frozen=True)
class CalibrationStats:
    requested_batches: int
    observed_batches: int
    observed_samples: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def choose_backend() -> str:
    if "fbgemm" in torch.backends.quantized.supported_engines:
        return "fbgemm"
    return "qnnpack"


def build_qconfig(backend: str, weight_mode: str = "per_channel") -> QConfig:
    if weight_mode == "per_channel":
        act_observer = HistogramObserver.with_args(reduce_range=False)
        wt_observer = PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )
        return QConfig(activation=act_observer, weight=wt_observer)
    if weight_mode == "per_tensor":
        act_observer = HistogramObserver.with_args(reduce_range=False)
        wt_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        return QConfig(activation=act_observer, weight=wt_observer)
    return torch.ao.quantization.get_default_qconfig(backend)


def prepare_static_ptq(model: nn.Module, qconfig: QConfig) -> nn.Module:
    model = deepcopy(model).cpu().eval()
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    example_inputs = (torch.randn(1, 3, 224, 224),)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs=example_inputs)
    return prepared


def calibrate(prepared_model: nn.Module, calibration_loader, max_batches: int = 100) -> CalibrationStats:
    prepared_model.eval()
    observed_batches = 0
    observed_samples = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for i, (images, _) in enumerate(calibration_loader):
            prepared_model(images.cpu())
            observed_batches += 1
            observed_samples += images.size(0)
            if i + 1 >= max_batches:
                break
    return CalibrationStats(
        requested_batches=max_batches,
        observed_batches=observed_batches,
        observed_samples=observed_samples,
        elapsed_seconds=time.perf_counter() - start,
    )


def convert_static_ptq(prepared_model: nn.Module) -> nn.Module:
    quantized = convert_fx(prepared_model)
    quantized.eval()
    return quantized
