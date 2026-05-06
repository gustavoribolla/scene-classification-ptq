# PTQ Places365 Sprint 4 Report

This report summarizes the Sprint 4 experiments for static Post-Training Quantization of ResNet50-Places365.
The experiments compare per-tensor and per-channel INT8 quantization under different calibration sizes.

## Baseline FP32

- Mode: full
- Dataset: torchvision-places365:/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq/places365_data:val
- Samples: 36500
- Top-1 Accuracy: 0.5461
- Top-5 Accuracy: 0.8495
- Average latency: 68.6605 ms/image
- Model size: 92.83 MB

## Structured PTQ Results

| Weight mode | Calibration batches | Calibration samples | Top-1 | Top-5 | Top-1 loss (pp) | Top-5 loss (pp) | Size (MB) | Latency (ms/img) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per_tensor | 1 | 32 | 0.5439 | 0.8490 | 0.21 | 0.05 | 23.26 | 21.33 | 3.22x |
| per_tensor | 5 | 160 | 0.5436 | 0.8496 | 0.25 | -0.01 | 23.26 | 21.14 | 3.25x |
| per_tensor | 10 | 320 | 0.5440 | 0.8488 | 0.21 | 0.06 | 23.26 | 21.07 | 3.26x |
| per_tensor | 25 | 800 | 0.5440 | 0.8487 | 0.21 | 0.08 | 23.26 | 21.86 | 3.14x |
| per_tensor | 50 | 1600 | 0.5441 | 0.8488 | 0.20 | 0.07 | 23.26 | 20.76 | 3.31x |
| per_tensor | 100 | 3200 | 0.5436 | 0.8494 | 0.25 | 0.01 | 23.26 | 20.97 | 3.27x |
| per_channel | 1 | 32 | 0.5454 | 0.8493 | 0.07 | 0.02 | 23.70 | 21.48 | 3.20x |
| per_channel | 5 | 160 | 0.5453 | 0.8490 | 0.08 | 0.05 | 23.70 | 21.12 | 3.25x |
| per_channel | 10 | 320 | 0.5454 | 0.8490 | 0.07 | 0.04 | 23.70 | 21.07 | 3.26x |
| per_channel | 25 | 800 | 0.5455 | 0.8488 | 0.06 | 0.07 | 23.70 | 21.75 | 3.16x |
| per_channel | 50 | 1600 | 0.5452 | 0.8485 | 0.09 | 0.09 | 23.70 | 20.52 | 3.35x |
| per_channel | 100 | 3200 | 0.5450 | 0.8492 | 0.11 | 0.03 | 23.70 | 20.55 | 3.34x |

## Preliminary Analysis

The configuration with the best Top-1 preservation was `per_channel` using 25 calibration batches (800 samples). It achieved Top-1 accuracy of 0.5455, with a Top-1 loss of 0.06 percentage points.

The fastest configuration was `per_channel` using 50 calibration batches. It achieved a speedup of 3.35x over FP32, with latency of 20.52 ms/image.

Overall, these experiments allow a direct comparison between per-tensor and per-channel quantization, showing how calibration size affects accuracy, latency, model size, and the trade-off between efficiency and predictive performance.
