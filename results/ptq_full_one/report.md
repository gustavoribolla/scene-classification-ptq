# PTQ Places365 Report

## Baseline FP32
- Mode: full
- Dataset: torchvision-places365:/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data:val
- Samples: 36500
- Top-1: 0.5461
- Top-5: 0.8495
- Avg latency (ms/img): 67.1535
- Model size (MB): 92.83

## PTQ Runs
- weight_mode=per_channel, calib_batches=100, calib_samples=6400, top1=0.5455, top5=0.8491, loss_top1=0.06 pp, loss_top5=0.04 pp, size_ratio=3.92x, speedup=3.15x

## Preliminary Accuracy-Loss Analysis
- Best Top-1 preservation: per_channel with 100 calibration batches (6400 samples).
- Top-1 loss: 0.06 percentage points; Top-5 loss: 0.04 percentage points.
- Compression: 3.92x smaller; latency speedup: 3.15x.
