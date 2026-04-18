# PTQ Places365 Report

## Baseline FP32
- Top-1: 0.0020
- Top-5: 0.0117
- Avg latency (ms/img): 75.0139
- Model size (MB): 92.83

## PTQ Runs
- weight_mode=per_tensor, calib_batches=10, top1=0.0020, top5=0.0117, delta_top1=0.0000, size_ratio=3.99x, speedup=3.13x
- weight_mode=per_tensor, calib_batches=50, top1=0.0020, top5=0.0117, delta_top1=0.0000, size_ratio=3.99x, speedup=2.86x
- weight_mode=per_tensor, calib_batches=100, top1=0.0020, top5=0.0117, delta_top1=0.0000, size_ratio=3.99x, speedup=2.68x
- weight_mode=per_tensor, calib_batches=500, top1=0.0020, top5=0.0117, delta_top1=0.0000, size_ratio=3.99x, speedup=2.70x
- weight_mode=per_channel, calib_batches=10, top1=0.0020, top5=0.0117, delta_top1=0.0000, size_ratio=3.92x, speedup=2.67x
- weight_mode=per_channel, calib_batches=50, top1=0.0020, top5=0.0117, delta_top1=0.0000, size_ratio=3.92x, speedup=2.70x
- weight_mode=per_channel, calib_batches=100, top1=0.0020, top5=0.0117, delta_top1=0.0000, size_ratio=3.92x, speedup=2.27x
- weight_mode=per_channel, calib_batches=500, top1=0.0020, top5=0.0117, delta_top1=0.0000, size_ratio=3.92x, speedup=2.27x
