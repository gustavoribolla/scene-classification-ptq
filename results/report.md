# PTQ Places365 Report

## Baseline FP32
- Top-1: 0.0039
- Top-5: 0.0078
- Avg latency (ms/img): 82.3072
- Model size (MB): 97.80

## PTQ Runs
- weight_mode=per_tensor, calib_batches=10, top1=0.0000, top5=0.0000, delta_top1=0.0039, size_ratio=3.99x, speedup=3.35x
- weight_mode=per_tensor, calib_batches=50, top1=0.0000, top5=0.0000, delta_top1=0.0039, size_ratio=3.99x, speedup=3.32x
- weight_mode=per_tensor, calib_batches=100, top1=0.0000, top5=0.0000, delta_top1=0.0039, size_ratio=3.99x, speedup=3.26x
- weight_mode=per_tensor, calib_batches=500, top1=0.0000, top5=0.0000, delta_top1=0.0039, size_ratio=3.99x, speedup=2.73x
- weight_mode=per_channel, calib_batches=10, top1=0.0000, top5=0.0000, delta_top1=0.0039, size_ratio=3.92x, speedup=3.10x
- weight_mode=per_channel, calib_batches=50, top1=0.0000, top5=0.0000, delta_top1=0.0039, size_ratio=3.92x, speedup=3.01x
- weight_mode=per_channel, calib_batches=100, top1=0.0000, top5=0.0000, delta_top1=0.0039, size_ratio=3.92x, speedup=3.03x
- weight_mode=per_channel, calib_batches=500, top1=0.0000, top5=0.0000, delta_top1=0.0039, size_ratio=3.92x, speedup=3.15x
