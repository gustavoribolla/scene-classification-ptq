# PTQ Places365 Bootstrap

Bootstrap project to evaluate FP32 vs static INT8 PTQ for `ResNet50-Places365` on CPU.

## 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Model weights

The project uses the **official PyTorch Places365 checkpoint** (`resnet50_places365.pth.tar`)
which must be placed at the project root. This file is automatically used by default.

> **Note:** The `.t7` file (`resnet50_places365.t7`) is a legacy **Torch7 (Lua)** format
> and cannot be loaded by PyTorch. Use the `.pth.tar` version from
> [http://places2.csail.mit.edu/models_places365/](http://places2.csail.mit.edu/models_places365/)

## 3) Dataset layout

Set `PLACES365_ROOT` to your local Places365 directory. Expected structure:

```text
PLACES365_ROOT/
  val/
    class_a/*.jpg
    class_b/*.jpg
  test/
    class_a/*.jpg
    class_b/*.jpg
```

The scripts also accept the official torchvision Places365 directory layout
when `PLACES365_ROOT` points to the dataset root.

You can override split directories with:
- `PLACES365_VAL_DIR`
- `PLACES365_TEST_DIR`

## 4) Run baseline FP32

Smoke run (synthetic data, uses real model weights):

```bash
python3 -m src.experiments.run_baseline --smoke
```

Normal run (requires Places365 dataset):

```bash
python3 -m src.experiments.run_baseline
```

## 5) Run PTQ grid

Smoke run:

```bash
python3 -m src.experiments.run_ptq_grid --smoke
```

Normal run:

```bash
python3 -m src.experiments.run_ptq_grid
```

Fast implementation check:

```bash
python3 -m src.experiments.run_ptq_grid --smoke \
  --smoke-calib-samples 8 \
  --smoke-test-samples 8 \
  --calibration-batches 1 \
  --weight-modes per_tensor \
  --batch-size 4 \
  --num-workers 0 \
  --results-dir /tmp/scene-classification-ptq-smoke
```

## 6) Outputs

All artifacts are saved under `results/`:
- `baseline_fp32.json`
- `ptq_grid.json`: FP32 baseline, INT8 runs, calibration statistics, size ratio, latency speedup, and Top-1/Top-5 loss.
- `report.md`: concise FP32 vs INT8 comparison and preliminary accuracy-loss analysis.

## 7) Tangible INT8 demo

Build one quantized INT8 model and classify real Places365 validation images:

```bash
./run_quantized_demo.sh
```

Outputs are saved under `results/quantized_demo/`:
- `places365_resnet50_int8_torchscript.pt`: saved executable INT8 model artifact.
- `quantized_model_metadata.json`: backend, weight mode, dataset path, and calibration stats.
- `predictions.json`: raw Top-5 predictions for the demo images.
- `demo_report.md`: readable report to show the quantized model classifying environments.

Quick check with less calibration:

```bash
./run_quantized_demo.sh --calibration-batches 10 --batch-size 16 --results-dir /tmp/quantized_demo_check
```

## Notes

- This bootstrap uses static PTQ from `torch.ao.quantization`.
- Quantization backend defaults to `fbgemm` for x86 CPUs and falls back to `qnnpack`.
- In `--smoke` mode, if `PLACES365` folders are missing, scripts automatically use synthetic data to validate the pipeline.
- The default weights source is `local` pointing to `resnet50_places365.pth.tar` (365 output classes).
- You can override with `--weights-source torchvision` to use ImageNet-pretrained weights (1000 classes).
