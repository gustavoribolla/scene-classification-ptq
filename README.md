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

You can override split directories with:
- `PLACES365_VAL_DIR`
- `PLACES365_TEST_DIR`

## 4) Run baseline FP32

Smoke run (synthetic data, uses real model weights):

```bash
python -m src.experiments.run_baseline --smoke
```

Normal run (requires Places365 dataset):

```bash
python -m src.experiments.run_baseline
```

## 5) Run PTQ grid

Smoke run:

```bash
python -m src.experiments.run_ptq_grid --smoke
```

Normal run:

```bash
python -m src.experiments.run_ptq_grid
```

## 6) Outputs

All artifacts are saved under `results/`:
- `baseline_fp32.json`
- `ptq_grid.json`
- `report.md`

## Notes

- This bootstrap uses static PTQ from `torch.ao.quantization`.
- Quantization backend defaults to `fbgemm` for x86 CPUs and falls back to `qnnpack`.
- In `--smoke` mode, if `PLACES365` folders are missing, scripts automatically use synthetic data to validate the pipeline.
- The default weights source is `local` pointing to `resnet50_places365.pth.tar` (365 output classes).
- You can override with `--weights-source torchvision` to use ImageNet-pretrained weights (1000 classes).

