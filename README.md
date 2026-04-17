# PTQ Places365 Bootstrap

Bootstrap project to evaluate FP32 vs static INT8 PTQ for `ResNet50-Places365` on CPU.

## 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Dataset layout

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

## 3) Run baseline FP32

Smoke run:

```bash
python -m src.experiments.run_baseline --smoke
```

Normal run:

```bash
python -m src.experiments.run_baseline
```

## 4) Run PTQ grid

Smoke run:

```bash
python -m src.experiments.run_ptq_grid --smoke
```

Normal run:

```bash
python -m src.experiments.run_ptq_grid
```

## 5) Outputs

All artifacts are saved under `results/`:
- `baseline_fp32.json`
- `ptq_grid.json`
- `report.md`

## Notes

- This bootstrap uses static PTQ from `torch.ao.quantization`.
- Quantization backend defaults to `fbgemm` for x86 CPUs and falls back to `qnnpack`.
- In `--smoke` mode, if `PLACES365` folders are missing, scripts automatically use synthetic data to validate the pipeline.
- The script supports either:
  - torchvision pretrained weights (`--weights-source torchvision`)
  - a local Places365 checkpoint (`--weights-source local --local-weights /path/to/checkpoint.pth`)
