# PTQ Places365 Bootstrap

Bootstrap project to evaluate FP32 vs static INT8 PTQ for `ResNet50-Places365` on CPU.

---

## 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt
````

---

## 2) Model weights

The project uses the **official PyTorch Places365 checkpoint** (`resnet50_places365.pth.tar`)
which must be placed at the project root.

> **Note:** The `.t7` file is a legacy Torch7 format and is not supported by PyTorch.

Download:
[http://places2.csail.mit.edu/models_places365/](http://places2.csail.mit.edu/models_places365/)

---

## 3) Dataset layout

Set `PLACES365_ROOT` to your local dataset directory:

```text
PLACES365_ROOT/
  val/
    class_a/*.jpg
    class_b/*.jpg
  test/
    class_a/*.jpg
    class_b/*.jpg
```

Environment variables (optional):

* `PLACES365_VAL_DIR`
* `PLACES365_TEST_DIR`

---

## 4) Run baseline FP32

Smoke test:

```bash
python -m src.experiments.run_baseline --smoke
```

Full evaluation:

```bash
python -m src.experiments.run_baseline
```

---

## 5) Run PTQ grid

Smoke:

```bash
python -m src.experiments.run_ptq_grid --smoke
```

Full:

```bash
python -m src.experiments.run_ptq_grid
```

Fast check:

```bash
python -m src.experiments.run_ptq_grid --smoke \
  --smoke-calib-samples 8 \
  --smoke-test-samples 8 \
  --calibration-batches 1 \
  --weight-modes per_tensor \
  --batch-size 4 \
  --num-workers 0 \
  --results-dir /tmp/scene-classification-ptq-smoke
```

---

## 6) Outputs

Saved in `results/`:

* `baseline_fp32.json`
* `ptq_grid.json`
* `calibration_curve.png`
* `baseline_confusion_matrix.png`

---

## 7) Tangible INT8 demo (real inference)

Build a quantized INT8 model and classify real images.

### Run demo with a test image

```bash
python -m src.experiments.run_quantized_demo --image-path assets/test1.webp --rebuild
```

### Input image

Place your test image in:

```text
assets/test1.webp
```

You can use any image (kitchen, bedroom, office, etc).

---

## Outputs (important)

Saved in:

```text
results/quantized_demo/
```

Files:

* `places365_resnet50_int8_torchscript.pt` → quantized model
* `quantized_model_metadata.json` → calibration + backend info
* `predictions.json` → raw predictions
* `demo_report.md` → formatted readable output

---

## Example output

```text
Top-1: kitchen (80.48%)
Top-5: kitchen, galley, restaurant kitchen, wet bar, utility room
```

---

## 8) Notes

* Static PTQ using `torch.ao.quantization`

* Backend is **automatically selected** depending on environment:

  * `fbgemm` (x86 CPUs)
  * `qnnpack` (ARM / some builds)
  * `onednn` (Intel / Windows builds)

* In `--smoke` mode, synthetic data is used if Places365 is not available

* Default weights: `resnet50_places365.pth.tar`

* Optional:

  ```bash
  --weights-source torchvision
  ```

---

## 9) Tips

* Use clear images (e.g., kitchen) for better predictions
* If results look wrong, try another image
* Always use `--rebuild` when changing quantization backend
