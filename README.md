# PTQ Places365 Bootstrap

Bootstrap project to evaluate FP32 vs static INT8 Post-Training Quantization (PTQ) for `ResNet50-Places365` on CPU.

The project supports:

- FP32 baseline evaluation
- Static INT8 PTQ
- Comparison between `per_tensor` and `per_channel` weight quantization
- Experiments with different calibration sizes
- Structured result export in JSON, CSV and Markdown
- Automatic plots for Sprint 4 analysis
- Real-image INT8 inference demo

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

The project uses the **official PyTorch Places365 checkpoint**:

```text
resnet50_places365.pth.tar
```

This file must be placed at the project root.

> **Note:** The `.t7` file is a legacy Torch7 format and is not supported by PyTorch.

Download:

[http://places2.csail.mit.edu/models_places365/](http://places2.csail.mit.edu/models_places365/)

---

## 3) Dataset layout

Set `PLACES365_ROOT` to your local dataset directory.

Expected structure:

```text
PLACES365_ROOT/
  val/
    class_a/*.jpg
    class_b/*.jpg
    ...
  test/
    class_a/*.jpg
    class_b/*.jpg
    ...
```

On Windows PowerShell:

```bash
$env:PLACES365_ROOT="C:\path\to\places365_data"
```

On Linux/Mac:

```bash
export PLACES365_ROOT="/path/to/places365_data"
```

Environment variables supported by the project:

* `PLACES365_ROOT`
* `PLACES365_VAL_DIR`
* `PLACES365_TEST_DIR`

If Places365 is not available, use `--smoke` mode to run a quick validation with synthetic data.

### External validation images from QuintoAndar

To collect an external real-estate validation set with labeled room photos:

```bash
python scripts/scrape_quintoandar_images.py \
  --output-dir data/external/quintoandar \
  --min-area 100 \
  --max-photos 1000
```

The scraper reads public QuintoAndar listing pages, keeps only listings that are furnished and larger than 100 m², then downloads photos whose QuintoAndar subtitle maps to an environment label such as `sala`, `quarto`, `cozinha`, `banheiro`, `varanda`, or `area_servico`.

Outputs:

```text
data/external/quintoandar/
  images/<label>/*.jpg
  manifest.csv
  manifest.jsonl
  summary.json
```

Useful options:

```bash
python scripts/scrape_quintoandar_images.py --dry-run --max-photos 50
python scripts/scrape_quintoandar_images.py --image-size xxl --overwrite
python scripts/scrape_quintoandar_images.py --search-url "https://www.quintoandar.com.br/alugar/imovel/sao-paulo-sp-brasil/mobiliado"
python scripts/scrape_quintoandar_images.py --search-url URL_1 --search-url URL_2
```

Evaluate those images with the quantized Places365 model:

```bash
python -m src.experiments.evaluate_quintoandar_quantized
```

This writes mapped-label metrics and per-image predictions to:

```text
results/quintoandar_quantized_eval/
  metrics.json
  predictions.csv
  predictions.jsonl
  report.md
```

The evaluator maps QuintoAndar room labels to compatible Places365 categories before scoring. For example, `sala` accepts `living_room`, `dining_room`, and `television_room`; `quarto` accepts `bedroom`, `bedchamber`, and `dorm_room`; `area_servico` accepts `utility_room`, `storage_room`, and `laundromat`.

The downloaded image files and generated Word summaries are local artifacts and are ignored by git. The lightweight manifests and evaluation result files can be versioned for reproducibility.

---

## 4) Run baseline FP32

The FP32 baseline is used as the reference for accuracy, latency and model size.

Smoke test:

```bash
python -m src.experiments.run_baseline --smoke --num-workers 0
```

Full evaluation:

```bash
python -m src.experiments.run_baseline --num-workers 0
```

This generates:

```text
results/baseline_fp32.json
```

---

## 5) Run PTQ grid

The PTQ grid runs static INT8 quantization experiments and compares different calibration sizes and weight quantization modes.

### Smoke test

Use this to check if the pipeline is working:

```bash
python -m src.experiments.run_ptq_grid --smoke \
  --calibration-batches 1 2 \
  --weight-modes per_tensor per_channel \
  --batch-size 16 \
  --num-workers 0
```

### Full Sprint 4 experiment

Use this for the Sprint 4 results:

```bash
python -m src.experiments.run_ptq_grid \
  --calibration-batches 1 5 10 25 50 100 \
  --weight-modes per_tensor per_channel \
  --batch-size 32 \
  --num-workers 0
```

On Windows PowerShell, you can also run it in one line:

```bash
python -m src.experiments.run_ptq_grid --calibration-batches 1 5 10 25 50 100 --weight-modes per_tensor per_channel --batch-size 32 --num-workers 0
```

This tests:

```text
per_tensor  + 1, 5, 10, 25, 50, 100 calibration batches
per_channel + 1, 5, 10, 25, 50, 100 calibration batches
```

---

## 6) Generate Sprint 4 plots

After running the PTQ grid, generate the comparison plots:

```bash
python -m src.experiments.plot_ptq_grid
```

The script reads:

```text
results/ptq_grid.json
```

and saves plots into:

```text
results/
```

Generated plots:

```text
ptq_top1_vs_calibration.png
ptq_top5_vs_calibration.png
ptq_latency_vs_calibration.png
ptq_speedup_vs_calibration.png
ptq_accuracy_loss_vs_calibration.png
```

---

## 7) Sprint 4 outputs

Saved in `results/`:

```text
baseline_fp32.json
ptq_grid.json
ptq_grid.csv
report.md
ptq_top1_vs_calibration.png
ptq_top5_vs_calibration.png
ptq_latency_vs_calibration.png
ptq_speedup_vs_calibration.png
ptq_accuracy_loss_vs_calibration.png
```

### Main files

* `baseline_fp32.json`
  FP32 reference metrics.

* `ptq_grid.json`
  Complete structured PTQ experiment results.

* `ptq_grid.csv`
  Table-ready version of the PTQ results.

* `report.md`
  Automatic Markdown report with baseline, PTQ table and preliminary analysis.

* `ptq_*.png`
  Graphs comparing calibration size, accuracy, latency and speedup.

---

## 8) Tangible INT8 demo: real inference

Build a quantized INT8 model and classify a real image.

### Run demo with a test image

```bash
python -m src.experiments.run_quantized_demo --image-path assets/test1.webp --rebuild
```

### Input image

Place your test image in:

```text
assets/test1.webp
```

You can use any indoor image, such as kitchen, bedroom, office or living room.

---

## 9) Demo outputs

Saved in:

```text
results/quantized_demo/
```

Files:

```text
places365_resnet50_int8_torchscript.pt
quantized_model_metadata.json
predictions.json
demo_report.md
```

### Example output

```text
Top-1: kitchen (80.48%)
Top-5: kitchen, galley, restaurant kitchen, wet bar, utility room
```

---

## 10) Notes

* Static PTQ uses `torch.ao.quantization`.
* Quantization backend is automatically selected depending on the environment:

  * `fbgemm` for x86 CPUs
  * `qnnpack` for ARM or some builds
  * `onednn` for Intel / Windows builds
* In `--smoke` mode, synthetic data is used if Places365 is not available.
* Default weights file:

```text
resnet50_places365.pth.tar
```

Optional model source:

```bash
--weights-source torchvision
```

---

## 11) Tips

* Use `--smoke` first to check if the pipeline works.
* Use `--num-workers 0` on Windows to avoid multiprocessing issues.
* Run the FP32 baseline before the full PTQ grid.
* Use clear indoor images for the real inference demo.
* Always use `--rebuild` when changing the quantization backend or calibration setup.
* The CSV and plots are the most useful files for the Sprint 4 report.
