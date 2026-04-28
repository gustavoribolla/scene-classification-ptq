# Quantized Places365 Demo

## Artifact
- Quantized model: `results/quantized_demo/places365_resnet50_int8_torchscript.pt`
- Backend: `qnnpack`
- Weight mode: `per_channel`
- Calibration batches: `100`
- Calibration samples: `3200`

## Predictions

### Places365_val_00000001.jpg
- Image: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data/val_256/Places365_val_00000001.jpg`
- True label: `g / greenhouse / indoor`
- Top-1 prediction: `b / boardwalk` (43.93%)
- Top predictions:
  - `b / boardwalk`: 43.93%
  - `r / rope bridge`: 24.72%
  - `r / rainforest`: 8.78%
  - `b / botanical garden`: 6.22%
  - `p / park`: 2.78%

### Places365_val_00000002.jpg
- Image: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data/val_256/Places365_val_00000002.jpg`
- True label: `w / wet bar`
- Top-1 prediction: `k / kitchen` (48.34%)
- Top predictions:
  - `k / kitchen`: 48.34%
  - `w / wet bar`: 24.24%
  - `p / pantry`: 10.84%
  - `g / galley`: 8.61%
  - `r / restaurant kitchen`: 3.85%

### Places365_val_00000003.jpg
- Image: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data/val_256/Places365_val_00000003.jpg`
- True label: `c / clean room`
- Top-1 prediction: `o / operating room` (65.17%)
- Top predictions:
  - `o / operating room`: 65.17%
  - `c / clean room`: 25.97%
  - `h / hospital room`: 4.13%
  - `v / veterinarians office`: 2.60%
  - `b / biology laboratory`: 0.93%

### Places365_val_00000004.jpg
- Image: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data/val_256/Places365_val_00000004.jpg`
- True label: `g / golf course`
- Top-1 prediction: `g / golf course` (31.04%)
- Top predictions:
  - `g / golf course`: 31.04%
  - `s / soccer field`: 17.47%
  - `f / football field`: 9.83%
  - `p / park`: 9.83%
  - `p / playground`: 3.49%

### Places365_val_00000005.jpg
- Image: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data/val_256/Places365_val_00000005.jpg`
- True label: `r / rock arch`
- Top-1 prediction: `r / rock arch` (69.25%)
- Top predictions:
  - `r / rock arch`: 69.25%
  - `c / cliff`: 17.42%
  - `i / islet`: 3.10%
  - `a / arch`: 2.77%
  - `c / coast`: 2.77%

### Places365_val_00000006.jpg
- Image: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data/val_256/Places365_val_00000006.jpg`
- True label: `c / corridor`
- Top-1 prediction: `c / corridor` (69.11%)
- Top predictions:
  - `c / corridor`: 69.11%
  - `e / elevator lobby`: 15.50%
  - `a / airport terminal`: 4.37%
  - `o / office cubicles`: 2.19%
  - `l / lobby`: 1.39%

### Places365_val_00000007.jpg
- Image: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data/val_256/Places365_val_00000007.jpg`
- True label: `c / canyon`
- Top-1 prediction: `r / rock arch` (40.50%)
- Top predictions:
  - `r / rock arch`: 40.50%
  - `c / cliff`: 12.83%
  - `c / canyon`: 7.22%
  - `r / ruin`: 4.06%
  - `c / coast`: 3.23%

### Places365_val_00000008.jpg
- Image: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data/val_256/Places365_val_00000008.jpg`
- True label: `d / dining room`
- Top-1 prediction: `d / dining room` (79.39%)
- Top predictions:
  - `d / dining room`: 79.39%
  - `d / dining hall`: 17.81%
  - `b / banquet hall`: 1.13%
  - `l / living room`: 0.45%
  - `w / wet bar`: 0.25%

## Demo Sample Accuracy
- Top-1 on shown samples: 50.00%
- Top-5 on shown samples: 87.50%
