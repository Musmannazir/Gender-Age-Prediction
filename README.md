# Gender Classification & Age Prediction using VGG16 + Flask

This project implements a multi-task CNN model that predicts:
- Gender (Male/Female) as binary classification
- Age (0-80) as regression

The model uses pretrained VGG16, freezes pretrained feature layers, and adds custom classifier + two task heads.

## Project Structure

```text
project/
├── app.py
├── model.pth
├── model_def.py
├── train.py
├── requirements.txt
└── templates/
    └── index.html
```

## 1) Setup

```bash
pip install -r requirements.txt
```

## 2) Dataset
Use FairFace dataset from Kaggle:
https://www.kaggle.com/datasets/mehmoodsheikh/fairface-dataset

Place dataset so CSV files and image paths are reachable by `--data-root`.

## 3) Train

```bash
python train.py --epochs 10 --batch-size 32 --output-dir .
```

`python train.py` now uses a fast default setup for lab iteration:
- epochs: 3
- batch size: 16
- image size: 128
- train limit: 8000
- val limit: 2000
- unfreeze last blocks: 1
- augmentation: enabled

For a much faster lab run (subset training):

```bash
python train.py --epochs 1 --batch-size 16 --image-size 128 --num-workers 2 --train-limit 8000 --val-limit 2000 --output-dir .
```

To run full FairFace data (slow):

```bash
python train.py --epochs 5 --batch-size 32 --image-size 224 --train-limit 0 --val-limit 0 --output-dir .
```

For a better-quality model (recommended):

```bash
python train.py --epochs 20 --batch-size 32 --image-size 224 --train-limit 0 --val-limit 0 --unfreeze-last-blocks 2 --age-loss-weight 0.15 --gender-loss-weight 1.0 --early-stopping-patience 5 --lr-patience 2 --output-dir .
```

Default dataset path is already set in `train.py` to:

```text
project/FairFace
```

If needed, you can still override it:

```bash
python train.py --data-root "YOUR_PATH" --epochs 10 --batch-size 32 --output-dir .
```

Outputs:
- `model.pth`
- `training_curves.png`
- `metrics.txt`

## 4) Flask Inference
After training, keep `model.pth` inside `project/`, then run:

```bash
python app.py
```

Open:
- http://127.0.0.1:5000

Upload image and view predicted gender + age.

## Notes for Lab Requirements
- Backbone: pretrained VGG16
- Pretrained layers frozen
- Modified `avgpool` and `classifier`
- Two outputs (gender + age)
- Losses: `BCELoss + L1Loss`

## Recommendation to make this full-fledged project
- Add REST API endpoint (`/predict`) returning JSON
- Add confidence calibration and uncertainty for age
- Add TensorBoard logging
- Add Docker deployment and CI/CD for reproducible deployment
