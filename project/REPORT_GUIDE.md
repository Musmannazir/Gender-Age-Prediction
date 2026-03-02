# Gender Classification & Age Prediction (CNN + Flask) Report Guide

## 1) Cover Page
- Course name, lab title, student name, roll no, semester, instructor, submission date.

## 2) Model Explanation
- Backbone: VGG16 pretrained on ImageNet.
- Frozen layers: all convolutional layers in `features`.
- Modified layers:
  - `avgpool = AdaptiveAvgPool2d((4,4))`
  - `classifier = [8192->1024->256]`
- Two output heads:
  - Gender head: `256 -> 1`, Sigmoid, BCELoss
  - Age head: `256 -> 1`, Linear, L1Loss (MAE)
- Total loss:

$$
\mathcal{L}_{total}=\mathcal{L}_{gender}+\mathcal{L}_{age}
$$

## 3) Handwritten Parameter Calculations (Compulsory)
Use these equations in your handwritten section.

### General equations
For Conv2D:

$$
\#params=(k_h\times k_w\times C_{in}\times C_{out})+C_{out}
$$

For Linear:

$$
\#params=(in\_features\times out\_features)+out\_features
$$

For BatchNorm (if present):

$$
\#params=2\times C
$$

### New trainable layers in this project
1. Classifier layer 1: `8192 -> 1024`

$$
(8192\times1024)+1024=8,389,632
$$

2. Classifier layer 2: `1024 -> 256`

$$
(1024\times256)+256=262,400
$$

3. Gender head: `256 -> 1`

$$
(256\times1)+1=257
$$

4. Age head: `256 -> 1`

$$
(256\times1)+1=257
$$

Total new trainable parameters:

$$
8,389,632+262,400+257+257=8,652,546
$$

### Pretrained frozen VGG16 parameters
Standard VGG16 total parameters:

$$
138,357,544
$$

In this project, we replace original VGG16 classifier with a new classifier, so total parameters differ from full original VGG16.
Use printed output from training:
- `Total parameters`
- `Trainable parameters`
- `Frozen parameters`

Then compare with your manual calculation in a table.

## 4) Training Results
Attach:
- Epoch-wise losses
- Final validation gender accuracy
- Final validation age MAE
- `training_curves.png`
- `metrics.txt`

## 5) Flask Screenshots
Include screenshots of:
- Web form
- Uploaded test image
- Predicted gender and age result

## 6) Conclusion
- Performance summary (gender accuracy + age MAE)
- Limitations (age bin conversion, bias, domain shift)
- Future work (data balancing, augmentation, better multi-task loss weighting)

---

## Appendix: Required Commands
Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py --epochs 5 --batch-size 32 --output-dir .
```

Default dataset path used by the project:

```text
C:\Users\nazir\.cache\kagglehub\datasets\mehmoodsheikh\fairface-dataset\versions\1
```

Run Flask app:

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```
