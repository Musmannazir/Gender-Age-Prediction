# Gender Classification and Age Prediction using VGG16 (Multi-Task Learning)

## 1) Cover Page

**Course Title:** ____________________________  
**Lab Title:** Gender Classification and Age Prediction using CNN + Flask  
**Student Name:** ____________________________  
**Roll Number:** ____________________________  
**Semester/Section:** ____________________________  
**Instructor Name:** ____________________________  
**Department/University:** ____________________________  
**Submission Date:** ____________________________

---

## 2) Model Explanation

### 2.1 Problem Statement
This project implements a multi-task deep learning system for facial analysis with two outputs from the same input image:
- **Task 1:** Gender prediction (Binary Classification: Male/Female)
- **Task 2:** Age prediction (Regression)

A shared visual backbone is used to learn common facial features, while two task-specific heads produce final predictions.

In addition to core prediction, the deployed Flask system now includes an **AI Wellness Recommendation Layer** that converts model outputs into:
- personalized skincare routine suggestions,
- non-clinical diagnostic-style skin profile summary,
- prognostic outlook based on routine consistency.

### 2.2 Dataset
The model is trained on the **FairFace** dataset using:
- `fairface_label_train.csv` (training split)
- `fairface_label_val.csv` (validation split)

The training script reads image paths from CSV and maps:
- Gender labels → numeric values (`Male=1`, `Female=0`)
- Age bins (e.g., `20-29`) → midpoint values (e.g., `25`)

### 2.3 Architecture Overview
Backbone and heads are implemented with transfer learning:
- **Backbone:** Pretrained `VGG16` (ImageNet weights)
- **Pooling:** `AdaptiveAvgPool2d((4,4))`
- **Custom classifier:** `8192 → 1024 → 256`
- **Gender head:** `256 → 1` with **Sigmoid**
- **Age head:** `256 → 1` linear output

### 2.4 Loss Functions and Optimization
This is a weighted multi-task objective:

$$
\mathcal{L}_{total}=\alpha\,\mathcal{L}_{gender}+\beta\,\mathcal{L}_{age}
$$

Where:
- $\mathcal{L}_{gender}$ is Binary Cross Entropy (BCE)
- $\mathcal{L}_{age}$ is Mean Absolute Error (L1)
- Typical values in code: $\alpha=1.0$, $\beta=0.15$

Additional training improvements used:
- Data augmentation for training (flip, rotation, color jitter)
- Learning rate scheduling (`ReduceLROnPlateau`)
- Early stopping based on validation composite score

### 2.5 Why This Design Is Suitable
- Transfer learning reduces training time and improves convergence.
- Shared backbone helps both tasks learn strong facial features.
- Separate heads allow task-specific prediction behavior.
- Weighted loss balances classification and regression objectives.

### 2.6 Full-Fledged Project Recommendation Layer
To evolve this into a complete applied AI product, the interface now includes post-inference intelligence:
- **Diagnostic Insight (Non-Clinical):** summarizes likely skin-profile traits inferred from age group output.
- **Prognostic Insight:** provides expected skin trajectory if recommended routines are followed consistently.
- **Skincare Routines:** structured morning and evening plans personalized by predicted age group.

This transforms the system from “prediction-only” into a more complete user-facing decision-support prototype.

---

## 3) Handwritten Parameter Calculations (Compulsory)

> **Note:** The following formulas and computations should be rewritten by hand in your report notebook/pages and then inserted as images.

### 3.1 General Formulas
For a fully connected (Linear) layer:

$$
\#\text{params}=(in\_features\times out\_features)+out\_features
$$

For Conv2D (reference):

$$
\#\text{params}=(k_h\times k_w\times C_{in}\times C_{out})+C_{out}
$$

### 3.2 New Trainable Layers in This Project
1. **Classifier Layer 1** (`8192 → 1024`)

$$
(8192\times1024)+1024 = 8,389,632
$$

2. **Classifier Layer 2** (`1024 → 256`)

$$
(1024\times256)+256 = 262,400
$$

3. **Gender Head** (`256 → 1`)

$$
(256\times1)+1 = 257
$$

4. **Age Head** (`256 → 1`)

$$
(256\times1)+1 = 257
$$

### 3.3 Manual Total of New Trainable Parameters

$$
8,389,632 + 262,400 + 257 + 257 = 8,652,546
$$

### 3.4 Match With Training Output
From latest `metrics.txt`:
- **Total Parameters:** 23,367,234
- **Trainable Parameters:** 8,652,546
- **Frozen Parameters:** 14,714,688

The manual calculation of trainable layers exactly matches the training output.

> **Insert Handwritten Image Placeholder:**  
> **[Insert Figure H1: Handwritten Layer-wise Parameter Calculations]**

---

## 4) Training Graphs and Performance

### 4.1 Latest Validation Metrics
Based on the latest run:
- **Best Composite Score:** 0.6644
- **Validation Gender Accuracy:** 0.7605 (76.05%)
- **Validation Age MAE:** 9.6147 years

### 4.2 Interpretation
- Gender prediction is reasonably good for a baseline model.
- Age prediction is moderately accurate but still has room for improvement.
- The MAE indicates average absolute age error of about 9.6 years.

### 4.3 Graph Placement
> **[Insert Figure T1: training_curves.png (loss curves)]**

Suggested caption:  
**Figure T1.** Training loss trends across epochs for total, gender, and age losses.

### 4.4 Optional Epoch Table (Fill from Console Logs)
| Epoch | Train Total Loss | Train Gender Loss | Train Age Loss | Val Gender Accuracy | Val Age MAE |
|------:|------------------:|------------------:|---------------:|--------------------:|------------:|
| 1 | ____ | ____ | ____ | ____ | ____ |
| 2 | ____ | ____ | ____ | ____ | ____ |
| 3 | ____ | ____ | ____ | ____ | ____ |

---

## 5) Flask Application Screenshots

The project includes a Flask web interface for single-image inference plus recommendation intelligence.

### 5.1 Screenshots to Insert
1. **Main upload page**
   - **[Insert Figure F1: Flask Home Page]**
2. **Image selected/uploaded**
   - **[Insert Figure F2: Uploaded Test Image]**
3. **Prediction output shown** (Gender, Predicted Age, Male Probability)
   - **[Insert Figure F3: Prediction Result Screen]**
4. **Diagnostic + Prognostic panel shown**
   - **[Insert Figure F4: AI Diagnostic and Prognostic Insights]**
5. **Morning + Evening skincare routine panel shown**
   - **[Insert Figure F5: Personalized Skincare Routine Output]**
6. **Safety disclaimer and confidence context shown**
   - **[Insert Figure F6: Wellness Disclaimer and Confidence Block]**

### 5.2 Suggested Caption Style
- **Figure F1.** Flask interface for image upload and inference request.
- **Figure F2.** Test image selected in the web UI.
- **Figure F3.** Model output showing predicted gender, age, and confidence.
- **Figure F4.** Non-clinical diagnostic and prognostic insight section generated from AI outputs.
- **Figure F5.** Personalized morning/evening skincare routine recommendation panel.
- **Figure F6.** Safety disclaimer indicating cosmetic guidance, not medical diagnosis.

---

## 6) Conclusion

This project successfully demonstrates an end-to-end **AI/ML pipeline**:
- dataset-driven training with FairFace,
- multi-task VGG16 model for gender + age prediction,
- deployment through a Flask web interface,
- and post-prediction recommendation features (diagnostic, prognostic, skincare routines).

### Final Summary
- The model achieved **76.05% validation gender accuracy**.
- The age regression achieved **9.61 years MAE**.
- The trainable parameter design is validated by both manual and program outputs.

### Limitations
- Age labels are bin-based and converted to midpoints, which introduces target noise.
- Domain shift (lighting, camera quality, pose, ethnicity imbalance) can reduce real-world performance.
- The current model is not yet at state-of-the-art age precision.

### Future Work
- Train on full data with longer schedules and larger image size.
- Improve age loss weighting and fine-tuning depth.
- Add detailed evaluation (confusion matrix, per-age-bin MAE).
- Consider stronger backbones (e.g., EfficientNet/ResNet) and test-time augmentation.
- Introduce user profile history and routine adherence tracking.
- Add dermatologist-approved rule sets for higher recommendation reliability.

---

## Appendix (Optional in Final Submission)

### Commands Used
Install dependencies:

```bash
pip install -r requirements.txt
```

Training:

```bash
python train.py
```

Stronger training setup (recommended):

```bash
python train.py --epochs 20 --batch-size 32 --image-size 224 --train-limit 0 --val-limit 0 --unfreeze-last-blocks 2 --age-loss-weight 0.15 --gender-loss-weight 1.0 --early-stopping-patience 5 --lr-patience 2 --output-dir .
```

Run Flask app:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```
