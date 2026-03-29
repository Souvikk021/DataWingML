# DataWing — Dysgraphia Prediction via Handwriting Analysis

---

## Project Structure

```
DataWing/
├── app.py
├── train.py
├── model.pkl
├── model_report.txt
├── requirements.txt
├── dataset/
│   ├── dysgraphic/
│   └── normal/
├── static/
│   ├── uploads/
│   ├── processed/
│   └── charts/
└── templates/
    ├── index.html
    ├── result.html
    └── dashboard.html
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset

Mendeley Dysgraphia Dataset: https://data.mendeley.com/datasets/39hr8dx76p/1

Place images into:
```
dataset/dysgraphic/
dataset/normal/
```
Minimum 20 per class. 50+ recommended.

---

## Usage

```bash
python train.py
python app.py
```

| URL | Description |
|-----|-------------|
| `/` | Upload image, get prediction |
| `/dashboard` | Analysis dashboard |

---

## Features Extracted

| Feature | Description |
|---------|-------------|
| Mean / Std Letter Height | Letter size consistency |
| Mean / Std Letter Width | Width variation |
| Letter Height CV | Coefficient of variation |
| Proportion Consistency Std | Height-to-width ratio consistency |
| Letter Spacing | 30th percentile gap between components |
| Word Spacing | 80th percentile gap |
| Total Stroke Length | Skeleton-based stroke measure |
| Corner Count / Density | Direction changes per ink area |
| Num Components / Density | Writing fragmentation |
| Total Ink Area | Ink coverage |
| Baseline Regularity Std | Letter alignment to baseline |
| Vertical Regularity Std | Vertical consistency |
| Margin Alignment Std | Left margin consistency |
| Slant Angle | Character slant |

---

## Models

| Model | Type |
|-------|------|
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| SVM RBF Kernel | Kernel-based |
| K-Nearest Neighbours | Instance-based |
| Logistic Regression | Linear |

Best model selected via 5-fold cross-validation.

---

## Charts Generated

| Chart |
|-------|
| Feature Correlation Heatmap |
| SelectKBest ANOVA F-Score |
| Random Forest Feature Importance |
| Model Comparison (Accuracy, F1, Precision, Recall) |
| Confusion Matrices |
| ROC Curves with AUC |
| Cross-Validation Score Distribution |
| Class Distribution (before / after balancing) |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| Image Processing | OpenCV, scikit-image |
| ML | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Data | NumPy, Pandas, SciPy |
| PDF Support | PyMuPDF |
| Frontend | HTML, Bootstrap 5 |
