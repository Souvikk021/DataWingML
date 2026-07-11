# 🪶 DataWing — Multi-Model Dysgraphia Detection via Handwriting Analysis

> An end-to-end machine learning web application that screens handwriting samples for **dysgraphia** using classical computer vision feature extraction and an ensemble of ML classifiers.

---

## 📌 Overview

DataWing ingests handwriting images or PDFs, extracts 18 biomechanical and spatial features from the writing, and runs them through a multi-model ML pipeline. It supports:

- **Single-image analysis** with per-sample predictions
- **Multi-dataset training** across three independent handwriting corpora
- **Cross-dataset generalisation testing** (train on one, test on another)
- **Rule-based screening** as a transparent interpretable baseline
- **Interactive dashboard** with 8+ diagnostic charts

---

## 🗂️ Project Structure

```
DataWing/
├── app.py                    # Flask web application & feature extraction pipeline
├── train.py                  # Single-dataset ML training script
├── multi_dataset_train.py    # Multi-dataset training, pooling & cross-testing
├── model.pkl                 # Serialised best model (auto-selected by CV accuracy)
├── model_report.txt          # Human-readable model performance report
├── requirements.txt          # Python dependencies
│
├── dataset/                  # Handwriting image datasets (not tracked in git)
│   ├── Gambo/
│   │   ├── Train/
│   │   │   ├── Normal/
│   │   │   ├── Corrected/
│   │   │   └── Reversal/
│   │   └── Test/
│   │       ├── Normal/
│   │       ├── Corrected/
│   │       └── Reversal/
│   ├── dysgraphia_patients/
│   │   ├── high potential/
│   │   ├── low potential/
│   │   └── corrected/
│   └── mendeley/
│       ├── normal/
│       └── dysgraphic/
│
├── static/
│   ├── uploads/              # Raw uploaded images
│   ├── processed/            # Grayscale, threshold & skeleton visualisations
│   └── charts/               # Training & analysis charts
│       └── multi_dataset/    # Cross-dataset analysis charts & CSVs
│
└── templates/
    ├── index.html            # Upload page
    ├── result.html           # Per-sample prediction results
    ├── dashboard.html        # Single-dataset training dashboard
    └── multi_analysis.html   # Multi-dataset analysis dashboard
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Souvikk021/DataWingML.git
cd DataWingML
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

Dataset files are **not tracked in this repository** (too large). Download and place them manually:

| Dataset | Source | Place in |
|---|---|---|
| Mendeley Dysgraphia | [data.mendeley.com/datasets/39hr8dx76p/1](https://data.mendeley.com/datasets/39hr8dx76p/1) | `dataset/mendeley/normal/` and `dataset/mendeley/dysgraphic/` |
| Gambo Dataset | Sourced separately | `dataset/Gambo/Train/` and `dataset/Gambo/Test/` |
| Dysgraphia Patients | Sourced separately | `dataset/dysgraphia_patients/{high potential, low potential, corrected}/` |

> Minimum **20 images per class** required. 50+ per class recommended for reliable model training.

---

## 🚀 Usage

### Step 1 — Train a model

**Single dataset (Mendeley):**
```bash
python train.py
```

**Multi-dataset (all three datasets + pooled + cross-testing):**
```bash
python multi_dataset_train.py
```

Both scripts save `model.pkl` and generate charts into `static/charts/`.

### Step 2 — Run the web app
```bash
python app.py
```

The server starts at **http://localhost:8501**

| Route | Description |
|---|---|
| `/` | Upload handwriting image/PDF, receive prediction |
| `/dashboard` | Single-dataset model comparison & diagnostic charts |
| `/multi-analysis` | Multi-dataset results, cross-dataset matrix, ANOVA feature shift |

---

## 🧠 How It Works

### Feature Extraction Pipeline

Each uploaded image goes through:

1. **Ruled-line detection & removal** — horizontal line suppression via morphological ops
2. **Binarisation** — Gaussian blur → Otsu thresholding
3. **Connected component analysis** — isolates individual letter components
4. **Skeletonization** — extracts stroke skeleton for length measurement
5. **Feature computation** — 18 quantitative features derived

| Feature | Description |
|---|---|
| Mean / Std Letter Height | Size consistency across the sample |
| Mean / Std Letter Width | Width variation |
| Letter Height CV | Coefficient of variation in letter height |
| Proportion Consistency Std | Height-to-width ratio variability |
| Letter Spacing | 30th percentile gap between components |
| Word Spacing | 80th percentile gap between components |
| Total Stroke Length | Skeleton-based ink path length |
| Corner Count / Density | Direction changes per ink area |
| Num Components / Density | Writing fragmentation measure |
| Total Ink Area | Total ink pixel coverage |
| Baseline Regularity Std | Vertical deviation from baseline |
| Vertical Regularity Std | Height consistency across letters |
| Margin Alignment Std | Left margin consistency |
| Slant Angle | Character forward/backward lean |

### Dual Prediction System

Every sample receives two independent predictions:

- **Rule-based screening** — threshold logic across 6 features; flags as *Dysgraphic* if ≥ 3 are triggered
- **ML model prediction** — trained classifier with confidence score (%)

---

## 🤖 Models Trained

| Model | Type |
|---|---|
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |
| SVM (RBF Kernel) | Kernel-based |
| K-Nearest Neighbours | Instance-based |
| Logistic Regression | Linear baseline |

**Model selection:** 5-fold stratified cross-validation. Best CV accuracy model is saved as `model.pkl`.

**SMOTE** is applied during training to handle class imbalance.

---

## 📊 Charts Generated

### Single-Dataset Dashboard (`/dashboard`)
| Chart | Description |
|---|---|
| Model Comparison | Accuracy, F1, Precision, Recall across all models |
| ROC Curves with AUC | Per-model ROC curves |
| Confusion Matrices | Grid of confusion matrices |
| CV Score Distribution | Box plots of cross-validation folds |
| Feature Importance (RF) | Random Forest feature importances |
| SelectKBest ANOVA F-Score | Statistical feature relevance |
| Correlation Heatmap | Inter-feature Pearson correlation |
| Class Distribution | Before/after SMOTE balancing |

### Multi-Dataset Dashboard (`/multi-analysis`)
| Chart | Description |
|---|---|
| Cross-Dataset Matrix | Train-on-X, test-on-Y accuracy heatmap |
| Feature Shift Boxplots | Feature distribution shifts across datasets |
| Per-Dataset Results | Best model per corpus |
| Pooled Results | Model performance on merged data |
| ANOVA Feature Shift Table | Top 10 features most shifted between datasets |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.x, Flask |
| Image Processing | OpenCV, scikit-image |
| Machine Learning | scikit-learn, imbalanced-learn (SMOTE) |
| Visualisation | Matplotlib, Seaborn |
| Numerical | NumPy, Pandas, SciPy |
| PDF Support | PyMuPDF (fitz) |
| Serialisation | Joblib |
| Frontend | HTML5, Bootstrap 5 |

---

## 📁 Output Files

| File | Description |
|---|---|
| `model.pkl` | Best serialised ML model |
| `model_report.txt` | Full classification report & CV scores |
| `static/charts/*.png` | All diagnostic chart images |
| `static/charts/multi_dataset/per_dataset_results.csv` | Per-dataset model metrics |
| `static/charts/multi_dataset/pooled_results.csv` | Pooled training metrics |
| `static/charts/multi_dataset/cross_dataset_matrix.csv` | Cross-generalisation matrix |
| `static/charts/multi_dataset/feature_shift_anova.csv` | ANOVA feature shift p-values |
| `static/processed/features.csv` | Features extracted from last uploaded batch |

---

## 🔒 Notes

- Dataset images are **gitignored** — only the folder structure is tracked via `.gitkeep` files
- The `model.pkl` file (~5 MB) is tracked and can be used directly without retraining
- Set `FLASK_DEBUG=1` environment variable to enable Flask debug mode
- App runs on port `8501` by default (overridable via `PORT` env var)

---

## 📄 License

This project is for academic research and educational purposes.
