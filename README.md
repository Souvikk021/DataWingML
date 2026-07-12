# 🪶 DataWing — Multi-Model Dysgraphia Detection via Handwriting Analysis

> An end-to-end machine learning web application that screens handwriting samples for **dysgraphia** using classical computer vision feature extraction and an ensemble of ML classifiers.

---

## 📌 Overview

DataWing ingests handwriting images or PDFs, extracts 18 biomechanical and spatial features, and runs them through a multi-model ML pipeline. It supports:

- **Single-image / multi-image / PDF analysis** with per-sample predictions
- **Multi-dataset training** across three independent handwriting corpora
- **Cross-dataset generalisation testing** (train on one, test on another)
- **Rule-based screening** as a transparent interpretable baseline
- **Interactive dashboard** with 8+ diagnostic charts
- **Medical disclaimer** shown whenever dysgraphia indicators are detected

---

## 🗂️ Project Structure

```
DataWing/
├── app.py                    # Flask web application & feature extraction pipeline
├── train.py                  # Single-dataset ML training script (Mendeley)
├── multi_dataset_train.py    # Multi-dataset training, pooling & cross-testing
├── model.pkl                 # Serialised best model (auto-selected by CV accuracy)
├── model_report.txt          # Human-readable model performance report
├── requirements.txt          # Python dependencies
│
├── dataset/                  # Handwriting image datasets (not tracked in git)
│   ├── Gambo/                # Single-letter dataset
│   │   ├── Train/  (Normal / Corrected / Reversal)
│   │   └── Test/   (Normal / Corrected / Reversal)
│   ├── dysgraphia_patients/  # Paragraph-level dataset
│   │   ├── high potential/
│   │   ├── low potential/
│   │   └── corrected/
│   └── mendeley/             # Sentence-level dataset (primary training set)
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
    ├── index.html            # Upload page (home)
    ├── result.html           # Per-sample prediction results
    ├── dashboard.html        # Single-dataset training dashboard
    └── multi_analysis.html   # Multi-dataset comparative analysis dashboard
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Souvikk021/Datawing---Multi-Model.git
cd Datawing---Multi-Model
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

Dataset files are **not tracked in this repository** (too large for Git). Download and place them manually:

| Dataset | Source | Place in |
|---|---|---|
| Mendeley Dysgraphia | [data.mendeley.com/datasets/39hr8dx76p/1](https://data.mendeley.com/datasets/39hr8dx76p/1) | `dataset/mendeley/normal/` and `dataset/mendeley/dysgraphic/` |
| Gambo Dataset | Sourced separately | `dataset/Gambo/Train/` and `dataset/Gambo/Test/` |
| Dysgraphia Patients | Sourced separately | `dataset/dysgraphia_patients/{high potential, low potential, corrected}/` |

> Minimum **20 images per class** required. 50+ per class recommended for reliable model training.

---

## 🚀 Usage

### Step 1 — Train a model

**Single dataset (Mendeley — powers live predictions):**
```bash
python train.py
```

**Multi-dataset (all three datasets + pooled model + cross-dataset testing):**
```bash
python multi_dataset_train.py
```

Both scripts auto-select the best model by CV accuracy, save `model.pkl`, and generate charts into `static/charts/`.

### Step 2 — Run the web app
```bash
python app.py
```

The server starts at **http://localhost:8501**

| Route | Description |
|---|---|
| `/` | Upload handwriting image or PDF — get per-sample predictions |
| `/dashboard` | Single-dataset model comparison & 8 diagnostic charts |
| `/multi-analysis` | Multi-dataset results, cross-dataset matrix, ANOVA feature shift |

---

## 🧠 How It Works

### Upload & Analysis Flow

1. Upload one or more images **or a multi-page PDF**
2. Each page/image is processed independently
3. Results are displayed as separate cards — one per image
4. Each card shows: 4 image views, dual predictions, extracted features, and (if dysgraphic) a doctor review notice

### Feature Extraction Pipeline

Each image goes through:

1. **Ruled-line detection & removal** — horizontal line suppression via morphological ops
2. **Binarisation** — Gaussian blur → Otsu thresholding
3. **Connected component analysis** — isolates individual letter components
4. **Skeletonization** — extracts stroke skeleton for stroke length measurement
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
| Corner Count / Density | Direction changes per ink area (tremor indicator) |
| Num Components / Density | Writing fragmentation measure |
| Total Ink Area | Total ink pixel coverage |
| Baseline Regularity Std | Vertical deviation from baseline |
| Vertical Regularity Std | Height consistency across letters |
| Margin Alignment Std | Left margin consistency |
| Slant Angle | Character forward/backward lean |

### Dual Prediction System

Every sample receives two independent predictions:

- **Rule-based screening** — threshold logic across 6 features; flags as *Dysgraphic* if ≥ 3 are triggered
- **ML model prediction** — trained Random Forest (Mendeley) with confidence score (%)

If either system flags dysgraphia, a **Doctor Review Required** notice is shown, clarifying that this is a screening tool — not a clinical diagnosis.

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
| Pooled Results | Model performance on merged 3-dataset data |
| ANOVA Feature Shift Table | Top 10 features most shifted between datasets |

---

## 📐 Multi-Dataset Methodology

The multi-dataset analysis deliberately separates two comparison modes:

| Mode | Features Used | Why |
|---|---|---|
| **Per-dataset** (each corpus alone) | All 18 features | Same content scale within each dataset |
| **Pooled + Cross-dataset** | 5 core features only | Avoid false signal from content-scale differences |

The 5 **core features** used for cross-dataset comparisons are:  
`Mean Letter Height`, `Mean Letter Width`, `Corner Density`, `Component Density`, `Slant Angle`

These are the only features that remain comparable whether the sample is a single letter (Gambo), a sentence (Mendeley), or a full paragraph (Dysgraphia Patients).

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
| Frontend | HTML5, Vanilla CSS (DM Sans / DM Mono) |

---

## 📁 Output Files

| File | Description |
|---|---|
| `model.pkl` | Best serialised ML model |
| `model_report.txt` | Full classification report & CV scores |
| `static/charts/*.png` | All single-dataset diagnostic chart images |
| `static/charts/multi_dataset/per_dataset_results.csv` | Per-dataset model metrics |
| `static/charts/multi_dataset/pooled_results.csv` | Pooled training metrics |
| `static/charts/multi_dataset/cross_dataset_matrix.csv` | Cross-generalisation accuracy matrix |
| `static/charts/multi_dataset/feature_shift_anova.csv` | ANOVA feature shift p-values |
| `static/charts/multi_dataset/final_summary.txt` | Human-readable multi-dataset verdict |
| `static/processed/features.csv` | Feature values from the last uploaded batch |

---

## 🔒 Notes

- Dataset images are **gitignored** — only folder structure is tracked via `.gitkeep` files
- The `model.pkl` file is tracked and can be used directly without retraining
- Set `FLASK_DEBUG=1` environment variable to enable Flask debug mode
- App runs on port `8501` by default (overridable via `PORT` env var)
- All output text files are saved as **UTF-8** to ensure correct rendering of special characters

---

## ⚠️ Medical Disclaimer

DataWing is a **research and educational screening tool only**. It is **not a clinical diagnostic instrument**. Any positive result should be reviewed by a qualified medical professional, educational psychologist, or specialist before any conclusions are drawn.

---

## 📄 License

This project is for academic research and educational purposes.
