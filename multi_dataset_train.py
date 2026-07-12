"""
DataWing - Multi-Dataset Comparative Analysis
================================================
Extends the original single-dataset pipeline (train.py) to work across
THREE handwriting datasets. Produces:

  1. Individual performance for each dataset (same 5-model comparison
     as the original project, run separately per dataset)
  2. A combined/pooled model trained on all three datasets merged
  3. A cross-dataset generalization matrix: train on dataset X,
     test on dataset Y, for every pair -> reveals whether a model
     trained on one dataset actually works on the others
  4. Feature-distribution comparison across datasets (do the 3
     datasets "look" statistically similar?)
  5. A final summary that states which dataset/model combination
     gives the BEST result, and which generalizes best

FOLDER STRUCTURE EXPECTED
--------------------------
Each dataset can use its OWN class-folder names and its OWN internal
layout (flat, or nested under Train/Test) - the loader below walks
every sub-folder recursively and matches folder names against
LABEL_MAP, so nesting doesn't matter.

dataset/
    mendeley/
        dysgraphic/   *.png / *.jpg
        normal/
    dysgraphia_patients/          <- "Handwriting Dataset of Dysgraphia Patients"
        low potential/
        high potential/
        corrected/                 (mapped to normal/0, see LABEL_MAP)
    gambo/                         <- has Train/ and Test/ subfolders, each
        Train/                        containing Reversal/ Normal/ Corrected/
            Reversal/
            Normal/
            Corrected/
        Test/
            Reversal/
            Normal/
            Corrected/



Run:  python multi_dataset_train.py
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import f_oneway

# Reuse the exact feature-extraction logic from the original project so
# results stay comparable across datasets.
from train import extract_features, ML_FEATURES, FEATURE_DISPLAY

DATASET_DIR  = "dataset"
CHARTS_DIR   = "static/charts/multi_dataset"
MODEL_DIR    = "models_multi"
RANDOM_STATE = 42
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# QUICK-TEST MODE
# Set this to a number (e.g. 300) to randomly cap how many images are
# processed PER CLASS PER DATASET - lets you sanity-check the whole
# pipeline and charts in a few minutes before committing to the full run.
# Set back to None for the real, final run used in your report.
# ---------------------------------------------------------------------------
MAX_SAMPLES_PER_CLASS = 5000   # 5000/class = 10k samples per dataset — statistically solid; set to None only if you have hours to spare (SVM is O(n²)).

# ---------------------------------------------------------------------------
# Per-dataset class-folder -> binary label mapping.
# Keys = dataset source folder name (under dataset/). Values = mapping from
# class-folder name (case-insensitive, matched anywhere in the tree - so
# Train/Test nesting is handled automatically) to a binary dysgraphia label.
# "corrected" = handwriting that no longer shows dysgraphia symptoms ->
# treated as normal (0), same as low-potential/Normal samples.
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "mendeley": {
        "dysgraphic": 1,
        "normal": 0,
    },
    "dysgraphia_patients": {          # "Handwriting Dataset of Dysgraphia Patients" -
                                       # both potential categories ARE dysgraphic patients,
                                       # split by doctor's recovery outlook; only "corrected"
                                       # means they've recovered -> normal
        "low potential": 1,           # dysgraphic - low potential for recovery
        "high potential": 1,          # dysgraphic - high potential for recovery
        "corrected": 0,               # recovered -> normal
    },
    "gambo": {
        "reversal": 1,                # letter reversal - classic dysgraphia/dyslexia marker
        "normal": 0,
        "corrected": 0,               # no longer dysgraphic -> normal
    },
}

# ---------------------------------------------------------------------------
# DATASET GRANULARITY & FEATURE APPLICABILITY
#
# The three datasets contain fundamentally different amounts of content per
# image:
#   - Gambo             -> single isolated LETTER   (e.g. one "R")
#   - mendeley           -> single SENTENCE/LINE     (one line of cursive text)
#   - dysgraphia_patients -> full multi-line PARAGRAPH
#
# Several of the 18 features require multiple letters (or multiple lines) to
# mean anything. On a single-letter image, "word_spacing" or "margin_alignment"
# aren't just noisy - they're structurally undefined (there's only one letter,
# no words, no second line to compare a margin against), so they collapse to a
# constant/degenerate value that has nothing to do with dysgraphia. Pooling
# these features across datasets risks teaching a model to recognise WHICH
# DATASET a sample came from (via its content amount) rather than genuine
# handwriting-based dysgraphia markers - which is a likely contributor to the
# weak cross-dataset transfer observed in the generalization matrix.
#
# CORE_FEATURES below is the subset that remains meaningful and reasonably
# scale-comparable down to the single-letter level - i.e. the only fair
# common ground for POOLED and CROSS-DATASET comparisons. Per-dataset-only
# comparisons keep using the FULL 18 features, since within one dataset every
# sample shares the same granularity, so there is no cross-contamination.
# ---------------------------------------------------------------------------
DATASET_GRANULARITY = {
    "gambo": "letter",
    "mendeley": "line",
    "dysgraphia_patients": "paragraph",
}

# For each feature: the minimum granularity at which it is structurally
# defined, and whether its raw magnitude scales with amount of ink/content
# (which makes it a "dataset detector" even when technically computable).
FEATURE_NOTES = {
    "mean_letter_height":                 {"min_granularity": "letter", "scale_dependent": False},
    "mean_letter_width":                  {"min_granularity": "letter", "scale_dependent": False},
    "corner_density":                     {"min_granularity": "letter", "scale_dependent": False},
    "component_density":                  {"min_granularity": "letter", "scale_dependent": False},
    "slant_angle_deg":                    {"min_granularity": "letter", "scale_dependent": False},

    "std_letter_height":                  {"min_granularity": "line", "scale_dependent": False},
    "std_letter_width":                   {"min_granularity": "line", "scale_dependent": False},
    "letter_height_cv":                   {"min_granularity": "line", "scale_dependent": False},
    "proportion_consistency_std":         {"min_granularity": "line", "scale_dependent": False},
    "letter_spacing":                     {"min_granularity": "line", "scale_dependent": False},
    "word_spacing":                       {"min_granularity": "line", "scale_dependent": False},
    "horizontal_regularity_baseline_std": {"min_granularity": "line", "scale_dependent": False},
    "vertical_regularity_height_std":     {"min_granularity": "line", "scale_dependent": False},

    "stroke_length_total":                {"min_granularity": "letter", "scale_dependent": True},
    "corner_count":                       {"min_granularity": "letter", "scale_dependent": True},
    "num_components":                     {"min_granularity": "letter", "scale_dependent": True},
    "total_ink_area":                     {"min_granularity": "letter", "scale_dependent": True},

    "margin_alignment_std":               {"min_granularity": "paragraph", "scale_dependent": False},
}

CORE_FEATURES = [
    f for f in ML_FEATURES
    if FEATURE_NOTES[f]["min_granularity"] == "letter" and not FEATURE_NOTES[f]["scale_dependent"]
]
# -> ["mean_letter_height", "mean_letter_width", "corner_density",
#     "component_density", "slant_angle_deg"]


MODELS = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE))]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE))]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE))]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))]),
}


# ---------------------------------------------------------------------------
# 1. LOAD ALL DATASETS, TAGGED BY SOURCE
# ---------------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_all_datasets():
    """Walks dataset/<source>/... recursively (any depth - so Train/Test
    nesting like Gambo's is handled automatically) and matches every
    leaf folder name against that source's entry in LABEL_MAP
    (case-insensitive). Folders not listed in LABEL_MAP (e.g. "corrected")
    are skipped with a note, but their image count is still reported so
    you can see what's being left out.

    Returns a single DataFrame with columns: [18 features..., label, source]
    """
    rows = []
    excluded_counts = {}
    sources = sorted([d for d in os.listdir(DATASET_DIR)
                       if os.path.isdir(os.path.join(DATASET_DIR, d))])
    if not sources:
        raise RuntimeError(f"No dataset folders found inside {DATASET_DIR}/")

    print(f"Found {len(sources)} dataset folder(s): {sources}\n")

    label_map_lower = {k.lower(): v for k, v in LABEL_MAP.items()}

    for source in sources:
        source_key = source.lower()
        if source_key not in label_map_lower:
            print(f"  [!] '{source}' has no entry in LABEL_MAP - skipping entirely. "
                  f"Add it to LABEL_MAP at the top of this script.")
            continue

        class_map = {k.lower(): v for k, v in label_map_lower[source_key].items()}
        source_root = os.path.join(DATASET_DIR, source)
        loaded_this_source = 0

        for dirpath, dirnames, filenames in os.walk(source_root):
            leaf_name = os.path.basename(dirpath).lower()
            image_files = [f for f in filenames if f.lower().endswith(IMAGE_EXTS)]
            if not image_files:
                continue

            if leaf_name not in class_map:
                excluded_counts[f"{source}/{leaf_name}"] = len(image_files)
                continue

            if MAX_SAMPLES_PER_CLASS is not None and len(image_files) > MAX_SAMPLES_PER_CLASS:
                rng = np.random.RandomState(RANDOM_STATE)
                image_files = list(rng.choice(image_files, size=MAX_SAMPLES_PER_CLASS, replace=False))

            label_val = class_map[leaf_name]
            capped_note = f" (capped from folder, quick-test mode)" if MAX_SAMPLES_PER_CLASS is not None else ""
            print(f"  {source:20s} | {leaf_name:15s} -> label {label_val} | {len(image_files)} images{capped_note}")
            t_start = time.time()
            for i, fname in enumerate(image_files, 1):
                img = cv2.imread(os.path.join(dirpath, fname))
                if img is None:
                    continue
                feats = extract_features(img)
                if feats is None:
                    continue
                feats["label"] = label_val
                feats["source"] = source
                rows.append(feats)
                loaded_this_source += 1
                if i % 200 == 0 or i == len(image_files):
                    elapsed = time.time() - t_start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(image_files) - i) / rate if rate > 0 else 0
                    print(f"      ...{i}/{len(image_files)} processed "
                          f"({rate:.1f} img/s, ~{eta/60:.1f} min remaining for this folder)")

        if loaded_this_source == 0:
            print(f"  [!] '{source}' - no images matched LABEL_MAP class names. "
                  f"Check folder names against LABEL_MAP.")

    if excluded_counts:
        print("\nFolders found but NOT included (not in LABEL_MAP, e.g. 'corrected'):")
        for k, v in excluded_counts.items():
            print(f"    {k}: {v} images")

    df = pd.DataFrame(rows)
    print(f"\nTotal samples loaded for binary comparison: {len(df)}")
    if len(df):
        print(df.groupby(["source", "label"]).size())
    return df


# ---------------------------------------------------------------------------
# 2. EVALUATE ALL 5 MODELS ON A GIVEN (X_train, y_train, X_test, y_test)
# ---------------------------------------------------------------------------
def evaluate_models(X_train, y_train, X_test, y_test, cv_folds=5):
    cv = StratifiedKFold(n_splits=min(cv_folds, np.bincount(y_train).min()),
                          shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    for name, pipeline in MODELS.items():
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                         scoring="accuracy", n_jobs=-1)
            cv_mean = cv_scores.mean()
        except ValueError:
            cv_mean = np.nan  # too few samples for the requested folds

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        try:
            auc_val = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc_val = np.nan

        rows.append({"Model": name, "CV Accuracy": round(cv_mean, 4) if not np.isnan(cv_mean) else None,
                     "Accuracy": round(acc, 4), "F1 Score": round(f1, 4),
                     "Precision": round(prec, 4), "Recall": round(rec, 4),
                     "AUC": round(auc_val, 4) if not np.isnan(auc_val) else None})
        fitted[name] = pipeline
    return pd.DataFrame(rows), fitted


# ---------------------------------------------------------------------------
# 3. PER-DATASET INDIVIDUAL PERFORMANCE
# ---------------------------------------------------------------------------
def per_dataset_performance(df):
    print("\n" + "=" * 60)
    print("STEP: Individual per-dataset performance")
    print("=" * 60)
    all_results = {}
    for source in df["source"].unique():
        sub = df[df["source"] == source]
        X = sub[ML_FEATURES].values
        y = sub["label"].values
        if len(np.unique(y)) < 2 or len(sub) < 10:
            print(f"  [!] Skipping {source} - not enough samples/classes")
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        results_df, _ = evaluate_models(X_train, y_train, X_test, y_test)
        results_df.insert(0, "Dataset", source)
        all_results[source] = results_df
        print(f"\n-- {source} --")
        print(results_df.to_string(index=False))

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(f"{CHARTS_DIR}/per_dataset_results.csv", index=False)
    return combined


# ---------------------------------------------------------------------------
# 4. COMBINED / POOLED MODEL (all datasets merged)
# ---------------------------------------------------------------------------
def pooled_performance(df):
    print("\n" + "=" * 60)
    print("STEP: Combined / pooled model (all datasets merged)")
    print(f"Using CORE_FEATURES only ({len(CORE_FEATURES)}/{len(ML_FEATURES)} features) - "
          f"see DATASET_GRANULARITY note at top of script for why.")
    print("=" * 60)
    X = df[CORE_FEATURES].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    results_df, fitted = evaluate_models(X_train, y_train, X_test, y_test)
    results_df.insert(0, "Dataset", "POOLED (all 3)")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{CHARTS_DIR}/pooled_results.csv", index=False)

    # Save the single best-performing pooled model for the web app
    best_row = results_df.sort_values("CV Accuracy", ascending=False).iloc[0]
    best_model = fitted[best_row["Model"]]
    joblib.dump(best_model, f"{MODEL_DIR}/pooled_best_model.pkl")
    print(f"\nBest pooled model: {best_row['Model']} (saved to {MODEL_DIR}/pooled_best_model.pkl)")
    return results_df


# ---------------------------------------------------------------------------
# 5. CROSS-DATASET GENERALIZATION MATRIX
#    Train on dataset X (full), test on dataset Y (full) - for every pair.
#    Uses a single representative model (Random Forest, the strongest
#    performer in the original project) to keep the matrix readable.
# ---------------------------------------------------------------------------
def cross_dataset_matrix(df, model_name="Random Forest"):
    print("\n" + "=" * 60)
    print(f"STEP: Cross-dataset generalization matrix ({model_name})")
    print(f"Using CORE_FEATURES only ({len(CORE_FEATURES)}/{len(ML_FEATURES)} features)")
    print("=" * 60)
    sources = sorted(df["source"].unique())
    matrix = pd.DataFrame(index=sources, columns=sources, dtype=float)

    # Pre-split EVERY dataset once into train/test up front. This is the
    # critical fix: the diagonal (train on X, test on X) must use a
    # genuinely held-out test split, never rows the model was fit on -
    # otherwise the diagonal trivially reads 1.000 (memorization, not
    # generalization). Off-diagonal cells also use the same held-out
    # test split of the target dataset, so every column is comparable.
    splits = {}
    for source in sources:
        sub = df[df["source"] == source]
        X = sub[CORE_FEATURES].values
        y = sub["label"].values
        if len(np.unique(y)) < 2 or len(sub) < 10:
            print(f"  [!] Skipping {source} - not enough samples/classes")
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        splits[source] = (X_tr, X_te, y_tr, y_te)

    from sklearn.base import clone
    for train_src in sources:
        if train_src not in splits:
            continue
        X_tr, _, y_tr, _ = splits[train_src]
        pipeline = clone(MODELS[model_name])
        pipeline.fit(X_tr, y_tr)

        for test_src in sources:
            if test_src not in splits:
                continue
            _, X_te, _, y_te = splits[test_src]
            acc = accuracy_score(y_te, pipeline.predict(X_te))
            matrix.loc[train_src, test_src] = round(acc, 4)

    print(matrix)
    matrix.to_csv(f"{CHARTS_DIR}/cross_dataset_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Accuracy"})
    ax.set_xlabel("Tested on ->")
    ax.set_ylabel("Trained on ->")
    ax.set_title(f"Cross-Dataset Generalization Matrix\n({model_name})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/cross_dataset_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    return matrix


# ---------------------------------------------------------------------------
# 6. FEATURE DISTRIBUTION SHIFT ACROSS DATASETS (ANOVA + boxplots)
# ---------------------------------------------------------------------------
def feature_shift_analysis(df):
    print("\n" + "=" * 60)
    print("STEP: Feature distribution shift across datasets (ANOVA)")
    print("=" * 60)
    sources = sorted(df["source"].unique())
    anova_rows = []
    for feat in ML_FEATURES:
        groups = [df[df["source"] == s][feat].values for s in sources]
        try:
            fstat, pval = f_oneway(*groups)
        except ValueError:
            fstat, pval = np.nan, np.nan
        anova_rows.append({
            "Feature": FEATURE_DISPLAY[feat],
            "F-stat": fstat,
            "p-value": pval,
            "Significant Shift (p<0.05)": pval < 0.05 if not np.isnan(pval) else False,
            "Core Feature (comparable across datasets)": feat in CORE_FEATURES,
        })
    anova_df = pd.DataFrame(anova_rows).sort_values("p-value")
    print(anova_df.to_string(index=False))
    print("\nNote: a 'significant shift' in a non-core feature is EXPECTED (letter vs. "
          "line vs. paragraph content differs structurally) and is not evidence of a "
          "real dysgraphia-related difference. Only shifts in CORE features are "
          "potentially meaningful across datasets.")
    anova_df.to_csv(f"{CHARTS_DIR}/feature_shift_anova.csv", index=False)

    # Boxplots for the 6 most-shifted features (lowest p-value)
    top_feats = anova_df.head(6)["Feature"].tolist()
    top_keys = [k for k in ML_FEATURES if FEATURE_DISPLAY[k] in top_feats]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feat in zip(axes.flat, top_keys):
        sns.boxplot(data=df, x="source", y=feat, ax=ax, palette="Set2")
        ax.set_title(FEATURE_DISPLAY[feat], fontsize=10, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Most Dataset-Dependent Features (lowest ANOVA p-value)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/feature_shift_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    return anova_df


# ---------------------------------------------------------------------------
# 7. FINAL SUMMARY - which dataset/model wins, and does it generalize
# ---------------------------------------------------------------------------
def write_summary(per_dataset_df, pooled_df, matrix):
    lines = []
    lines.append("DataWing — Multi-Dataset Comparative Analysis Summary")
    lines.append("=" * 60 + "\n")

    best_individual = per_dataset_df.sort_values("CV Accuracy", ascending=False).iloc[0]
    lines.append(f"Best single dataset+model combination:")
    lines.append(f"  Dataset : {best_individual['Dataset']}")
    lines.append(f"  Model   : {best_individual['Model']}")
    lines.append(f"  CV Acc  : {best_individual['CV Accuracy']}   Accuracy: {best_individual['Accuracy']}   F1: {best_individual['F1 Score']}\n")

    best_pooled = pooled_df.sort_values("CV Accuracy", ascending=False).iloc[0]
    lines.append(f"Best pooled (all-3-combined) model:")
    lines.append(f"  Model   : {best_pooled['Model']}")
    lines.append(f"  CV Acc  : {best_pooled['CV Accuracy']}   Accuracy: {best_pooled['Accuracy']}   F1: {best_pooled['F1 Score']}\n")

    diag = np.diag(matrix.values.astype(float))
    off_diag_mask = ~np.eye(len(matrix), dtype=bool)
    avg_within = np.nanmean(diag)
    avg_cross = np.nanmean(matrix.values.astype(float)[off_diag_mask])
    lines.append("Generalization check:")
    lines.append(f"  Avg within-dataset accuracy : {avg_within:.4f}")
    lines.append(f"  Avg cross-dataset accuracy  : {avg_cross:.4f}")
    gap = avg_within - avg_cross
    if gap > 0.1:
        verdict = ("Large drop when testing across datasets -> the model is "
                   "overfitting to dataset-specific quirks (scanner/pen/paper), "
                   "not learning general dysgraphia patterns.")
    elif gap > 0.03:
        verdict = ("Moderate drop across datasets -> some generalization, "
                   "but dataset-specific bias is still present.")
    else:
        verdict = ("Small drop across datasets -> the model generalizes well; "
                   "features are capturing real handwriting patterns, not dataset artifacts.")
    lines.append(f"  Verdict: {verdict}\n")

    summary = "\n".join(lines)
    summary += (f"\nNote on methodology: the pooled model and cross-dataset matrix above "
                f"were trained using only {len(CORE_FEATURES)} of the 18 features "
                f"({', '.join(FEATURE_DISPLAY[f] for f in CORE_FEATURES)}), since the "
                f"other features (letter/word spacing, baseline/margin regularity, and "
                f"raw content-scale features) are not meaningfully comparable across "
                f"datasets with different content granularity (single letters vs. "
                f"single lines vs. full paragraphs). Per-dataset results in Section 10.3 "
                f"still use the full 18-feature set.\n")
    print("\n" + summary)
    with open(f"{CHARTS_DIR}/final_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)


# ---------------------------------------------------------------------------
def main():
    df = load_all_datasets()
    df.to_csv(f"{CHARTS_DIR}/all_features_tagged.csv", index=False)

    per_dataset_df = per_dataset_performance(df)
    pooled_df = pooled_performance(df)
    matrix = cross_dataset_matrix(df)
    feature_shift_analysis(df)
    write_summary(per_dataset_df, pooled_df, matrix)

    print(f"\nAll charts & CSVs saved under: {CHARTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()