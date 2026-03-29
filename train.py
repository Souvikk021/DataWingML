import os, cv2, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.spatial.distance import euclidean

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_curve, auc)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

DATASET_DIR  = "dataset"
MODEL_PATH   = "model.pkl"
REPORT_PATH  = "model_report.txt"
CHARTS_DIR   = "static/charts"
RANDOM_STATE = 42
os.makedirs(CHARTS_DIR, exist_ok=True)

ML_FEATURES = [
    "mean_letter_height", "std_letter_height", "mean_letter_width",
    "std_letter_width", "letter_height_cv", "proportion_consistency_std",
    "letter_spacing", "word_spacing", "stroke_length_total",
    "corner_count", "corner_density", "num_components", "total_ink_area",
    "component_density", "horizontal_regularity_baseline_std",
    "vertical_regularity_height_std", "margin_alignment_std", "slant_angle_deg"
]

FEATURE_DISPLAY = {
    "mean_letter_height":                   "Mean Letter Height",
    "std_letter_height":                    "Std Letter Height",
    "mean_letter_width":                    "Mean Letter Width",
    "std_letter_width":                     "Std Letter Width",
    "letter_height_cv":                     "Height Coeff. of Variation",
    "proportion_consistency_std":           "Proportion Consistency Std",
    "letter_spacing":                       "Letter Spacing",
    "word_spacing":                         "Word Spacing",
    "stroke_length_total":                  "Total Stroke Length",
    "corner_count":                         "Corner Count",
    "corner_density":                       "Corner Density",
    "num_components":                       "Num Components",
    "total_ink_area":                       "Total Ink Area",
    "component_density":                    "Component Density",
    "horizontal_regularity_baseline_std":   "Baseline Regularity Std",
    "vertical_regularity_height_std":       "Vertical Regularity Std",
    "margin_alignment_std":                 "Margin Alignment Std",
    "slant_angle_deg":                      "Slant Angle (deg)"
}

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    comps = []
    for i in range(1, num_labels):
        x, y, w, h, a = stats[i]
        if a > 20:
            comps.append((x, y, w, h, a, centroids[i]))
    if not comps:
        return None
    heights = np.array([c[3] for c in comps])
    widths  = np.array([c[2] for c in comps])
    lefts   = np.array([c[0] for c in comps])
    mean_h  = float(np.mean(heights)); std_h = float(np.std(heights))
    mean_w  = float(np.mean(widths));  std_w = float(np.std(widths))
    letter_height_cv           = std_h / (mean_h + 1e-6)
    proportions                = widths / (heights + 1e-6)
    proportion_consistency_std = float(np.std(proportions))
    comps_lr = sorted(comps, key=lambda c: c[0])
    gaps = [comps_lr[i+1][0] - (comps_lr[i][0] + comps_lr[i][2]) for i in range(len(comps_lr)-1)]
    letter_spacing = float(np.percentile(gaps, 30)) if gaps else 0
    word_spacing   = float(np.percentile(gaps, 80)) if gaps else 0
    skel    = skeletonize(thresh > 0)
    skel_u8 = img_as_ubyte(skel)
    ys, xs  = np.where(skel_u8 > 0)
    stroke_len = sum(euclidean((xs[i],ys[i]),(xs[i+1],ys[i+1])) for i in range(len(xs)-1)) if len(xs)>1 else 0
    corners       = cv2.goodFeaturesToTrack(thresh, maxCorners=500, qualityLevel=0.01, minDistance=6)
    corner_count  = 0 if corners is None else len(corners)
    total_ink     = float(np.sum(thresh > 0))
    corner_density = corner_count / (total_ink + 1e-6)
    baseline_std  = np.std([c[5][1] for c in comps]) / (mean_h + 1e-6)
    return {
        "mean_letter_height": mean_h, "std_letter_height": std_h,
        "mean_letter_width": mean_w,  "std_letter_width": std_w,
        "letter_height_cv": letter_height_cv,
        "proportion_consistency_std": proportion_consistency_std,
        "letter_spacing": letter_spacing, "word_spacing": word_spacing,
        "stroke_length_total": float(stroke_len),
        "corner_count": corner_count, "corner_density": corner_density,
        "num_components": len(comps), "total_ink_area": total_ink,
        "component_density": len(comps) / (total_ink + 1e-6),
        "horizontal_regularity_baseline_std": float(baseline_std),
        "vertical_regularity_height_std": float(np.std(heights)),
        "margin_alignment_std": float(np.std(lefts)),
        "slant_angle_deg": 0.0
    }

def load_dataset():
    records, labels = [], []
    for label_name, label_val in [("dysgraphic", 1), ("normal", 0)]:
        folder = os.path.join(DATASET_DIR, label_name)
        if not os.path.exists(folder):
            print(f"  Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))]
        print(f"  {label_name}: {len(files)} images")
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None: continue
            feats = extract_features(img)
            if feats is None: continue
            records.append(feats); labels.append(label_val)
    return records, labels

def balance_classes(X, y):
    df = pd.DataFrame(X); df["__label__"] = y
    counts = df["__label__"].value_counts()
    majority = df[df["__label__"] == counts.idxmax()]
    minority = df[df["__label__"] == counts.idxmin()]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=RANDOM_STATE)
    balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    y_bal = balanced["__label__"].tolist()
    X_bal = balanced.drop(columns=["__label__"]).to_dict(orient="records")
    return X_bal, y_bal

def plot_correlation_heatmap(df_features):
    print("  Correlation heatmap...")
    corr = df_features[ML_FEATURES].corr()
    labels = [FEATURE_DISPLAY[f] for f in ML_FEATURES]
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, linewidths=0.4,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.xticks(fontsize=7, rotation=45, ha="right"); plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()

def plot_selectkbest(X, y):
    print("  SelectKBest feature relevance...")
    sel = SelectKBest(score_func=f_classif, k="all")
    sel.fit(X, y)
    df_imp = pd.DataFrame({
        "feature": [FEATURE_DISPLAY[f] for f in ML_FEATURES],
        "score": sel.scores_, "pvalue": sel.pvalues_
    }).sort_values("score", ascending=True)
    colors = ["#dc2626" if p < 0.05 else "#94a3b8" for p in df_imp["pvalue"]]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(df_imp["feature"], df_imp["score"], color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("F-Score (ANOVA)", fontsize=11)
    ax.set_title("Feature Relevance — SelectKBest (ANOVA F-Test)\nRed = significant (p < 0.05)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color="#dc2626", label="Significant (p<0.05)"),
                        mpatches.Patch(color="#94a3b8", label="Not significant")],
              loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/feature_relevance_selectkbest.png", dpi=150, bbox_inches="tight"); plt.close()
    return df_imp[df_imp["pvalue"] < 0.05]["feature"].tolist()

def plot_rf_importance(X, y):
    print("  Random Forest feature importance...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced")
    rf.fit(X, y)
    df_imp = pd.DataFrame({
        "feature": [FEATURE_DISPLAY[f] for f in ML_FEATURES],
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df_imp)))
    ax.barh(df_imp["feature"], df_imp["importance"], color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/feature_importance_rf.png", dpi=150, bbox_inches="tight"); plt.close()

def plot_model_comparison(results_df):
    print("  Model comparison chart...")
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    x = np.arange(len(results_df)); width = 0.2
    colors = ["#0f3460","#16213e","#e94560","#533483"]
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = results_df[metric].values
        bars = ax.bar(x + i*width, vals, width, label=metric, color=color, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x + width*1.5); ax.set_xticklabels(results_df["Model"], fontsize=11)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("ML Model Comparison — Accuracy, F1, Precision, Recall",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.spines[["top","right"]].set_visible(False); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/model_comparison.png", dpi=150, bbox_inches="tight"); plt.close()

def plot_confusion_matrices(cm_dict):
    print("  Confusion matrices...")
    n = len(cm_dict); ncols = 3; nrows = (n+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows)); axes = axes.flatten()
    for idx, (name, cm) in enumerate(cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal","Dysgraphic"],
                    yticklabels=["Normal","Dysgraphic"],
                    ax=axes[idx], cbar=False, linewidths=1)
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted"); axes[idx].set_ylabel("Actual")
    for j in range(idx+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight"); plt.close()

def plot_roc_curves(roc_dict):
    print("  ROC curves...")
    colors = ["#0f3460","#e94560","#16a34a","#d97706","#7c3aed"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, (fpr, tpr, roc_auc)), color in zip(roc_dict.items(), colors):
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1.5,alpha=0.5,label="Random Classifier")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("False Positive Rate",fontsize=12); ax.set_ylabel("True Positive Rate",fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.spines[["top","right"]].set_visible(False); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/roc_curves.png", dpi=150, bbox_inches="tight"); plt.close()

def plot_cv_scores(cv_results):
    print("  CV score distribution...")
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(cv_results.keys()); scores = list(cv_results.values())
    colors = ["#0f3460","#e94560","#16a34a","#d97706","#7c3aed"]
    bp = ax.boxplot(scores, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Cross-Validation Accuracy", fontsize=12)
    ax.set_title("5-Fold Cross-Validation Score Distribution", fontsize=14, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/cv_score_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

def plot_class_distribution(y_raw, y_balanced):
    print("  Class distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title in zip(axes, [y_raw, y_balanced], ["Before Balancing","After Balancing"]):
        counts = [y.count(0), y.count(1)]
        bars = ax.bar(["Normal","Dysgraphic"], counts, color=["#16a34a","#dc2626"], edgecolor="white", width=0.5)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(count), ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Sample Count", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylim(0, max(counts)*1.25)
    fig.suptitle("Class Distribution in Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/class_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

def train():
    print("\n" + "="*60)
    print("   DataWing — ML Training Pipeline")
    print("="*60)

    print("\nStep 1: Loading dataset...")
    records, labels = load_dataset()

    if len(records) < 10:
        print(f"\nNot enough samples ({len(records)} found, need >= 10).")
        print("   Add images to dataset/dysgraphic/ and dataset/normal/ then re-run.")
        return

    print(f"\nLoaded {len(records)} samples  "
          f"[Dysgraphic: {labels.count(1)}  |  Normal: {labels.count(0)}]")

    df_all = pd.DataFrame(records); df_all["label"] = labels
    df_all.to_csv(f"{CHARTS_DIR}/dataset_features.csv", index=False)
    y_raw = labels[:]

    if abs(labels.count(1) - labels.count(0)) > 3:
        print("\nStep 2: Balancing classes (oversampling minority)...")
        records, labels = balance_classes(records, labels)
        print(f"   Dysgraphic: {labels.count(1)}  |  Normal: {labels.count(0)}")
    else:
        print("\nStep 2: Classes already balanced.")

    X = np.array([[r[k] for k in ML_FEATURES] for r in records])
    y = np.array(labels)

    print("\nStep 3: Feature relevance analysis...")
    df_feat = pd.DataFrame(records, columns=ML_FEATURES)
    plot_correlation_heatmap(df_feat)
    sig_features = plot_selectkbest(X, y)
    plot_rf_importance(X, y)
    plot_class_distribution(y_raw, labels)

    print(f"\n  Significant features ({len(sig_features)}): {', '.join(sig_features[:5])}{'...' if len(sig_features)>5 else ''}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = {
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nStep 4: Training & evaluating 5 models...\n")
    results_rows = []; cm_dict = {}; roc_dict = {}; cv_results = {}
    best_name = None; best_score = 0; best_model = None

    for name, pipeline in models.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_results[name] = cv_scores
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        cv_mean = cv_scores.mean()
        print(f"  {name:22s} | CV: {cv_mean:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
        results_rows.append({"Model": name, "CV Accuracy": round(cv_mean,4),
                              "Accuracy": round(acc,4), "F1 Score": round(f1,4),
                              "Precision": round(prec,4), "Recall": round(rec,4),
                              "AUC": round(roc_auc,4)})
        cm_dict[name]  = confusion_matrix(y_test, y_pred)
        roc_dict[name] = (fpr, tpr, roc_auc)
        if cv_mean > best_score:
            best_score = cv_mean; best_name = name; best_model = pipeline

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(f"{CHARTS_DIR}/model_results.csv", index=False)
    print(f"\nBest model: {best_name}  (CV accuracy: {best_score:.4f})")

    print("\nStep 5: Generating all charts...")
    plot_model_comparison(results_df)
    plot_confusion_matrices(cm_dict)
    plot_roc_curves(roc_dict)
    plot_cv_scores(cv_results)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write("DataWing — ML Training Report\n" + "="*50 + "\n\n")
        f.write(f"Best Model  : {best_name}\nCV Accuracy : {best_score:.4f}\n\n")
        f.write("All Models:\n"); f.write(results_df.to_string(index=False))
        f.write("\n\nStatistically Significant Features:\n")
        for sf in sig_features: f.write(f"  -> {sf}\n")
        f.write("\nClassification Report (Best Model):\n")
        best_model.fit(X_train, y_train)
        f.write(classification_report(y_test, best_model.predict(X_test),
                                      target_names=["Normal","Dysgraphic"]))

    print(f"Report saved -> {REPORT_PATH}")
    print("\nAll done. Run python app.py — ML model + dashboard are ready.\n" + "="*60)

if __name__ == "__main__":
    train()