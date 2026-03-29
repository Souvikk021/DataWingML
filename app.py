import os, uuid, cv2, warnings
import numpy as np
import pandas as pd
import joblib
import fitz
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, url_for, jsonify
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.spatial.distance import euclidean


UPLOAD_FOLDER    = "static/uploads"
PROCESSED_FOLDER = "static/processed"
CHARTS_DIR       = "static/charts"
MODEL_PATH       = "model.pkl"
ALLOWED_EXT      = {"png","jpg","jpeg","tif","tiff","pdf"}

for d in [UPLOAD_FOLDER, PROCESSED_FOLDER, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)

ml_model = None
if os.path.exists(MODEL_PATH):
    ml_model = joblib.load(MODEL_PATH)
    print("ML model loaded.")
else:
    print("No model.pkl found. Run train.py first.")

ML_FEATURES = [
    "mean_letter_height","std_letter_height","mean_letter_width","std_letter_width",
    "letter_height_cv","proportion_consistency_std","letter_spacing","word_spacing",
    "stroke_length_total","corner_count","corner_density","num_components",
    "total_ink_area","component_density","horizontal_regularity_baseline_std",
    "vertical_regularity_height_std","margin_alignment_std","slant_angle_deg"
]


def allowed_file(name):
    return "." in name and name.rsplit(".",1)[1].lower() in ALLOWED_EXT

def imwrite_safe(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok: buf.tofile(path)

def pixmap_to_cv2(pix):
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.h, pix.w, 4 if pix.n==4 else 3)
    if pix.n == 4: arr = arr[:,:,:3]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def dysgraphia_screening(f):
    score = 0
    if f["letter_height_cv"] > 0.55:                   score += 1
    if f["horizontal_regularity_baseline_std"] > 2.5:  score += 1
    if f["letter_spacing"] < -40:                      score += 1
    if f["corner_density"] > 0.025:                    score += 1
    if f["proportion_consistency_std"] > 0.7:          score += 1
    if f["vertical_regularity_height_std"] > 10.0:     score += 1
    label = "Dysgraphic" if score >= 3 else "No Strong Indicators"
    return label, score

def ml_predict(features):
    if ml_model is None:
        return "Model not trained yet", None
    try:
        vec  = np.array([[features[k] for k in ML_FEATURES]])
        pred = ml_model.predict(vec)[0]
        proba = ml_model.predict_proba(vec)[0]
        confidence = round(float(np.max(proba)) * 100, 2)
        return ("Dysgraphic" if pred == 1 else "Normal"), confidence
    except Exception as e:
        return f"Error: {str(e)}", None

def remove_ruled_lines(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    horiz_len = max(w // 6, 40)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))

    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    line_rows = np.sum(horiz_lines > 0, axis=1)
    mask = np.zeros_like(horiz_lines)
    for row_idx, row_sum in enumerate(line_rows):
        if row_sum > w * 0.40:
            mask[row_idx, :] = 255

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(mask))

    gray_cleaned = gray.copy()
    gray_cleaned[mask > 0] = 255

    return gray_cleaned, cleaned

def has_ruled_lines(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    line_rows = np.sum(np.sum(horiz > 0, axis=1) > w * 0.40)
    return line_rows >= 3

def extract_features(img, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if has_ruled_lines(gray):
        gray, _ = remove_ruled_lines(gray)

    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    comps = []
    for i in range(1, num_labels):
        x, y, w, h, a = stats[i]
        if a > 20: comps.append((x,y,w,h,a,centroids[i]))

    heights = np.array([c[3] for c in comps]) if comps else np.array([0])
    widths  = np.array([c[2] for c in comps]) if comps else np.array([0])
    lefts   = np.array([c[0] for c in comps]) if comps else np.array([0])

    mean_h = float(np.mean(heights)); std_h = float(np.std(heights))
    mean_w = float(np.mean(widths));  std_w = float(np.std(widths))
    letter_height_cv = std_h / (mean_h + 1e-6)
    proportions = widths / (heights + 1e-6)
    proportion_consistency_std = float(np.std(proportions))

    comps_lr = sorted(comps, key=lambda c: c[0])
    gaps = [comps_lr[i+1][0]-(comps_lr[i][0]+comps_lr[i][2]) for i in range(len(comps_lr)-1)]
    letter_spacing = float(np.percentile(gaps, 30)) if gaps else 0
    word_spacing   = float(np.percentile(gaps, 80)) if gaps else 0

    skel = skeletonize(thresh > 0)
    skel_u8 = img_as_ubyte(skel)
    ys, xs = np.where(skel_u8 > 0)
    stroke_len = sum(euclidean((xs[i],ys[i]),(xs[i+1],ys[i+1])) for i in range(len(xs)-1)) if len(xs)>1 else 0

    corners = cv2.goodFeaturesToTrack(thresh, maxCorners=500, qualityLevel=0.01, minDistance=6)
    corner_count = 0 if corners is None else len(corners)
    total_ink    = float(np.sum(thresh > 0))
    corner_density = corner_count / (total_ink + 1e-6)
    baseline_std = np.std([c[5][1] for c in comps]) / (mean_h + 1e-6) if comps else 0

    features = {
        "image_name": name,
        "mean_letter_height": mean_h, "std_letter_height": std_h,
        "mean_letter_width": mean_w,  "std_letter_width": std_w,
        "letter_height_cv": letter_height_cv,
        "proportion_consistency_std": proportion_consistency_std,
        "letter_spacing": letter_spacing, "word_spacing": word_spacing,
        "stroke_length_total": float(stroke_len),
        "corner_count": corner_count, "corner_density": corner_density,
        "num_components": len(comps), "total_ink_area": total_ink,
        "component_density": len(comps)/(total_ink+1e-6),
        "horizontal_regularity_baseline_std": float(baseline_std),
        "vertical_regularity_height_std": float(np.std(heights)),
        "margin_alignment_std": float(np.std(lefts)),
        "slant_angle_deg": 0.0
    }

    rule_label, rule_score = dysgraphia_screening(features)
    features["dysgraphia_label"]  = rule_label
    features["dysgraphia_score"]  = rule_score
    ml_label, ml_conf             = ml_predict(features)
    features["ml_prediction"]     = ml_label
    features["ml_confidence"]     = ml_conf

    return features, gray, thresh, skel_u8

@app.route("/")
def index():
    return render_template("index.html", model_ready=(ml_model is not None))

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")
    results, all_features = [], []

    for f in files:
        if not allowed_file(f.filename): continue
        name, ext = os.path.splitext(f.filename)

        if ext.lower() == ".pdf":
            doc = fitz.open(stream=f.read(), filetype="pdf")
            for i, page in enumerate(doc):
                pix    = page.get_pixmap(dpi=96)
                img    = pixmap_to_cv2(pix)
                h0, w0 = img.shape[:2]
                if w0 > 900:
                    scale = 800 / w0
                    img = cv2.resize(img, (800, int(h0*scale)), interpolation=cv2.INTER_AREA)
                sample = f"{name}_page{i+1}"
                feats, g, t, s = extract_features(img, sample)
                feats["source_file"] = f.filename
                for suffix, data in [("", img), ("_g", g), ("_t", t), ("_s", s)]:
                    imwrite_safe(os.path.join(UPLOAD_FOLDER if suffix=="" else PROCESSED_FOLDER,
                                              f"{sample}{suffix}.png"), data)
                all_features.append(feats)
                results.append({"sample_name": sample, "source_filename": f.filename,
                    "upload_url": url_for("static", filename=f"uploads/{sample}.png"),
                    "gray_url":   url_for("static", filename=f"processed/{sample}_g.png"),
                    "thresh_url": url_for("static", filename=f"processed/{sample}_t.png"),
                    "skel_url":   url_for("static", filename=f"processed/{sample}_s.png"),
                    "features": feats})
        else:
            uid  = uuid.uuid4().hex[:8]
            path = os.path.join(UPLOAD_FOLDER, uid + ext)
            f.save(path)
            img = cv2.imread(path)
            if img is not None:
                h0, w0 = img.shape[:2]
                if w0 > 900:
                    scale = 800 / w0
                    img = cv2.resize(img, (800, int(h0*scale)), interpolation=cv2.INTER_AREA)
            feats, g, t, s = extract_features(img, uid)
            feats["source_file"] = f.filename
            for suffix, folder, data in [("", UPLOAD_FOLDER, img), ("_g", PROCESSED_FOLDER, g),
                                          ("_t", PROCESSED_FOLDER, t), ("_s", PROCESSED_FOLDER, s)]:
                imwrite_safe(os.path.join(folder, f"{uid}{suffix}.png"), data)
            all_features.append(feats)
            results.append({"sample_name": uid, "source_filename": f.filename,
                "upload_url": url_for("static", filename=f"uploads/{uid}.png"),
                "gray_url":   url_for("static", filename=f"processed/{uid}_g.png"),
                "thresh_url": url_for("static", filename=f"processed/{uid}_t.png"),
                "skel_url":   url_for("static", filename=f"processed/{uid}_s.png"),
                "features": feats})

    pd.DataFrame(all_features).to_csv(os.path.join(PROCESSED_FOLDER, "features.csv"), index=False)
    return render_template("result.html", results=results,
                           csv_url=url_for("static", filename="processed/features.csv"),
                           model_ready=(ml_model is not None))

@app.route("/dashboard")
def dashboard():
    results_csv = os.path.join(CHARTS_DIR, "model_results.csv")
    model_results = None
    best_model_name = None
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        model_results = df.to_dict(orient="records")
        best_model_name = df.loc[df["CV Accuracy"].idxmax(), "Model"]

    chart_files = {
        "model_comparison":          "charts/model_comparison.png",
        "roc_curves":                "charts/roc_curves.png",
        "confusion_matrices":        "charts/confusion_matrices.png",
        "cv_score_distribution":     "charts/cv_score_distribution.png",
        "feature_importance_rf":     "charts/feature_importance_rf.png",
        "feature_relevance_selectkbest": "charts/feature_relevance_selectkbest.png",
        "correlation_heatmap":       "charts/correlation_heatmap.png",
        "class_distribution":        "charts/class_distribution.png",
    }
    charts = {k: url_for("static", filename=v)
              for k, v in chart_files.items()
              if os.path.exists(os.path.join("static", v))}

    report_text = None
    if os.path.exists("model_report.txt"):
        with open("model_report.txt") as f:
            report_text = f.read()

    return render_template("dashboard.html",
                           model_results=model_results,
                           best_model_name=best_model_name,
                           charts=charts,
                           report_text=report_text,
                           model_ready=(ml_model is not None))

if __name__ == "__main__":
    app.run(debug=True, port=8501)