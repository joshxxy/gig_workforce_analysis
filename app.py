from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
import pandas as pd


app = Flask(__name__)

# ── Load all saved models & assets ──────────────────────────────
with open("models/xgb_model.pkl",        "rb") as f: xgb_model     = pickle.load(f)
with open("models/rf_model.pkl",          "rb") as f: rf_model      = pickle.load(f)
with open("models/kmeans_model.pkl",      "rb") as f: kmeans        = pickle.load(f)
with open("models/scaler_kmeans.pkl",     "rb") as f: scaler_kmeans = pickle.load(f)
with open("models/label_encoders.pkl",    "rb") as f: label_enc     = pickle.load(f)
with open("models/encode_maps.json")         as f: encode_maps   = json.load(f)
with open("models/feature_names.json")       as f: feature_names  = json.load(f)
with open("models/feature_importance.json")  as f: feat_imp       = json.load(f)
with open("models/cluster_risk_map.json")    as f: cluster_risk   = json.load(f)

# Fix cluster_risk keys (JSON keys are strings, need int)
cluster_risk = {int(k): v for k, v in cluster_risk.items()}

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_options", methods=["GET"])
def get_options():
    """Return all dropdown options for the form."""
    options = {}
    for col, mapping in encode_maps.items():
        if col not in ["future_gig_retention", "primary_retention_reason",
                        "likely_attrition_reason", "expected_gig_duration"]:
            options[col] = list(mapping.keys())
    return jsonify(options)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form data → run XGBoost + KMeans → return prediction."""
    try:
        data = request.get_json()

        # Build input row in correct feature order
        row = []
        for feat in feature_names:
            val = data.get(feat)
            if val is None:
                return jsonify({"error": f"Missing field: {feat}"}), 400

            # Encode categorical
            if feat in encode_maps:
                mapping = encode_maps[feat]
                if str(val) not in mapping:
                    return jsonify({"error": f"Unknown value '{val}' for '{feat}'"}), 400
                row.append(mapping[str(val)])
            else:
                row.append(float(val))

        row_array = np.array(row).reshape(1, -1)

        # ── Primary Model: XGBoost Prediction ──
        xgb_pred_idx = int(xgb_model.predict(row_array)[0])
        xgb_proba    = xgb_model.predict_proba(row_array)[0]
        retention_classes = label_enc["future_gig_retention"].classes_
        prediction_label  = retention_classes[xgb_pred_idx]

        # RF prediction (secondary check)
        rf_pred_idx   = int(rf_model.predict(row_array)[0])
        rf_label      = retention_classes[rf_pred_idx]

        # ── Supporting Model: KMeans Cluster ──
        row_scaled    = scaler_kmeans.transform(row_array)
        cluster_id    = int(kmeans.predict(row_scaled)[0])
        risk_label    = cluster_risk.get(cluster_id, "Unknown")

        # ── Top 5 Feature Importances ──
        top_features = list(feat_imp.items())[:5]

        # ── Probability breakdown ──
        proba_breakdown = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(retention_classes, xgb_proba)
        }

        return jsonify({
            "prediction":       prediction_label,
            "rf_prediction":    rf_label,
            "risk_cluster":     risk_label,
            "cluster_id":       cluster_id,
            "probabilities":    proba_breakdown,
            "top_features":     top_features,
            "confidence":       round(float(max(xgb_proba)) * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    """Return feature importance data for chart."""
    top10 = list(feat_imp.items())[:10]
    return jsonify({
        "labels": [f[0].replace("_", " ").title() for f in top10],
        "values": [round(f[1] * 100, 2) for f in top10]
    })


if __name__ == "__main__":
    print("🚀 Starting Gig Workforce Retention App...")
    print("🌐 Open: http://127.0.0.1:5000")
    app.run(debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)