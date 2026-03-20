from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pickle
import json
import numpy as np
import pandas as pd
import os
import sqlite3
import random
import string
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "gigpulse.db")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.secret_key = "gigpulse_hr_secret_2024"

HR_CREDENTIALS = {
    "JOSH": "Joshi@2005",
    "admin@gigpulse.in": "hr2024",
}

# DATABASE

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id     TEXT UNIQUE NOT NULL,
            employee_name TEXT,
            employee_id   TEXT,
            submitted_at  TEXT,
            payload       TEXT,
            prediction    TEXT,
            rf_prediction TEXT,
            risk_cluster  TEXT,
            confidence    REAL,
            probabilities TEXT,
            result        TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database ready:", DB_PATH)

init_db()

# Load ML models & assets

with open("models/xgb_model.pkl",        "rb") as f: xgb_model     = pickle.load(f)
with open("models/rf_model.pkl",          "rb") as f: rf_model      = pickle.load(f)
with open("models/kmeans_model.pkl",      "rb") as f: kmeans        = pickle.load(f)
with open("models/scaler_kmeans.pkl",     "rb") as f: scaler_kmeans = pickle.load(f)
with open("models/label_encoders.pkl",    "rb") as f: label_enc     = pickle.load(f)
with open("models/encode_maps.json")        as f: encode_maps   = json.load(f)
with open("models/feature_names.json")      as f: feature_names = json.load(f)
with open("models/feature_importance.json") as f: feat_imp      = json.load(f)
with open("models/cluster_risk_map.json")   as f: cluster_risk  = json.load(f)

# Load the exact column list the model was trained on
# columns.pkl contains the final list after encoding (e.g. 73 columns)
TRAINED_COLUMNS = None
if os.path.exists("models/columns.pkl"):
    with open("models/columns.pkl", "rb") as f:
        TRAINED_COLUMNS = pickle.load(f)
    print(f" Loaded trained columns: {len(TRAINED_COLUMNS)} features")
else:
    print("columns.pkl not found — will use feature_names.json")

cluster_risk    = {int(k): v for k, v in cluster_risk.items()}
IDENTITY_FIELDS = {"employee_name", "employee_id", "_worker_id", "_submitted_at", "_fields"}


# Helpers

def hr_logged_in():
    return session.get("hr_auth") is True

def generate_worker_id():
    return "W-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

def run_prediction(data):
    """
    Build input row matching exactly how the model was trained.

    Strategy A — columns.pkl exists (model trained with get_dummies / one-hot):
        Build a DataFrame with all original columns, label-encode categoricals,
        then align to TRAINED_COLUMNS order (fills missing with 0).

    Strategy B — columns.pkl missing (model trained with LabelEncoder only):
        Use encode_maps to integer-encode each field in feature_names order.
    """
    classes = label_enc["future_gig_retention"].classes_

    # Strip identity/meta fields 
    clean = {k: v for k, v in data.items() if k not in IDENTITY_FIELDS}

    if TRAINED_COLUMNS is not None:
        # Strategy A: one-hot / get_dummies pipeline
        # 1. Build a 1-row DataFrame with raw values
        df = pd.DataFrame([clean])

        # 2. Label-encode every categorical column using encode_maps
        for col in df.columns:
            if col in encode_maps:
                mapping = encode_maps[col]
                df[col] = df[col].astype(str).map(
                    lambda x, m=mapping: m.get(x, 0)
                )
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 3. Align to exact trained column order, fill any missing with 0
        df = df.reindex(columns=TRAINED_COLUMNS, fill_value=0)
        row_array = df.values.astype(float)

    else:
        # Strategy B: simple LabelEncoder pipeline 
        row = []
        for feat in feature_names:
            if feat in IDENTITY_FIELDS:
                continue
            val = clean.get(feat)
            if val is None:
                raise ValueError(f"Missing field: {feat}")
            if feat in encode_maps:
                mapping = encode_maps[feat]
                str_val = str(val)
                if str_val not in mapping:
                    raise ValueError(f"Unknown value '{val}' for field '{feat}'")
                row.append(mapping[str_val])
            else:
                try:
                    row.append(float(val))
                except:
                    raise ValueError(f"Field '{feat}' must be numeric, got '{val}'")
        row_array = np.array(row).reshape(1, -1)

    # Run models 
    xgb_idx   = int(xgb_model.predict(row_array)[0])
    xgb_proba = xgb_model.predict_proba(row_array)[0]
    xgb_label = classes[xgb_idx]

    rf_idx    = int(rf_model.predict(row_array)[0])
    rf_label  = classes[rf_idx]

    row_scaled = scaler_kmeans.transform(row_array)
    cluster_id = int(kmeans.predict(row_scaled)[0])
    risk_label = cluster_risk.get(cluster_id, "Unknown")

    proba_breakdown = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(classes, xgb_proba)
    }

    return {
        "prediction":    xgb_label,
        "rf_prediction": rf_label,
        "risk_cluster":  risk_label,
        "cluster_id":    cluster_id,
        "probabilities": proba_breakdown,
        "confidence":    round(float(max(xgb_proba)) * 100, 1),
    }

# EMPLOYEE ROUTES
@app.route("/")
@app.route("/worker_form")
def worker_form():
    return render_template("worker_form.html")


@app.route("/thankyou")
def thankyou():
    return render_template("thankyou.html")


@app.route("/submit_form", methods=["POST"])
def submit_form():
    """
    1. Receive form JSON from employee
    2. Run ML prediction
    3. Save everything to SQLite DB
    4. Return worker_id to client
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No data received"}), 400

        worker_id    = generate_worker_id()
        submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Run ML prediction
        result = run_prediction(data)

        # Save to database
        conn = get_db()
        conn.execute("""
            INSERT INTO submissions
              (worker_id, employee_name, employee_id, submitted_at,
               payload, prediction, rf_prediction, risk_cluster,
               confidence, probabilities, result)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            worker_id,
            data.get("employee_name", "—"),
            data.get("employee_id",   "—"),
            submitted_at,
            json.dumps(data),
            result["prediction"],
            result["rf_prediction"],
            result["risk_cluster"],
            result["confidence"],
            json.dumps(result["probabilities"]),
            json.dumps(result),
        ))
        conn.commit()
        conn.close()

        print(f" Saved submission {worker_id} | {data.get('employee_name')} | {result['prediction']}")

        return jsonify({
            "success":   True,
            "worker_id": worker_id,
            "fields":    len([k for k in data if not k.startswith("_")]),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# HR / ADMIN ROUTES


@app.route("/hr_login", methods=["GET"])
def hr_login_page():
    if hr_logged_in():
        return redirect(url_for("predictions_page"))
    return render_template("hr_login.html")


@app.route("/hr_auth", methods=["POST"])
def hr_auth():
    data     = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    expected = HR_CREDENTIALS.get(username)
    if expected and expected == password:
        session["hr_auth"] = True
        session["hr_user"] = username
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid credentials"}), 401


@app.route("/hr_logout")
def hr_logout():
    session.clear()
    return redirect(url_for("hr_login_page"))


@app.route("/predictions")
def predictions_page():
    if not hr_logged_in():
        return redirect(url_for("hr_login_page"))
    return render_template("predictions.html")


#  HR: get all submissions 
@app.route("/get_submissions", methods=["GET"])
def get_submissions():
    if not hr_logged_in():
        return jsonify({"error": "Unauthorised"}), 401
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT worker_id, employee_name, employee_id,
                   submitted_at, prediction, confidence, risk_cluster
            FROM submissions
            ORDER BY id DESC
        """).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# HR: get single submission 
@app.route("/get_submission/<worker_id>", methods=["GET"])
def get_submission(worker_id):
    if not hr_logged_in():
        return jsonify({"error": "Unauthorised"}), 401
    try:
        conn = get_db()
        row  = conn.execute(
            "SELECT * FROM submissions WHERE worker_id = ?", (worker_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "Submission not found"}), 404
        rec = dict(row)
        rec["payload"] = json.loads(rec["payload"])
        rec["result"]  = json.loads(rec["result"])
        rec["probabilities"] = json.loads(rec["probabilities"])
        return jsonify(rec)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#  HR: delete a submission
@app.route("/delete_submission/<worker_id>", methods=["DELETE"])
def delete_submission(worker_id):
    if not hr_logged_in():
        return jsonify({"error": "Unauthorised"}), 401
    try:
        conn = get_db()
        conn.execute("DELETE FROM submissions WHERE worker_id = ?", (worker_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# OTHER APIs

@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    top10 = list(feat_imp.items())[:10]
    return jsonify({
        "labels": [f[0].replace("_", " ").title() for f in top10],
        "values": [round(f[1] * 100, 2) for f in top10],
    })


@app.route("/debug_columns", methods=["GET"])
def debug_columns():
    """HR-only: inspect what columns the model expects vs what form sends."""
    if not hr_logged_in():
        return jsonify({"error": "Unauthorised"}), 401
    return jsonify({
        "trained_columns_count": len(TRAINED_COLUMNS) if TRAINED_COLUMNS is not None else "N/A",
        "trained_columns":       list(TRAINED_COLUMNS) if TRAINED_COLUMNS is not None else [],
        "feature_names_count":   len(feature_names),
        "feature_names":         feature_names,
        "encode_maps_keys":      list(encode_maps.keys()),
    })


@app.route("/hr_status", methods=["GET"])
def hr_status():
    return jsonify({
        "authenticated": hr_logged_in(),
        "user": session.get("hr_user", None),
    })

# Run

if __name__ == "__main__":
    print("=" * 55)
    print("  GigPulse — Workforce Retention App")
    print("=" * 55)
    print("  Employee form  →  http://127.0.0.1:5000/")
    print("  Thank you page →  http://127.0.0.1:5000/thankyou")
    print("  HR login       →  http://127.0.0.1:5000/hr_login")
    print("  HR dashboard   →  http://127.0.0.1:5000/predictions")
    print(f"  Database       →  {DB_PATH}")
    print("=" * 55)
if __name__ == "__main__":
    app.run(debug=True)
@app.route("/ping")
def ping():
    return "OK"
