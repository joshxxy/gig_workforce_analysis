import pandas as pd
import numpy as np
import pickle
import os
import json

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  GIG WORKFORCE RETENTION - ML MODEL TRAINING")
print("=" * 60)

#1.LOADING THE DATA
df = pd.read_excel("Data/Gig Workforce Retention Analysis.xlsx")
print("First 5 Rows:")
print(df.head())
print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# 2. ENCODE CATEGORICAL DATA

label_encoders = {}
encode_maps    = {}

df_encoded = df.copy()

# Convert all string-type columns to plain Python str first
for col in df_encoded.columns:
    if str(df_encoded[col].dtype) in ['str', 'string', 'object']:
        df_encoded[col] = df_encoded[col].astype(str)

# Now encode
for column in df_encoded.columns:
    if df_encoded[column].dtype == object:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
        encode_maps[column] = {
            label: int(idx) for idx, label in enumerate(le.classes_)
        }

print("\n✅ Encoding complete")

with open("models/encode_maps.json", "w") as f:
    json.dump(encode_maps, f, indent=2)
print("✅ Encode maps saved -> models/encode_maps.json")


# 3. DROP UNUSED COLUMNS

drop_cols = [
    "primary_retention_reason",
    "likely_attrition_reason",
    "expected_gig_duration"
]
df_encoded = df_encoded.drop(
    columns=[c for c in drop_cols if c in df_encoded.columns]
)

# 4. FEATURES & TARGET

X = df_encoded.drop("future_gig_retention", axis=1)
y = df_encoded["future_gig_retention"]

feature_names = X.columns.tolist()

with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print(f"\n✅ Features used: {len(feature_names)}")
print(f"   Target classes: {label_encoders['future_gig_retention'].classes_}")


# 5. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train size: {len(X_train)} | Test size: {len(X_test)}")


# 6. LOGISTIC REGRESSION (Baseline)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=2000, random_state=42)
log_model.fit(X_train_scaled, y_train)
log_pred  = log_model.predict(X_test_scaled)
log_acc   = accuracy_score(y_test, log_pred)
print(f"\n📊 Logistic Regression Accuracy : {log_acc*100:.2f}%")


# 7. RANDOM FOREST (GridSearchCV)

print("\n⏳ Training Random Forest with GridSearchCV ...")
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [5, 10, None],
    "min_samples_split": [2, 5]
}
grid    = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"✅ Random Forest Accuracy        : {rf_acc*100:.2f}%")
print(f"   Best Params: {grid.best_params_}")


# 8. XGBOOST
print("\n⏳ Training XGBoost ...")
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)
print(f"✅ XGBoost Accuracy               : {xgb_acc*100:.2f}%")

# 9. MODEL COMPARISON & BEST MODEL

print("\n" + "=" * 50)
print("  MODEL COMPARISON")
print("=" * 50)
print(f"  Logistic Regression : {log_acc*100:.2f}%")
print(f"  Random Forest       : {rf_acc*100:.2f}%")
print(f"  XGBoost             : {xgb_acc*100:.2f}%")

best_acc = max(log_acc, rf_acc, xgb_acc)

if xgb_acc == best_acc:
    best_name = "XGBoost"
    best_pred = xgb_pred
elif rf_acc == best_acc:
    best_name = "Random Forest"
    best_pred = rf_pred
else:
    best_name = "Logistic Regression"
    best_pred = log_pred

print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}%)")
print(f"\n📋 Classification Report ({best_name}):")
print(classification_report(
    y_test, best_pred,
    target_names=label_encoders['future_gig_retention'].classes_
))


# 10. FEATURE IMPORTANCE (XGBoost)

importances     = xgb_model.feature_importances_
feat_imp        = dict(zip(feature_names, [round(float(v), 4) for v in importances]))
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

with open("models/feature_importance.json", "w") as f:
    json.dump(feat_imp_sorted, f, indent=2)
print("✅ Feature importance saved -> models/feature_importance.json")


# 11. KMEANS CLUSTERING (Supporting Model)

print("\n⏳ Training KMeans Clustering (Supporting Model) ...")
scaler_kmeans = StandardScaler()
X_scaled_all  = scaler_kmeans.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled_all)

df_encoded["cluster"]     = kmeans.labels_
cluster_retention_avg     = df_encoded.groupby("cluster")["future_gig_retention"].mean()
sorted_clusters           = cluster_retention_avg.sort_values(ascending=False).index.tolist()
risk_labels               = ["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"]
cluster_risk_map          = {int(c): risk_labels[i] for i, c in enumerate(sorted_clusters)}

with open("models/cluster_risk_map.json", "w") as f:
    json.dump(cluster_risk_map, f)
print(f"✅ KMeans trained | Cluster Risk Map: {cluster_risk_map}")


# 12. SAVE ALL MODELS & SCALERS

pickle.dump(xgb_model,      open("models/xgb_model.pkl",      "wb"))
pickle.dump(best_rf,        open("models/rf_model.pkl",        "wb"))
pickle.dump(kmeans,         open("models/kmeans_model.pkl",    "wb"))
pickle.dump(scaler,         open("models/scaler.pkl",          "wb"))
pickle.dump(scaler_kmeans,  open("models/scaler_kmeans.pkl",   "wb"))
pickle.dump(label_encoders, open("models/label_encoders.pkl",  "wb"))

print("\n✅ All models saved to /models/")
print("\n🎉 Training Complete! Now run:  python app.py")
print("=" * 60)