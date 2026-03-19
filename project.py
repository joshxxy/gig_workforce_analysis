import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# SETUP
print("=" * 60)
print(" GIG WORKFORCE RETENTION - ADVANCED ML TRAINING")
print("=" * 60)

os.makedirs("models", exist_ok=True)

# 1. LOAD DATA

df = pd.read_excel("Data/Gig Workforce Retention Analysis.xlsx")
print("First 5 Rows:")
print(df.head())
print(f"\n Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# 2. CLEAN DATA

df = df.dropna()

# 3. ONE-HOT ENCODING (IMPORTANT IMPROVEMENT)

df_encoded = pd.get_dummies(df, drop_first=True)

print("One-hot encoding applied")

# 4. DEFINE TARGET


target_col = [col for col in df_encoded.columns if "future_gig_retention" in col][0]

y = df_encoded[target_col]
X = df_encoded.drop(target_col, axis=1)

feature_names = X.columns.tolist()

# Saving feature columns for frontend
pickle.dump(feature_names, open("models/columns.pkl", "wb"))

print(f"\nFeatures: {len(feature_names)}")

# 5. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. HANDLE CLASS IMBALANCE

counter = Counter(y)
scale_pos_weight = counter[0] / counter[1]

print(f"\nClass balance: {counter}")

# 7. LOGISTIC REGRESSION

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_acc  = accuracy_score(y_test, log_pred)

print(f"\n Logistic Accuracy: {log_acc*100:.2f}%")

# 8. RANDOM FOREST (TUNED)

print("\n Training Random Forest...")

rf = RandomForestClassifier()

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}

rf_grid = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_model = rf_grid.best_estimator_

rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)

print(f" Random Forest Accuracy: {rf_acc*100:.2f}%")

# 9. XGBOOST (BEST MODEL)

print("\n Training Optimized XGBoost...")

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_params = {
    "n_estimators": [200, 300],
    "max_depth": [4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "scale_pos_weight": [scale_pos_weight]
}

xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, n_jobs=-1)

xgb_grid.fit(X_train, y_train)

xgb_model = xgb_grid.best_estimator_

xgb_pred = xgb_model.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)

print(f"XGBoost Accuracy: {xgb_acc*100:.2f}%")
print("Best Params:", xgb_grid.best_params_)

# AUC Score 
auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])
print(f"AUC Score: {auc:.3f}")

# 10. MODEL COMPARISON

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

print(f"Logistic Regression : {log_acc*100:.2f}%")
print(f"Random Forest       : {rf_acc*100:.2f}%")
print(f"XGBoost             : {xgb_acc*100:.2f}%")

best_model = xgb_model

print("\n Best Model: XGBoost")

print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

# 11. FEATURE IMPORTANCE


importances = xgb_model.feature_importances_

feat_imp = {
    feature: float(value)   
    for feature, value in zip(feature_names, importances)
}


feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

with open("models/feature_importance.json", "w") as f:
    json.dump(feat_imp, f, indent=2)

print(" Feature importance saved successfully")

# 12. KMEANS CLUSTERING

print("\n Training KMeans...")

scaler_k = StandardScaler()
X_scaled = scaler_k.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

clusters = kmeans.labels_

df_encoded["cluster"] = clusters

cluster_avg = df_encoded.groupby("cluster")[target_col].mean()

sorted_clusters = cluster_avg.sort_values(ascending=False).index.tolist()

risk_labels = ["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"]

cluster_risk_map = {int(c): risk_labels[i] for i, c in enumerate(sorted_clusters)}

with open("models/cluster_risk_map.json", "w") as f:
    json.dump(cluster_risk_map, f)

print(" KMeans clustering complete")

# 13. SAVE MODELS

pickle.dump(xgb_model, open("models/xgb_model.pkl", "wb"))
pickle.dump(rf_model, open("models/rf_model.pkl", "wb"))
pickle.dump(kmeans, open("models/kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(scaler_k, open("models/scaler_kmeans.pkl", "wb"))

print("\n All models saved in /models/")

print("\n TRAINING COMPLETE — Run app.py now!")
print("=" * 60)
import pickle
trained_columns = X_train.columns.tolist()  # or X.columns.tolist()
pickle.dump(trained_columns, open("models/columns.pkl", "wb"))
print(f" Saved columns.pkl — {len(trained_columns)} features")