import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# LOAD ADVANCED FEATURES
# ==========================================

file_path = "data/processed/wesad_advanced_features.csv"

df = pd.read_csv(file_path)

print("Dataset loaded!")
print("Total windows:", len(df))


# ==========================================
# FEATURES
# ==========================================

features = [
    "EDA_mean",
    "EDA_std",
    "EDA_min",
    "EDA_max",
    "EDA_range",
    "EDA_change_mean",
    "EDA_change_std",
    "EDA_slope",
    "EDA_variability_ratio",

    "BVP_mean",
    "BVP_std",
    "BVP_min",
    "BVP_max",
    "BVP_range",
    "BVP_change_mean",
    "BVP_change_std",

    "ACC_mean",
    "ACC_std",
    "ACC_min",
    "ACC_max",
    "ACC_range",
    "ACC_change_mean",
    "ACC_change_std",

    "TEMP_mean",
    "TEMP_std",
    "TEMP_min",
    "TEMP_max",
    "TEMP_range"
]


X = df[features]
y = df["stress"]


# ==========================================
# TRAIN FINAL MODEL
# ==========================================

print("Training final Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X, y)

print("Model training completed!")


# ==========================================
# SAVE MODEL
# ==========================================

model_path = "data/processed/mindshield_stress_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(model_path)