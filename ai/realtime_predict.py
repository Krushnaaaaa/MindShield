import pandas as pd
import numpy as np
import joblib


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model_path = "data/processed/mindshield_stress_model.pkl"

model = joblib.load(model_path)

print("MindShield model loaded successfully!")


# ==========================================
# LOAD WESAD DATA
# ==========================================

data_path = "data/processed/wesad_all_subjects.csv"

df = pd.read_csv(data_path)

print("WESAD data loaded!")
print("Total rows:", len(df))


# ==========================================
# SELECT ONE 30-SECOND WINDOW
# ==========================================

# 4 samples per second × 30 seconds = 120 samples

window = df.iloc[0:120].copy()

print("\nSelected window:")
print("Samples:", len(window))


# ==========================================
# FEATURE EXTRACTION
# ==========================================

features = {}


# ---------- EDA ----------

features["EDA_mean"] = window["EDA"].mean()
features["EDA_std"] = window["EDA"].std()
features["EDA_min"] = window["EDA"].min()
features["EDA_max"] = window["EDA"].max()
features["EDA_range"] = (
    window["EDA"].max() - window["EDA"].min()
)

eda_diff = window["EDA"].diff().dropna()

features["EDA_change_mean"] = eda_diff.mean()
features["EDA_change_std"] = eda_diff.std()

features["EDA_slope"] = (
    window["EDA"].iloc[-1] - window["EDA"].iloc[0]
) / len(window)

features["EDA_variability_ratio"] = (
    eda_diff.std() / (abs(eda_diff.mean()) + 1e-6)
)


# ---------- BVP ----------

features["BVP_mean"] = window["BVP"].mean()
features["BVP_std"] = window["BVP"].std()
features["BVP_min"] = window["BVP"].min()
features["BVP_max"] = window["BVP"].max()
features["BVP_range"] = (
    window["BVP"].max() - window["BVP"].min()
)

bvp_diff = window["BVP"].diff().dropna()

features["BVP_change_mean"] = bvp_diff.mean()
features["BVP_change_std"] = bvp_diff.std()


# ---------- ACC ----------

features["ACC_mean"] = window["ACC"].mean()
features["ACC_std"] = window["ACC"].std()
features["ACC_min"] = window["ACC"].min()
features["ACC_max"] = window["ACC"].max()
features["ACC_range"] = (
    window["ACC"].max() - window["ACC"].min()
)

acc_diff = window["ACC"].diff().dropna()

features["ACC_change_mean"] = acc_diff.mean()
features["ACC_change_std"] = acc_diff.std()


# ---------- TEMP ----------

features["TEMP_mean"] = window["TEMP"].mean()
features["TEMP_std"] = window["TEMP"].std()
features["TEMP_min"] = window["TEMP"].min()
features["TEMP_max"] = window["TEMP"].max()
features["TEMP_range"] = (
    window["TEMP"].max() - window["TEMP"].min()
)


# ==========================================
# CREATE MODEL INPUT
# ==========================================

feature_order = [
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


X = pd.DataFrame(
    [[features[f] for f in feature_order]],
    columns=feature_order
)


# ==========================================
# DISPLAY FEATURES
# ==========================================

print("\nExtracted Features:")
print(X.T)


# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(X)[0]

probability = model.predict_proba(X)[0]

stress_probability = probability[1]


# ==========================================
# RESULT
# ==========================================

print("\n===================================")
print("MINDSHIELD PREDICTION")
print("===================================")

if prediction == 1:
    print("Prediction: STRESS")
else:
    print("Prediction: NON-STRESS")

print(
    "Stress Probability:",
    round(stress_probability * 100, 2),
    "%"
)