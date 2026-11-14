
import pandas as pd

hrv = pd.read_csv(
    "data/processed/qol_stress_hrv_qc.csv"
)

eda = pd.read_csv(
    "data/processed/qol_stress_eda.csv"
)

# Aggregate EDA to participant + phase level
eda_summary = (
    eda.groupby(["participant_id", "phase"])
    .agg(
        scl_mean=("scl_avg", "mean"),
        scl_std=("scl_avg", "std"),
        activation_mean=("activation", "mean"),
        activation_std=("activation", "std"),
        eda_samples=("scl_avg", "count")
    )
    .reset_index()
)

# Keep only required ECG features
hrv_features = hrv[
    [
        "participant_id",
        "phase",
        "RMSSD",
        "SDNN",
        "heart_rate",
        "quality_flag"
    ]
]

# Merge by participant and experimental phase
combined = pd.merge(
    hrv_features,
    eda_summary,
    on=["participant_id", "phase"],
    how="inner"
)

combined["stress_label"] = (
    combined["phase"].map({
        "relax": 0,
        "stress": 1
    })
)

combined["dataset_source"] = "QoL_Stress"

combined.to_csv(
    "data/processed/qol_stress_combined.csv",
    index=False
)

print("COMBINATION COMPLETE")
print("Total rows:", len(combined))
print("Unique participants:", combined["participant_id"].nunique())

print("\nLabel distribution:")
print(combined["stress_label"].value_counts())

print("\nMissing values:")
print(combined.isnull().sum())

print("\nPreview:")
print(combined.head())

print("\nSaved: data/processed/qol_stress_combined.csv")
