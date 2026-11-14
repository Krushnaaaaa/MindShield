
import pandas as pd

df = pd.read_csv("data/processed/qol_stress_hrv.csv")

print("TOTAL RECORDINGS:", len(df))
print("UNIQUE PARTICIPANTS:", df["participant_id"].nunique())

print("\nPHASE DISTRIBUTION:")
print(df["phase"].value_counts())

print("\nHRV STATISTICS BY PHASE:")
print(df.groupby("phase")[["RMSSD", "SDNN", "heart_rate"]].agg(
    ["count", "mean", "median", "min", "max"]
).round(2))

print("\nRECORDINGS WITH RMSSD ABOVE 200:")
print(
    df[df["RMSSD"] > 200][
        ["participant_id", "phase", "RMSSD", "SDNN", "heart_rate"]
    ].to_string(index=False)
)
