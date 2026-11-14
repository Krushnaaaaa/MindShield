
import pandas as pd
import numpy as np

import neurokit2 as nk

df = pd.read_csv(
    "data/processed/qol_stress_hrv.csv"
)

print("Checking high-RMSSD recordings...\n")

for _, row in df[df["RMSSD"] > 200].iterrows():

    print("Participant:", row["participant_id"])
    print("Phase:", row["phase"])
    print("RMSSD:", round(row["RMSSD"], 2))
    print("SDNN:", round(row["SDNN"], 2))
    print("Heart rate:", row["heart_rate"])
    print("-" * 40)

print("\nOverall phase comparison:")
print(
    df.groupby("phase")[["RMSSD", "SDNN", "heart_rate"]]
    .median()
    .round(2)
)
