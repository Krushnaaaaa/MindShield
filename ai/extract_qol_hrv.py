
import os
import numpy as np
import pandas as pd
import neurokit2 as nk

DATA_PATH = "data/raw/QoL_Stress"
OUTPUT_PATH = "data/processed/qol_stress_hrv.csv"

SAMPLING_RATE = 250

# Find ECG file
ecg_path = None

for root, dirs, files in os.walk(DATA_PATH):
    if "ecg.csv" in files:
        ecg_path = os.path.join(root, "ecg.csv")
        break

if ecg_path is None:
    raise FileNotFoundError("ecg.csv not found")

df = pd.read_csv(ecg_path)

results = []

for index, row in df.iterrows():
    try:
        # Convert waveform string into numerical signal
        signal = np.array(
            [float(x) for x in str(row["waveform_samples"]).split()]
        )

        # Clean ECG signal and detect R-peaks
        cleaned = nk.ecg_clean(
            signal,
            sampling_rate=SAMPLING_RATE
        )

        _, info = nk.ecg_peaks(
            cleaned,
            sampling_rate=SAMPLING_RATE
        )

        # Calculate HRV
        peaks = info["ECG_R_Peaks"]

        if len(peaks) < 3:
            print("Skipping recording:", index, "- insufficient peaks")
            continue

        hrv = nk.hrv_time(
            peaks,
            sampling_rate=SAMPLING_RATE,
            show=False
        )

        results.append({
            "participant_id": row["username"],
            "phase": row["phase"],
            "stress_label": 1 if row["phase"].lower() == "stress" else 0,
            "heart_rate": row["heart_rate"],
            "RMSSD": hrv["HRV_RMSSD"].iloc[0],
            "SDNN": hrv["HRV_SDNN"].iloc[0],
            "dataset_source": "QoL_Stress"
        })

        print("Processed recording:", index + 1)

    except Exception as e:
        print("Error in recording", index, ":", str(e))

# Save results
output = pd.DataFrame(results)

os.makedirs("data/processed", exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

print("\n========== HRV EXTRACTION COMPLETE ==========")
print("Successful recordings:", len(output))
print("Unique participants:", output["participant_id"].nunique())

if not output.empty:
    print("\nHRV summary:")
    print(output[["RMSSD", "SDNN", "heart_rate"]].describe())

print("\nSaved to:", OUTPUT_PATH)
