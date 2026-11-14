
import pandas as pd
import numpy as np

from pathlib import Path
import neurokit2 as nk

base = Path("data/raw/QoL_Stress")
ecg_file = list(base.rglob("ecg.csv"))[0]

ecg = pd.read_csv(ecg_file)
hrv = pd.read_csv("data/processed/qol_stress_hrv.csv")

results = []

for _, row in hrv.iterrows():

    participant = row["participant_id"]
    phase = row["phase"]

    match = ecg[
        (ecg["username"] == participant) &
        (ecg["phase"] == phase)
    ]

    if match.empty:
        continue

    raw = str(match.iloc[0]["waveform_samples"]).split()

    try:
        signal = np.array(raw, dtype=float)

        cleaned = nk.ecg_clean(
            signal,
            sampling_rate=250
        )

        _, info = nk.ecg_peaks(
            cleaned,
            sampling_rate=250
        )

        peaks = info["ECG_R_Peaks"]
        rr = np.diff(peaks) / 250 * 1000

        if len(rr) < 2:
            continue

        invalid_intervals = np.sum(
            (rr < 300) | (rr > 2000)
        )

        result = row.to_dict()
        result["detected_peaks"] = len(peaks)
        result["invalid_rr_intervals"] = invalid_intervals
        result["quality_flag"] = (
            "review" if invalid_intervals > 0 else "unflagged"
        )

        results.append(result)

    except (ValueError, TypeError):
        print("Incomplete waveform:", participant, phase)

output = pd.DataFrame(results)

output.to_csv(
    "data/processed/qol_stress_hrv_qc.csv",
    index=False
)

print("\nQUALITY CHECK COMPLETE")
print("Retained recordings:", len(output))
print("\nQuality flags:")
print(output["quality_flag"].value_counts())

print("\nSaved: data/processed/qol_stress_hrv_qc.csv")
