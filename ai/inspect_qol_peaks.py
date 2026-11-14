
import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path

base = Path("data/raw/QoL_Stress")

files = list(base.rglob("ecg.csv"))

if not files:
    raise FileNotFoundError("ecg.csv not found")

ecg = pd.read_csv(files[0])

hrv = pd.read_csv("data/processed/qol_stress_hrv.csv")

high_ids = hrv.loc[
    hrv["RMSSD"] > 200, "participant_id"
].unique()

for participant in high_ids:

    rows = ecg[ecg["username"] == participant]

    for _, row in rows.iterrows():

        raw = str(row["waveform_samples"]).split()

        # Do not remove missing samples from a waveform:
        # that would change the timing of the signal.
        try:
            signal = np.array(raw, dtype=float)
        except ValueError:
            print("Skipped incomplete waveform:", participant)
            continue

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

        print("\nParticipant:", participant)
        print("Phase:", row["phase"])
        print("Detected R-peaks:", len(peaks))

        if len(rr):
            print("Median RR (ms):", round(np.median(rr), 2))
            print("Minimum RR (ms):", round(np.min(rr), 2))
            print("Maximum RR (ms):", round(np.max(rr), 2))
            print(
                "Intervals outside 300–2000 ms:",
                np.sum((rr < 300) | (rr > 2000))
            )

