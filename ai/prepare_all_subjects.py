import pickle
import numpy as np
import pandas as pd
import os

# ==========================================
# SETTINGS
# ==========================================

WESAD_PATH = "/Users/raj/Desktop/WESAD"

OUTPUT_PATH = "data/processed/wesad_all_subjects.csv"

# WESAD sampling rates
ACC_FS = 32
BVP_FS = 64
EDA_FS = 4
TEMP_FS = 4
LABEL_FS = 700

# Subjects available in your WESAD folder
subjects = [
    "S2", "S3", "S4", "S5",
    "S6", "S7", "S8", "S9",
    "S10", "S11", "S13", "S14",
    "S15", "S16", "S17"
]

# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs("data/processed", exist_ok=True)

all_data = []

# ==========================================
# PROCESS EACH SUBJECT
# ==========================================

for subject in subjects:

    file_path = os.path.join(
        WESAD_PATH,
        subject,
        f"{subject}.pkl"
    )

    print(f"\nProcessing {subject}...")

    # --------------------------------------
    # Load subject
    # --------------------------------------

    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    wrist = data["signal"]["wrist"]

    acc = wrist["ACC"]
    bvp = wrist["BVP"].flatten()
    eda = wrist["EDA"].flatten()
    temp = wrist["TEMP"].flatten()

    labels = data["label"]

    # --------------------------------------
    # Create time arrays
    # --------------------------------------

    acc_time = np.arange(len(acc)) / ACC_FS
    bvp_time = np.arange(len(bvp)) / BVP_FS
    eda_time = np.arange(len(eda)) / EDA_FS
    temp_time = np.arange(len(temp)) / TEMP_FS
    label_time = np.arange(len(labels)) / LABEL_FS

    # --------------------------------------
    # ACC magnitude
    # --------------------------------------

    acc_magnitude = np.sqrt(
        acc[:, 0] ** 2 +
        acc[:, 1] ** 2 +
        acc[:, 2] ** 2
    )

    # --------------------------------------
    # Find common time range
    # --------------------------------------

    start_time = max(
        acc_time[0],
        bvp_time[0],
        eda_time[0],
        temp_time[0],
        label_time[0]
    )

    end_time = min(
        acc_time[-1],
        bvp_time[-1],
        eda_time[-1],
        temp_time[-1],
        label_time[-1]
    )

    # --------------------------------------
    # Common timeline
    # --------------------------------------

    common_time = np.arange(
        start_time,
        end_time,
        1 / EDA_FS
    )

    # --------------------------------------
    # Interpolate signals
    # --------------------------------------

    acc_common = np.interp(
        common_time,
        acc_time,
        acc_magnitude
    )

    bvp_common = np.interp(
        common_time,
        bvp_time,
        bvp
    )

    eda_common = np.interp(
        common_time,
        eda_time,
        eda
    )

    temp_common = np.interp(
        common_time,
        temp_time,
        temp
    )

    # --------------------------------------
    # Match labels
    # --------------------------------------

    label_indices = np.searchsorted(
        label_time,
        common_time
    )

    label_indices = np.clip(
        label_indices,
        0,
        len(labels) - 1
    )

    label_common = labels[label_indices]

    # --------------------------------------
    # Create dataframe
    # --------------------------------------

    df = pd.DataFrame({
        "subject": subject,
        "time": common_time,
        "ACC": acc_common,
        "BVP": bvp_common,
        "EDA": eda_common,
        "TEMP": temp_common,
        "label": label_common
    })

    # --------------------------------------
    # Remove undefined label
    # --------------------------------------

    df = df[df["label"] != 0].copy()

    # --------------------------------------
    # Convert to stress / non-stress
    # --------------------------------------

    df["stress"] = df["label"].apply(
        lambda x: 1 if x in [2, 7] else 0
    )

    # --------------------------------------
    # Remove original label
    # --------------------------------------

    df.drop(columns=["label"], inplace=True)

    all_data.append(df)

    print(
        f"{subject} completed → {len(df)} rows"
    )

# ==========================================
# COMBINE ALL SUBJECTS
# ==========================================

final_df = pd.concat(
    all_data,
    ignore_index=True
)

# ==========================================
# SAVE
# ==========================================

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("ALL SUBJECTS PROCESSED!")
print("===================================")

print("Total rows:", len(final_df))

print("\nSubjects:")
print(final_df["subject"].unique())

print("\nStress distribution:")
print(final_df["stress"].value_counts())

print("\nSaved file:")
print(OUTPUT_PATH)