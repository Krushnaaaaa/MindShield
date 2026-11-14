import pandas as pd
import numpy as np


# ==========================================
# SETTINGS
# ==========================================

INPUT_FILE = "data/processed/wesad_all_subjects.csv"
OUTPUT_FILE = "data/processed/wesad_advanced_features.csv"

WINDOW_SIZE = 30
SAMPLING_RATE = 4

WINDOW_SAMPLES = WINDOW_SIZE * SAMPLING_RATE


# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded!")
print("Total rows:", len(df))


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(window):

    features = {}

    # ======================================
    # EDA FEATURES
    # ======================================

    eda = window["EDA"].values

    features["EDA_mean"] = np.mean(eda)
    features["EDA_std"] = np.std(eda)
    features["EDA_min"] = np.min(eda)
    features["EDA_max"] = np.max(eda)

    features["EDA_range"] = (
        np.max(eda) - np.min(eda)
    )

    # Average change in EDA
    eda_diff = np.diff(eda)

    features["EDA_change_mean"] = np.mean(
        np.abs(eda_diff)
    )

    features["EDA_change_std"] = np.std(
        eda_diff
    )

    # EDA slope
    x = np.arange(len(eda))

    if len(eda) > 1:
        features["EDA_slope"] = np.polyfit(
            x,
            eda,
            1
        )[0]
    else:
        features["EDA_slope"] = 0


    # ======================================
    # BVP FEATURES
    # ======================================

    bvp = window["BVP"].values

    features["BVP_mean"] = np.mean(bvp)
    features["BVP_std"] = np.std(bvp)
    features["BVP_min"] = np.min(bvp)
    features["BVP_max"] = np.max(bvp)

    features["BVP_range"] = (
        np.max(bvp) - np.min(bvp)
    )

    # BVP signal variability
    bvp_diff = np.diff(bvp)

    features["BVP_change_mean"] = np.mean(
        np.abs(bvp_diff)
    )

    features["BVP_change_std"] = np.std(
        bvp_diff
    )


    # ======================================
    # ACC FEATURES
    # ======================================

    acc = window["ACC"].values

    features["ACC_mean"] = np.mean(acc)
    features["ACC_std"] = np.std(acc)
    features["ACC_min"] = np.min(acc)
    features["ACC_max"] = np.max(acc)

    features["ACC_range"] = (
        np.max(acc) - np.min(acc)
    )

    # Movement intensity
    acc_diff = np.diff(acc)

    features["ACC_change_mean"] = np.mean(
        np.abs(acc_diff)
    )

    features["ACC_change_std"] = np.std(
        acc_diff
    )


    # ======================================
    # TEMP FEATURES
    # ======================================

    temp = window["TEMP"].values

    features["TEMP_mean"] = np.mean(temp)
    features["TEMP_std"] = np.std(temp)
    features["TEMP_min"] = np.min(temp)
    features["TEMP_max"] = np.max(temp)

    features["TEMP_range"] = (
        np.max(temp) - np.min(temp)
    )


    # ======================================
    # COMBINED FEATURES
    # ======================================

    # Ratio between EDA variability
    # and its average level
    if features["EDA_mean"] != 0:

        features["EDA_variability_ratio"] = (
            features["EDA_std"]
            / abs(features["EDA_mean"])
        )

    else:

        features["EDA_variability_ratio"] = 0


    return features


# ==========================================
# PROCESS SUBJECTS
# ==========================================

all_features = []

subjects = df["subject"].unique()

print("\nSubjects found:")
print(subjects)


for subject in subjects:

    print("\nProcessing:", subject)

    subject_df = df[
        df["subject"] == subject
    ].reset_index(drop=True)

    total_samples = len(subject_df)


    # ======================================
    # CREATE 30 SECOND WINDOWS
    # ======================================

    for start in range(
        0,
        total_samples - WINDOW_SAMPLES + 1,
        WINDOW_SAMPLES
    ):

        end = start + WINDOW_SAMPLES

        window = subject_df.iloc[start:end]


        if len(window) < WINDOW_SAMPLES:
            continue


        # Extract features
        features = extract_features(window)


        # ==================================
        # LABEL
        # ==================================

        stress_label = window["stress"].mode()[0]

        features["stress"] = stress_label

        features["subject"] = subject


        all_features.append(features)


# ==========================================
# CREATE DATAFRAME
# ==========================================

features_df = pd.DataFrame(
    all_features
)


print("\n===================================")
print("ADVANCED FEATURE ENGINEERING")
print("===================================")

print(
    "Total windows:",
    len(features_df)
)

print(
    "Number of columns:",
    len(features_df.columns)
)

print("\nStress distribution:")

print(
    features_df["stress"].value_counts()
)


# ==========================================
# SAVE
# ==========================================

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved file:")
print(OUTPUT_FILE)