import pandas as pd
import numpy as np

# ==============================
# SETTINGS
# ==============================

INPUT_FILE = "data/processed/wesad_all_subjects.csv"
OUTPUT_FILE = "data/processed/wesad_features.csv"

WINDOW_SIZE = 30       # 30-second window
SAMPLING_RATE = 4      # Our current data is 4 Hz

# Number of samples in 30 seconds
WINDOW_SAMPLES = WINDOW_SIZE * SAMPLING_RATE


# ==============================
# LOAD DATA
# ==============================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded!")
print("Total rows:", len(df))


# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================

def extract_features(window):

    features = {}

    # --------------------------
    # EDA FEATURES
    # --------------------------

    features["EDA_mean"] = window["EDA"].mean()
    features["EDA_std"] = window["EDA"].std()
    features["EDA_min"] = window["EDA"].min()
    features["EDA_max"] = window["EDA"].max()

    # EDA change over the window
    features["EDA_range"] = (
        window["EDA"].max() - window["EDA"].min()
    )


    # --------------------------
    # BVP FEATURES
    # --------------------------

    features["BVP_mean"] = window["BVP"].mean()
    features["BVP_std"] = window["BVP"].std()
    features["BVP_min"] = window["BVP"].min()
    features["BVP_max"] = window["BVP"].max()

    features["BVP_range"] = (
        window["BVP"].max() - window["BVP"].min()
    )


    # --------------------------
    # ACC FEATURES
    # --------------------------

    features["ACC_mean"] = window["ACC"].mean()
    features["ACC_std"] = window["ACC"].std()
    features["ACC_min"] = window["ACC"].min()
    features["ACC_max"] = window["ACC"].max()

    features["ACC_range"] = (
        window["ACC"].max() - window["ACC"].min()
    )


    # --------------------------
    # TEMP FEATURES
    # --------------------------

    features["TEMP_mean"] = window["TEMP"].mean()
    features["TEMP_std"] = window["TEMP"].std()
    features["TEMP_min"] = window["TEMP"].min()
    features["TEMP_max"] = window["TEMP"].max()

    features["TEMP_range"] = (
        window["TEMP"].max() - window["TEMP"].min()
    )


    return features


# ==============================
# PROCESS EACH SUBJECT
# ==============================

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

    # Process 30-second windows
    for start in range(
        0,
        total_samples - WINDOW_SAMPLES + 1,
        WINDOW_SAMPLES
    ):

        end = start + WINDOW_SAMPLES

        window = subject_df.iloc[start:end]

        # Skip incomplete windows
        if len(window) < WINDOW_SAMPLES:
            continue

        # Extract numerical features
        features = extract_features(window)

        # --------------------------
        # WINDOW LABEL
        # --------------------------

        # Majority label in the window
        stress_label = window["stress"].mode()[0]

        features["stress"] = stress_label
        features["subject"] = subject

        all_features.append(features)


# ==============================
# CREATE FINAL DATAFRAME
# ==============================

features_df = pd.DataFrame(all_features)

print("\n===================================")
print("FEATURE ENGINEERING COMPLETED")
print("===================================")

print("Total windows:", len(features_df))

print("\nFeature dataset shape:")
print(features_df.shape)

print("\nStress distribution:")

print(
    features_df["stress"].value_counts()
)

print("\nSubjects:")

print(
    features_df["subject"].unique()
)


# ==============================
# SAVE DATASET
# ==============================

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved file:")
print(OUTPUT_FILE)