import os
import pandas as pd

DATA_PATH = "data/raw/QoL_Stress"

# Locate EDA file
eda_path = None

for root, dirs, files in os.walk(DATA_PATH):
    if "eda.csv" in files:
        eda_path = os.path.join(root, "eda.csv")
        break

if eda_path is None:
    raise FileNotFoundError("eda.csv not found")

# Load dataset
df = pd.read_csv(eda_path)

# Keep valid measurements
df = df[df["valid_data"] == True].copy()

# Create binary stress label
df["stress_label"] = df["phase"].map({
    "relax": 0,
    "stress": 1
})

# Remove rows without required values
df = df.dropna(
    subset=["username", "time_sec", "scl_avg", "activation", "stress_label"]
)

# Rename participant column
df = df.rename(columns={"username": "participant_id"})

# Add dataset identification
df["dataset_source"] = "QoL_Stress"

# Save processed data
output_path = "data/processed/qol_stress_eda.csv"
os.makedirs("data/processed", exist_ok=True)

df.to_csv(output_path, index=False)

print("QoL_Stress EDA preprocessing complete!")
print("Total rows:", len(df))
print("Participants:", df["participant_id"].nunique())
print("\nLabel distribution:")
print(df["stress_label"].value_counts())
print("\nSaved to:", output_path)