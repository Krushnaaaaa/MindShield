
import os
import pandas as pd
import matplotlib.pyplot as plt

# Dataset folder
DATA_PATH = "data/raw/QoL_Stress"

# Find ECG file
ecg_path = None

for root, dirs, files in os.walk(DATA_PATH):
    if "ecg.csv" in files:
        ecg_path = os.path.join(root, "ecg.csv")
        break

if ecg_path is None:
    raise FileNotFoundError("ecg.csv not found!")

# Load ECG dataset
df = pd.read_csv(ecg_path)

print("\n========== ECG DATASET ==========")
print("Total recordings:", len(df))
print("Unique participants:", df["username"].nunique())

print("\nPhase distribution:")
print(df["phase"].value_counts())

print("\nHeart rate statistics:")
print(df["heart_rate"].describe())

# Extract first recording
sample = df.iloc[0]["waveform_samples"]

# Convert waveform string into numerical values
values = [float(x) for x in str(sample).split()]

print("\n========== FIRST RECORDING DETAILS ==========")
print("Participant:", df.iloc[0]["username"])
print("Phase:", df.iloc[0]["phase"])
print("Heart rate:", df.iloc[0]["heart_rate"])
print("Number of samples:", len(values))
print("First 20 samples:", values[:20])
print("Minimum amplitude:", min(values))
print("Maximum amplitude:", max(values))

# Plot first 1500 waveform samples
plt.figure(figsize=(14, 5))

plt.plot(values[:1500])

plt.title("QoL_Stress ECG Waveform - First 1500 Samples")
plt.xlabel("Sample Number")
plt.ylabel("Raw Amplitude")

plt.grid(True)
plt.tight_layout()

plt.show()
