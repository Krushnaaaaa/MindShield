import pickle
import numpy as np
import pandas as pd

# ==========================================
# 1. Load WESAD S2
# ==========================================

file_path = "/Users/raj/Desktop/WESAD/S2/S2.pkl"

with open(file_path, "rb") as file:
    data = pickle.load(file, encoding="latin1")

wrist = data["signal"]["wrist"]

acc = wrist["ACC"]
bvp = wrist["BVP"].flatten()
eda = wrist["EDA"].flatten()
temp = wrist["TEMP"].flatten()
labels = data["label"]

print("WESAD S2 loaded successfully!")

# ==========================================
# 2. WESAD sampling rates
# ==========================================

ACC_FS = 32
BVP_FS = 64
EDA_FS = 4
TEMP_FS = 4

# ==========================================
# 3. Convert each signal to time
# ==========================================

acc_time = np.arange(len(acc)) / ACC_FS
bvp_time = np.arange(len(bvp)) / BVP_FS
eda_time = np.arange(len(eda)) / EDA_FS
temp_time = np.arange(len(temp)) / TEMP_FS
label_time = np.arange(len(labels)) / 700

# ==========================================
# 4. Convert ACC into movement magnitude
# ==========================================

acc_magnitude = np.sqrt(
    acc[:, 0] ** 2 +
    acc[:, 1] ** 2 +
    acc[:, 2] ** 2
)

# ==========================================
# 5. Create common timeline
# ==========================================

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

# Use EDA/TEMP rate as common timeline
common_time = np.arange(
    start_time,
    end_time,
    1 / EDA_FS
)

# ==========================================
# 6. Interpolate signals
# ==========================================

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

# ==========================================
# 7. Match labels to common timeline
# ==========================================

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

# ==========================================
# 8. Create dataframe
# ==========================================

df = pd.DataFrame({
    "time": common_time,
    "ACC": acc_common,
    "BVP": bvp_common,
    "EDA": eda_common,
    "TEMP": temp_common,
    "label": label_common
})

# ==========================================
# 9. Remove undefined label
# ==========================================

df = df[df["label"] != 0].copy()

# ==========================================
# 10. Convert labels
# ==========================================

df["stress"] = df["label"].apply(
    lambda x: 1 if x in [2, 7] else 0
)

# ==========================================
# 11. Remove original label
# ==========================================

df.drop(columns=["label"], inplace=True)

# ==========================================
# 12. Save processed data
# ==========================================

output_path = "data/processed/mindshield_data.csv"

df.to_csv(
    output_path,
    index=False
)

# ==========================================
# 13. Show result
# ==========================================

print("\nData preparation completed!")
print("--------------------------------")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\nStress distribution:")
print(df["stress"].value_counts())

print("\nSaved file:")
print(output_path)