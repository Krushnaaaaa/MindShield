import os
import pandas as pd

for filename in ["eda.csv", "ecg.csv"]:
    for root, dirs, files in os.walk("data/raw/QoL_Stress"):
        if filename in files:
            df = pd.read_csv(os.path.join(root, filename))

            print("\n==========", filename, "==========")
            print("Rows and columns:", df.shape)
            print("Participant count:", df["username"].nunique()
                  if "username" in df.columns else "No username column")
            print("Phases:", df["phase"].unique()
                  if "phase" in df.columns else "No phase column")
            print("Column names:", df.columns.tolist())
            break