import os
import pandas as pd

for root, dirs, files in os.walk("data/raw/QoL_Stress"):
    if "eda.csv" in files:
        df = pd.read_csv(os.path.join(root, "eda.csv"))

        print("Total rows:", len(df))
        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nValid data counts:")
        print(df["valid_data"].value_counts(dropna=False))

        print("\nParticipants per phase:")
        print(df.groupby("phase")["username"].nunique())

        print("\nEDA statistics:")
        print(df[["activation", "scl_avg"]].describe())
        break