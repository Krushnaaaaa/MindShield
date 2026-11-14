import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==============================
# LOAD FEATURE DATASET
# ==============================

file_path = "data/processed/wesad_features.csv"

df = pd.read_csv(file_path)

print("Dataset loaded!")
print("Total windows:", len(df))


# ==============================
# TRAIN / TEST SUBJECTS
# ==============================

train_subjects = [
    "S2", "S3", "S4", "S5",
    "S6", "S7", "S8", "S9",
    "S10", "S11", "S13", "S14", "S15"
]

test_subjects = [
    "S16", "S17"
]


train_df = df[
    df["subject"].isin(train_subjects)
].copy()

test_df = df[
    df["subject"].isin(test_subjects)
].copy()


print("\nTraining subjects:")
print(train_subjects)

print("\nTesting subjects:")
print(test_subjects)

print("\nTraining windows:", len(train_df))
print("Testing windows:", len(test_df))


# ==============================
# SELECT FEATURES
# ==============================

features = [
    "EDA_mean",
    "EDA_std",
    "EDA_min",
    "EDA_max",
    "EDA_range",

    "BVP_mean",
    "BVP_std",
    "BVP_min",
    "BVP_max",
    "BVP_range",

    "ACC_mean",
    "ACC_std",
    "ACC_min",
    "ACC_max",
    "ACC_range",

    "TEMP_mean",
    "TEMP_std",
    "TEMP_min",
    "TEMP_max",
    "TEMP_range"
]


X_train = train_df[features]
y_train = train_df["stress"]

X_test = test_df[features]
y_test = test_df["stress"]


# ==============================
# TRAIN RANDOM FOREST
# ==============================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training completed!")


# ==============================
# PREDICTIONS
# ==============================

y_pred = model.predict(X_test)


# ==============================
# MODEL RESULTS
# ==============================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("IMPROVED MODEL RESULTS")
print("===================================")

print("\nAccuracy:", accuracy)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Stress",
            "Stress"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")

print(importance)