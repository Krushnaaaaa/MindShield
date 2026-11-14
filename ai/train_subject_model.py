import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. Load all subjects
# ==========================================

file_path = "data/processed/wesad_all_subjects.csv"

df = pd.read_csv(file_path)

print("Dataset loaded!")
print("Total rows:", len(df))

# ==========================================
# 2. Define training and testing subjects
# ==========================================

train_subjects = [
    "S2", "S3", "S4", "S5",
    "S6", "S7", "S8", "S9",
    "S10", "S11", "S13", "S14", "S15"
]

test_subjects = [
    "S16", "S17"
]

# ==========================================
# 3. Split subjects
# ==========================================

train_df = df[df["subject"].isin(train_subjects)].copy()

test_df = df[df["subject"].isin(test_subjects)].copy()

print("\nTraining subjects:")
print(train_subjects)

print("\nTesting subjects:")
print(test_subjects)

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

# ==========================================
# 4. Features
# ==========================================

features = [
    "ACC",
    "BVP",
    "EDA",
    "TEMP"
]

X_train = train_df[features]
y_train = train_df["stress"]

X_test = test_df[features]
y_test = test_df["stress"]

# ==========================================
# 5. Train Random Forest
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 6. Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 7. Accuracy
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL RESULTS")
print("===================================")

print("\nAccuracy:", accuracy)

# ==========================================
# 8. Classification report
# ==========================================

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

# ==========================================
# 9. Confusion matrix
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================================
# 10. Feature importance
# ==========================================

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