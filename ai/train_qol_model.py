
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Load combined dataset
df = pd.read_csv(
    "data/processed/qol_stress_combined.csv"
)

# Select physiological features
features = [
    "RMSSD",
    "SDNN",
    "heart_rate",
    "scl_mean",
    "scl_std",
    "activation_mean",
    "activation_std"
]

X = df[features]
y = df["stress_label"]
groups = df["participant_id"]

# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42
)

# Participant-independent cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv
)

# Evaluation
print("\n========== QOL STRESS MODEL ==========")

print("Accuracy:", round(accuracy_score(y, predictions), 4))
print("F1 Score:", round(f1_score(y, predictions), 4))

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        target_names=["Relax", "Stress"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\n======================================")
