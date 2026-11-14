
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

df = pd.read_csv(
    "data/processed/qol_stress_combined.csv"
)

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

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            class_weight="balanced",
            random_state=42
        )
    )
}

scoring = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1"
]

print("\n========== MODEL COMPARISON ==========")

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring
    )

    print(f"\n{name}")

    for metric in scoring:
        values = scores["test_" + metric]

        print(
            f"{metric}: "
            f"{values.mean():.3f} "
            f"+/- {values.std():.3f}"
        )

print("\n======================================")
