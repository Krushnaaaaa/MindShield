import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# LOAD ADVANCED FEATURE DATASET
# ==========================================

file_path = "data/processed/wesad_advanced_features.csv"

df = pd.read_csv(file_path)

print("Dataset loaded!")
print("Total windows:", len(df))


# ==========================================
# FEATURES
# ==========================================

features = [
    "EDA_mean",
    "EDA_std",
    "EDA_min",
    "EDA_max",
    "EDA_range",
    "EDA_change_mean",
    "EDA_change_std",
    "EDA_slope",
    "EDA_variability_ratio",

    "BVP_mean",
    "BVP_std",
    "BVP_min",
    "BVP_max",
    "BVP_range",
    "BVP_change_mean",
    "BVP_change_std",

    "ACC_mean",
    "ACC_std",
    "ACC_min",
    "ACC_max",
    "ACC_range",
    "ACC_change_mean",
    "ACC_change_std",

    "TEMP_mean",
    "TEMP_std",
    "TEMP_min",
    "TEMP_max",
    "TEMP_range"
]


# ==========================================
# SUBJECTS
# ==========================================

subjects = df["subject"].unique()

print("\nSubjects:")
print(subjects)


# ==========================================
# STORE RESULTS
# ==========================================

results = []


# ==========================================
# LEAVE-ONE-SUBJECT-OUT VALIDATION
# ==========================================

for test_subject in subjects:

    print("\n===================================")
    print("Testing subject:", test_subject)
    print("===================================")

    # One subject = test
    test_df = df[
        df["subject"] == test_subject
    ].copy()

    # All remaining subjects = training
    train_df = df[
        df["subject"] != test_subject
    ].copy()


    X_train = train_df[features]
    y_train = train_df["stress"]

    X_test = test_df[features]
    y_test = test_df["stress"]


    # ======================================
    # RANDOM FOREST
    # ======================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)


    # ======================================
    # PREDICTION
    # ======================================

    y_pred = model.predict(X_test)


    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    print("Windows:", len(test_df))
    print("Accuracy:", round(accuracy, 4))
    print("Stress Precision:", round(precision, 4))
    print("Stress Recall:", round(recall, 4))
    print("Stress F1:", round(f1, 4))


    results.append({
        "subject": test_subject,
        "windows": len(test_df),
        "accuracy": accuracy,
        "stress_precision": precision,
        "stress_recall": recall,
        "stress_f1": f1
    })


# ==========================================
# FINAL RESULTS
# ==========================================

results_df = pd.DataFrame(results)


print("\n\n===================================")
print("ADVANCED LOSO FINAL RESULTS")
print("===================================")

print("\nResults by subject:")

print(results_df)


# ==========================================
# AVERAGE PERFORMANCE
# ==========================================

print("\n===================================")
print("AVERAGE PERFORMANCE")
print("===================================")

print(
    "Average Accuracy:",
    round(results_df["accuracy"].mean(), 4)
)

print(
    "Average Stress Precision:",
    round(results_df["stress_precision"].mean(), 4)
)

print(
    "Average Stress Recall:",
    round(results_df["stress_recall"].mean(), 4)
)

print(
    "Average Stress F1:",
    round(results_df["stress_f1"].mean(), 4)
)


# ==========================================
# SAVE RESULTS
# ==========================================

results_df.to_csv(
    "data/processed/loso_advanced_results.csv",
    index=False
)

print("\nSaved results:")
print("data/processed/loso_advanced_results.csv")