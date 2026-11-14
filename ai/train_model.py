import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Load processed data
# ==========================================

file_path = "data/processed/mindshield_data.csv"

df = pd.read_csv(file_path)

print("Dataset loaded!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ==========================================
# 2. Create features
# ==========================================

features = [
    "ACC",
    "BVP",
    "EDA",
    "TEMP"
]

X = df[features]
y = df["stress"]

# ==========================================
# 3. Split data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 4. Train Random Forest
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# ==========================================
# 5. Make predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 6. Evaluate model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 7. Feature importance
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