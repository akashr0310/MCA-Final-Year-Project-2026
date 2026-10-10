import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = "dataset/intrusion_dataset.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "intrusion_model.pkl")


# ============================================================
# 2. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n==========================================")
print(" CYBERSECURITY INTRUSION DETECTION SYSTEM")
print("==========================================")

print("\nLoading dataset...")

try:
    data = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print("\nERROR: Dataset file not found.")
    print("Make sure this file exists:")
    print("dataset/intrusion_dataset.csv")
    exit()


print("\nDataset loaded successfully!")

print("\nNumber of records:", len(data))
print("Number of columns:", len(data.columns))

print("\nDataset columns:")
print(list(data.columns))


# ============================================================
# 4. CHECK LABEL COLUMN
# ============================================================

if "label" not in data.columns:
    print("\nERROR: 'label' column was not found.")
    print("Your dataset must contain a column named 'label'.")
    exit()


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

missing_values = data.isnull().sum()

print(missing_values)

data = data.dropna()

print("\nRecords after removing missing values:", len(data))


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop("label", axis=1)
y = data["label"]

print("\nFeature columns:")
print(list(X.columns))

print("\nAttack classes:")
print(y.value_counts())


# ============================================================
# 7. IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumeric columns:")
print(numeric_columns)


# ============================================================
# 8. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ============================================================
# 10. CREATE MACHINE LEARNING PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ============================================================
# 13. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 14. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print(" MODEL PERFORMANCE")
print("==========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 17. SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")

joblib.dump(
    model,
    MODEL_PATH
)

print("\n==========================================")
print(" MODEL SAVED SUCCESSFULLY")
print("==========================================")

print("\nModel location:")
print(MODEL_PATH)

print("\nTraining completed!")