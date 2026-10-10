import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TRAIN_FILE = "dataset/UNSW_NB15_training-set.csv"
TEST_FILE = "dataset/UNSW_NB15_testing-set.csv"
MODEL_FILE = "model/intrusion_model.pkl"
METRICS_FILE = "model/unsw_metrics.json"
CONFUSION_FILE = "model/unsw_confusion_matrix.csv"

# The downloaded UNSW-NB15 files being used here do not contain a header row.
UNSW_COLUMNS = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts",
    "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin",
    "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
    "attack_cat", "label"
]


def load_unsw_file(filename):
    print(f"Loading: {filename}")

    # Read without assuming a header.
    df = pd.read_csv(
    filename,
    header=None,
    sep=None,
    engine="python"
)

    # If a header row is present in the downloaded file, remove it.
    first_row = [str(x).strip().lower() for x in df.iloc[0].tolist()]
    if "attack_cat" in first_row or "label" in first_row:
        df = pd.read_csv(filename, header=0)

    # Some copies of UNSW-NB15 have no header and exactly 45 columns.
    if len(df.columns) == len(UNSW_COLUMNS):
        df.columns = UNSW_COLUMNS
    else:
        raise ValueError(
            f"{filename} has {len(df.columns)} columns, but "
            f"UNSW-NB15 is expected to have {len(UNSW_COLUMNS)} columns."
        )

    # Clean text fields.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(
            {"nan": np.nan, "None": np.nan, "": np.nan}
        )

    return df


def main():
    print("=" * 70)
    print("UNSW-NB15 MACHINE LEARNING MODEL TRAINING")
    print("=" * 70)

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Testing file not found: {TEST_FILE}")

    print("\n[1/7] Loading datasets...")
    train_df = load_unsw_file(TRAIN_FILE)
    test_df = load_unsw_file(TEST_FILE)

    print(f"Training records : {len(train_df):,}")
    print(f"Testing records  : {len(test_df):,}")
    print(f"Training columns : {len(train_df.columns)}")

    target_col = "attack_cat"

    print("\n[2/7] Preparing target...")

    # Missing attack category means normal traffic in this dataset.
    train_df[target_col] = train_df[target_col].fillna("Normal").astype(str).str.strip()
    test_df[target_col] = test_df[target_col].fillna("Normal").astype(str).str.strip()

    train_df.loc[
        train_df[target_col].isin(["", "nan", "None"]), target_col
    ] = "Normal"

    test_df.loc[
        test_df[target_col].isin(["", "nan", "None"]), target_col
    ] = "Normal"

    # We classify attack categories.
    # id = identifier, label = binary flag, attack_cat = target.
    drop_columns = ["id", "label", target_col]

    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[target_col]

    common_features = [c for c in X_train.columns if c in X_test.columns]

    X_train = X_train[common_features]
    X_test = X_test[common_features]

    print(f"Features used : {len(common_features)}")

    print("\nClass distribution:")
    print(y_train.value_counts())

    print("\n[3/7] Building preprocessing pipeline...")

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_features = [
        c for c in X_train.columns if c not in categorical_features
    ]

    print(f"Numeric features     : {len(numeric_features)}")
    print(f"Categorical features : {len(categorical_features)}")
    print(f"Categorical columns  : {categorical_features}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    print("\n[4/7] Training Random Forest...")
    print("This can take a few minutes.")

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_features="sqrt"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    print("\n[5/7] Evaluating on UNSW-NB15 testing set...")

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    labels = sorted(y_train.unique().tolist())

    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("\n" + "=" * 70)
    print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 70)
    print("\nCLASSIFICATION REPORT:")
    print(report_text)

    print("\n[6/7] Saving model and evaluation results...")

    os.makedirs("model", exist_ok=True)

    joblib.dump(pipeline, MODEL_FILE)

    metrics = {
        "dataset": "UNSW-NB15",
        "target": target_col,
        "training_records": int(len(train_df)),
        "testing_records": int(len(test_df)),
        "feature_count": int(len(common_features)),
        "classes": labels,
        "accuracy": float(accuracy),
        "classification_report": report_dict,
        "feature_columns": common_features
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
    cm_df.to_csv(CONFUSION_FILE)

    print(f"Model saved       : {MODEL_FILE}")
    print(f"Metrics saved     : {METRICS_FILE}")
    print(f"Confusion matrix  : {CONFUSION_FILE}")

    print("\n[7/7] Training summary")
    print("-" * 70)
    print(f"Dataset           : UNSW-NB15")
    print(f"Training records  : {len(train_df):,}")
    print(f"Testing records   : {len(test_df):,}")
    print(f"Features          : {len(common_features)}")
    print(f"Accuracy          : {accuracy * 100:.2f}%")
    print("-" * 70)
    print("\nMODEL TRAINING SUCCESSFUL.")


if __name__ == "__main__":
    main()
