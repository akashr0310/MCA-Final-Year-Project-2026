import os
import random
from datetime import datetime

import pandas as pd

from prediction import predict_intrusion

TEST_FILE = "dataset/UNSW_NB15_testing-set.csv"

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

_test_data = None


def load_test_data():
    """Load UNSW-NB15 testing data once and reuse it."""
    global _test_data

    if _test_data is None:
        if not os.path.exists(TEST_FILE):
            raise FileNotFoundError(
                f"Testing dataset not found: {TEST_FILE}"
            )

        df = pd.read_csv(
            TEST_FILE,
            header=None,
            sep=None,
            engine="python"
        )

        if len(df.columns) != 45:
            raise ValueError(
                f"Expected 45 UNSW-NB15 columns, found {len(df.columns)}."
            )

        df.columns = UNSW_COLUMNS

        # Keep the original ground-truth fields in the dataframe only
        # for testing/reference. They are NOT sent to the ML model.
        _test_data = df

    return _test_data


def generate_traffic():
    """
    Select one random UNSW-NB15 testing record.

    Ground-truth attack_cat and label are retained internally for
    reference, but the prediction is generated only from the 42
    model features.
    """
    df = load_test_data()

    row = df.sample(n=1).iloc[0]

    traffic = row.to_dict()

    actual_category = traffic.get("attack_cat")
    actual_label = traffic.get("label")

    # The prediction function ignores id, attack_cat and label.
    result = predict_intrusion(traffic)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "traffic": traffic,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "severity": result["severity"],
        "actual_category": actual_category,
        "actual_label": actual_label
    }


def monitor_once():
    """Generate one simulated real-time network event."""
    return generate_traffic()


if __name__ == "__main__":
    print("=" * 70)
    print("UNSW-NB15 REAL-TIME MONITOR TEST")
    print("=" * 70)

    for i in range(10):
        result = monitor_once()

        print(
            f"{i + 1:02d}. "
            f"Protocol={result['traffic']['proto']} | "
            f"Prediction={result['prediction']} | "
            f"Confidence={result['confidence'] * 100:.2f}% | "
            f"Severity={result['severity']}"
        )

    print("=" * 70)
    print("REAL-TIME MONITOR TEST COMPLETED.")
