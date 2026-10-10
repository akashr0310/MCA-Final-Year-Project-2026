import os
import json
import joblib
import pandas as pd

MODEL_FILE = "model/intrusion_model.pkl"
METRICS_FILE = "model/unsw_metrics.json"

# These are the 42 features used by the UNSW-NB15 model.
UNSW_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts",
    "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin",
    "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
]

_model = None


def load_model():
    """Load the trained UNSW-NB15 pipeline once."""
    global _model

    if _model is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"Model not found: {MODEL_FILE}. "
                "Run train_model_unsw.py first."
            )
        _model = joblib.load(MODEL_FILE)

    return _model


def calculate_severity(prediction, confidence):
    """
    Convert the ML prediction into a dashboard-friendly severity.

    Normal traffic is LOW.
    Attack categories are assigned severity according to category.
    Confidence is also used so that low-confidence predictions are not
    automatically treated as critical.
    """
    prediction = str(prediction).strip()
    confidence = float(confidence)

    if prediction.lower() == "normal":
        return "LOW"

    # Higher-risk categories for this project.
    critical_attacks = {"Exploits", "Shellcode", "Backdoors", "Worms"}
    high_attacks = {"DoS", "Generic", "Fuzzers"}
    medium_attacks = {"Reconnaissance", "Analysis"}

    if prediction in critical_attacks:
        severity = "CRITICAL"
    elif prediction in high_attacks:
        severity = "HIGH"
    elif prediction in medium_attacks:
        severity = "MEDIUM"
    else:
        severity = "MEDIUM"

    # Avoid calling a low-confidence classification critical.
    if confidence < 0.70 and severity == "CRITICAL":
        severity = "HIGH"
    elif confidence < 0.50 and severity in {"HIGH", "CRITICAL"}:
        severity = "MEDIUM"

    return severity


def prepare_input(data):
    """
    Convert a dictionary or DataFrame into the exact 42-feature
    structure expected by the trained pipeline.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    # Add missing features with None. The trained pipeline's imputers
    # will handle missing numeric/categorical values.
    for feature in UNSW_FEATURES:
        if feature not in df.columns:
            df[feature] = None

    # Ignore extra columns and preserve the training feature order.
    df = df[UNSW_FEATURES].copy()

    return df


def predict_intrusion(data):
    """
    Predict an UNSW-NB15 traffic record.

    Returns:
        prediction: attack category or Normal
        confidence: probability of predicted class
        severity: LOW / MEDIUM / HIGH / CRITICAL
    """
    model = load_model()
    input_df = prepare_input(data)

    prediction = model.predict(input_df)[0]

    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

    severity = calculate_severity(prediction, confidence)

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "severity": severity
    }


def get_model_info():
    """Read saved training information for the dashboard/report."""
    if not os.path.exists(METRICS_FILE):
        return {}

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    # Simple test using a normal-looking UNSW-NB15 flow.
    test_data = {
        "dur": 0.00001,
        "proto": "udp",
        "service": "-",
        "state": "INT",
        "spkts": 2,
        "dpkts": 0,
        "sbytes": 496,
        "dbytes": 0,
        "rate": 100000.0,
        "sttl": 254,
        "dttl": 0,
        "sload": 198400000.0,
        "dload": 0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 0.011,
        "dinpkt": 0.0,
        "sjit": 0.0,
        "djit": 0.0,
        "swin": 0,
        "stcpb": 0,
        "dtcpb": 0,
        "dwin": 0,
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "smean": 248,
        "dmean": 0,
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": 1,
        "ct_state_ttl": 2,
        "ct_dst_ltm": 1,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 1,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0,
        "ct_src_ltm": 1,
        "ct_srv_dst": 1,
        "is_sm_ips_ports": 0
    }

    result = predict_intrusion(test_data)

    print("=" * 60)
    print("UNSW-NB15 PREDICTION TEST")
    print("=" * 60)
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print(f"Severity   : {result['severity']}")
    print("=" * 60)
