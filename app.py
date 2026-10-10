from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
from database import db, User, Prediction, Alert, init_db
from prediction import predict_intrusion
from realtime_monitor import monitor_once

app = Flask(__name__)
app.config.from_object(Config)
init_db(app)


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            session.clear()
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    error = None
    success = None

    if request.method == "POST":

        email = request.form.get("email", "").strip()

        if not email:
            error = "Please enter your registered email address."

        else:

            user = User.query.filter_by(email=email).first()

            if user:

                # Store the user ID temporarily for password reset
                session["reset_user_id"] = user.id

                return redirect(url_for("reset_password"))

            else:

                error = "No account found with this email address."

    return render_template(
        "forgot_password.html",
        error=error,
        success=success
    )
@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():

    if "reset_user_id" not in session:
        return redirect(url_for("login"))

    error = None

    if request.method == "POST":

        new_password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not new_password:

            error = "Please enter a new password."

        elif len(new_password) < 6:

            error = "Password must contain at least 6 characters."

        elif new_password != confirm_password:

            error = "Passwords do not match."

        else:

            user = User.query.get(session["reset_user_id"])

            if user:

                user.password_hash = generate_password_hash(
                    new_password
                )

                db.session.commit()

                session.pop("reset_user_id", None)

                return redirect(url_for("login"))

            else:

                error = "User account not found."

    return render_template(
        "reset_password.html",
        error=error
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Total predictions
    total_predictions = Prediction.query.count()

    # Normal traffic
    normal_predictions = Prediction.query.filter_by(
        prediction="Normal"
    ).count()

    # Malicious traffic
    malicious_predictions = Prediction.query.filter(
        Prediction.prediction != "Normal"
    ).count()

    # Critical threats
    critical_threats = Prediction.query.filter_by(
        severity="CRITICAL"
    ).count()

    # High severity threats
    high_threats = Prediction.query.filter_by(
        severity="HIGH"
    ).count()

    # Medium severity threats
    medium_threats = Prediction.query.filter_by(
        severity="MEDIUM"
    ).count()

    # Low severity traffic
    low_threats = Prediction.query.filter_by(
        severity="LOW"
    ).count()

    # Get recent predictions
    recent_predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).limit(10).all()

    # Attack category counts
    predictions = Prediction.query.all()

    attack_counts = {}

    for prediction in predictions:
        attack_name = prediction.prediction

        if attack_name not in attack_counts:
            attack_counts[attack_name] = 0

        attack_counts[attack_name] += 1

    # Recent alerts
    recent_alerts = Alert.query.order_by(
        Alert.created_at.desc()
    ).limit(5).all()

        # Chart data

    traffic_chart = {
        "Normal": normal_predictions,
        "Malicious": malicious_predictions
    }

    severity_chart = {
        "CRITICAL": critical_threats,
        "HIGH": high_threats,
        "MEDIUM": medium_threats,
        "LOW": low_threats
    }

    attack_chart = attack_counts

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        normal_predictions=normal_predictions,
        malicious_predictions=malicious_predictions,
        critical_threats=critical_threats,
        high_threats=high_threats,
        medium_threats=medium_threats,
        low_threats=low_threats,
        recent_predictions=recent_predictions,
        attack_counts=attack_counts,
        recent_alerts=recent_alerts,
        traffic_chart=traffic_chart,
        severity_chart=severity_chart,
        attack_chart=attack_chart
    )

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user_id" not in session:
        return redirect(url_for("login"))

    results = []
    error = None
    normal_count = 0
    malicious_count = 0

    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if not uploaded_file or uploaded_file.filename == "":
            error = "Please select a CSV file."

        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only CSV files are supported."

        else:
            try:
                import pandas as pd
                from io import BytesIO

                file_bytes = uploaded_file.read()

                df = pd.read_csv(
                    BytesIO(file_bytes),
                    sep=None,
                    engine="python",
                    header=None
                )

                # UNSW-NB15 has 45 columns.
                unsw_columns = [
                    "id", "dur", "proto", "service", "state",
                    "spkts", "dpkts", "sbytes", "dbytes", "rate",
                    "sttl", "dttl", "sload", "dload", "sloss",
                    "dloss", "sinpkt", "dinpkt", "sjit", "djit",
                    "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
                    "synack", "ackdat", "smean", "dmean",
                    "trans_depth", "response_body_len", "ct_srv_src",
                    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
                    "ct_dst_sport_ltm", "ct_dst_src_ltm",
                    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
                    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
                    "attack_cat", "label"
                ]

                if len(df.columns) != 45:
                    raise ValueError(
                        "Please upload a valid UNSW-NB15 CSV file."
                    )

                df.columns = unsw_columns

                if len(df) > 1000:
                    raise ValueError(
                        "Please upload a maximum of 1000 records at a time."
                    )

                for _, row in df.iterrows():

                    result = predict_intrusion(row.to_dict())

                    results.append(result)

                    prediction_record = Prediction(
                        source_ip="CSV Upload",
                        destination_ip="CSV Upload",
                        protocol=str(row.get("proto", "Unknown")),
                        prediction=result["prediction"],
                        confidence=result["confidence"],
                        severity=result["severity"]
                    )

                    db.session.add(prediction_record)
                    db.session.flush()

                    if result["prediction"] != "Normal":
                        alert = Alert(
                            prediction_id=prediction_record.id,
                            message=(
                                "Intrusion detected from CSV upload: "
                                + result["prediction"]
                            ),
                            severity=result["severity"],
                            status="OPEN"
                        )

                        db.session.add(alert)

                db.session.commit()

                normal_count = sum(
                    1 for item in results
                    if item["prediction"].lower() == "normal"
                )

                malicious_count = len(results) - normal_count

            except Exception as e:
                db.session.rollback()
                error = f"Prediction failed: {str(e)}"

    return render_template(
        "prediction.html",
        results=results,
        normal_count=normal_count,
        malicious_count=malicious_count,
        error=error
    )

@app.route("/monitoring")
def monitoring():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("monitoring.html")

@app.route("/api/monitoring")
def monitoring_api():
    if "user_id" not in session:
        return {"error": "Unauthorized"}, 401

    try:
        result = monitor_once()

        traffic = result.get("traffic", {})

        # Support different protocol key names
        protocol = (
            traffic.get("protocol_type")
            or traffic.get("proto")
            or traffic.get("protocol")
            or "Unknown"
        )

        prediction = result.get("prediction", "Unknown")
        confidence = result.get("confidence", 0)
        severity = result.get("severity", "LOW")

        # Save prediction
        prediction_record = Prediction(
            source_ip="Real-Time Simulator",
            destination_ip="Real-Time Simulator",
            protocol=protocol,
            prediction=prediction,
            confidence=confidence,
            severity=severity
        )

        db.session.add(prediction_record)
        db.session.commit()

        # Create alert for malicious traffic
        if prediction != "Normal":
            alert = Alert(
                prediction_id=prediction_record.id,
                message="Real-time intrusion detected: " + prediction,
                severity=severity,
                status="OPEN"
            )

            db.session.add(alert)
            db.session.commit()

        return {
            "timestamp": result.get("timestamp"),
            "protocol_type": protocol,
            "prediction": prediction,
            "confidence": confidence,
            "severity": severity
        }

    except Exception as e:
        return {"error": str(e)}, 500
    
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        predictions=predictions
    )


@app.route("/alerts")
def alerts():
    if "user_id" not in session:
        return redirect(url_for("login"))

    alerts = Alert.query.order_by(Alert.created_at.desc()).all()
    return render_template("alerts.html", alerts=alerts)


@app.route("/alerts/<int:alert_id>/close", methods=["POST"])
def close_alert(alert_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    alert = Alert.query.get_or_404(alert_id)
    alert.status = "CLOSED"
    db.session.commit()

    return redirect(url_for("alerts"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
