"""
Application Controller (Flask Main Entrypoint)
Handles routing, session authentication, AJAX actions, and streaming.
"""

import os
import io
import csv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    Response,
    jsonify,
)
from models import (
    init_db,
    get_db_connection,
    add_employee,
    get_all_employees,
    get_dashboard_metrics,
    query_attendance_reports,
)
from face_engine import (
    generate_face_dataset,
    train_model,
    generate_live_recognition_stream,
)

app = Flask(__name__)
app.secret_key = "super_secure_attendance_system_secret_key"

# Initialize DB tables on startup
init_db()


# ---------------- Authentication Decorator & Routes ---------------- #
def is_logged_in():
    return "admin" in session


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        admin = conn.execute(
            "SELECT * FROM admin_users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        conn.close()

        if admin:
            session["admin"] = username
            flash("Welcome back, Admin!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("admin", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ---------------- Dashboard & Management Routes ---------------- #
@app.route("/")
@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("login"))

    metrics = get_dashboard_metrics()
    return render_template("dashboard.html", metrics=metrics)


@app.route("/register", methods=["GET", "POST"])
def register_employee():
    if not is_logged_in():
        return redirect(url_for("login"))

    if request.method == "POST":
        emp_id = request.form.get("employee_id").strip()
        name = request.form.get("name").strip()
        department = request.form.get("department").strip()
        email = request.form.get("email").strip()
        phone = request.form.get("phone").strip()

        if not all([emp_id, name, department, email, phone]):
            flash("All fields are mandatory.", "warning")
            return redirect(url_for("register_employee"))

        success, message = add_employee(emp_id, name, department, email, phone)
        if success:
            flash(
                f"Employee {name} registered! Now proceed to capture face dataset.",
                "success",
            )
            return redirect(url_for("capture_faces", employee_id=emp_id))
        else:
            flash(message, "danger")

    return render_template("register_employee.html")


@app.route("/capture-faces")
def capture_faces():
    if not is_logged_in():
        return redirect(url_for("login"))

    emp_id = request.args.get("employee_id", "")
    employees = get_all_employees()
    return render_template(
        "capture_faces.html", selected_id=emp_id, employees=employees
    )


@app.route("/api/start-capture", methods=["POST"])
def start_capture_api():
    if not is_logged_in():
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    emp_id = request.json.get("employee_id")
    if not emp_id:
        return (
            jsonify({"status": "error", "message": "Employee ID is required."}),
            400,
        )

    try:
        count = generate_face_dataset(emp_id, max_samples=60)
        return jsonify(
            {
                "status": "success",
                "message": f"Successfully captured {count} face samples for {emp_id}.",
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/train-model")
def train_model_page():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("train_model.html")


@app.route("/api/run-training", methods=["POST"])
def run_training_api():
    if not is_logged_in():
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    success, message = train_model()
    if success:
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({"status": "error", "message": message}), 400


@app.route("/live-attendance")
def live_attendance():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("live_recognition.html")


@app.route("/video-feed")
def video_feed():
    """Video streaming route for real-time recognition."""
    if not is_logged_in():
        return Response("Unauthorized", status=401)
    return Response(
        generate_live_recognition_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/latest-logs")
def latest_logs():
    """AJAX polling endpoint for dashboard and live monitor."""
    if not is_logged_in():
        return jsonify([])
    metrics = get_dashboard_metrics()
    logs = [dict(row) for row in metrics["recent_logs"]]
    return jsonify(logs)


@app.route("/reports")
def reports():
    if not is_logged_in():
        return redirect(url_for("login"))

    date_filter = request.args.get("date", "")
    month_filter = request.args.get("month", "")
    emp_filter = request.args.get("employee_id", "")

    records = query_attendance_reports(date_filter, month_filter, emp_filter)
    employees = get_all_employees()

    return render_template(
        "reports.html",
        records=records,
        employees=employees,
        filters={
            "date": date_filter,
            "month": month_filter,
            "employee_id": emp_filter,
        },
    )


@app.route("/reports/export")
def export_csv():
    if not is_logged_in():
        return redirect(url_for("login"))

    date_filter = request.args.get("date", "")
    month_filter = request.args.get("month", "")
    emp_filter = request.args.get("employee_id", "")

    records = query_attendance_reports(date_filter, month_filter, emp_filter)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Attendance ID",
            "Employee ID",
            "Name",
            "Department",
            "Date",
            "Time",
            "Status",
        ]
    )

    for row in records:
        writer.writerow(
            [
                row["attendance_id"],
                row["employee_id"],
                row["name"],
                row["department"],
                row["date"],
                row["time"],
                row["status"],
            ]
        )

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment;filename=attendance_report.csv"
        },
    )


@app.route("/settings")
def settings():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("settings.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)