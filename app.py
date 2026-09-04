import os
import uuid
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from db import get_db, init_db, DB_PATH

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE, "static", "uploads")
ALLOWED_EXT = {"pdf", "doc", "docx"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "placement-predictor-secret-key-change-me")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

if not os.path.exists(DB_PATH):
    init_db()
else:
    init_db()  # safe: CREATE TABLE IF NOT EXISTS

# ---------- Load ML model ----------
MODEL_DIR = os.path.join(BASE, "model")
model = joblib.load(os.path.join(MODEL_DIR, "placement_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
FEATURES = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
with open(os.path.join(MODEL_DIR, "accuracy.txt")) as f:
    MODEL_ACCURACY = f.read().strip()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def generate_placement_id():
    return "STU" + datetime.now().strftime("%y%m") + uuid.uuid4().hex[:5].upper()


def run_prediction(row):
    X = np.array([[row[f] for f in FEATURES]])
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    prob = model.predict_proba(Xs)[0][1]
    package = 0
    if pred == 1:
        package = round(3 + row["cgpa"] * 0.6 + row["internships"] * 0.5, 2)
    return ("Likely to be Placed" if pred == 1 else "Needs Improvement"), round(prob * 100, 2), package


# ============ PUBLIC ============

@app.route("/")
def home():
    return render_template("index.html")


# ============ STUDENT AUTH ============

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        f = request.form
        username = f.get("username", "").strip()
        password = f.get("password", "")
        full_name = f.get("full_name", "").strip()
        email = f.get("email", "").strip()

        db = get_db()
        existing = db.execute("SELECT id FROM students WHERE username=?", (username,)).fetchone()
        if existing:
            flash("Username already taken. Please choose another.", "error")
            db.close()
            return render_template("register.html")

        resume_filename = None
        file = request.files.get("resume")
        if file and file.filename:
            if not allowed_file(file.filename):
                flash("Resume must be a PDF or Word document.", "error")
                db.close()
                return render_template("register.html")
            safe_name = secure_filename(f"{username}_{uuid.uuid4().hex[:6]}_{file.filename}")
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], safe_name))
            resume_filename = safe_name

        row = dict(
            tenth_percent=float(f.get("tenth_percent", 0)),
            twelfth_percent=float(f.get("twelfth_percent", 0)),
            cgpa=float(f.get("cgpa", 0)),
            internships=int(f.get("internships", 0)),
            projects=int(f.get("projects", 0)),
            backlogs=int(f.get("backlogs", 0)),
            communication_skill=int(f.get("communication_skill", 5)),
            extra_curricular=1 if f.get("extra_curricular") == "yes" else 0,
        )
        result, prob, package = run_prediction(row)
        placement_id = generate_placement_id()

        db.execute("""
            INSERT INTO students (
                placement_id, username, password, full_name, email,
                tenth_percent, twelfth_percent, degree, cgpa,
                internships, projects, backlogs, communication_skill, extra_curricular,
                interest, resume_filename, prediction_result, prediction_prob, predicted_package
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            placement_id, username, generate_password_hash(password), full_name, email,
            row["tenth_percent"], row["twelfth_percent"], f.get("degree"), row["cgpa"],
            row["internships"], row["projects"], row["backlogs"], row["communication_skill"], row["extra_curricular"],
            f.get("interest"), resume_filename, result, prob, package,
        ))
        db.commit()
        db.close()

        flash(f"Registration successful! Your Placement ID is {placement_id}. Please log in.", "success")
        return redirect(url_for("student_login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = get_db()
        student = db.execute("SELECT * FROM students WHERE username=?", (username,)).fetchone()
        db.close()
        if student and check_password_hash(student["password"], password):
            session["student_id"] = student["id"]
            session["student_name"] = student["full_name"]
            return redirect(url_for("student_dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))


# ============ STUDENT DASHBOARD ============

def require_student():
    return "student_id" in session


@app.route("/dashboard")
def student_dashboard():
    if not require_student():
        return redirect(url_for("student_login"))
    db = get_db()
    student = db.execute("SELECT * FROM students WHERE id=?", (session["student_id"],)).fetchone()
    complaints = db.execute(
        "SELECT * FROM complaints WHERE student_id=? ORDER BY created_at DESC", (session["student_id"],)
    ).fetchall()
    db.close()
    return render_template("student_dashboard.html", student=student, complaints=complaints)


@app.route("/complaint", methods=["GET", "POST"])
def complaint():
    if not require_student():
        return redirect(url_for("student_login"))
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()
        db = get_db()
        db.execute(
            "INSERT INTO complaints (student_id, subject, message) VALUES (?,?,?)",
            (session["student_id"], subject, message),
        )
        db.commit()
        db.close()
        flash("Your complaint/query has been submitted to the admin.", "success")
        return redirect(url_for("student_dashboard"))
    return render_template("complaint.html")


# ============ ADMIN AUTH ============

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = get_db()
        admin = db.execute("SELECT * FROM admin WHERE username=?", (username,)).fetchone()
        db.close()
        if admin and check_password_hash(admin["password"], password):
            session["admin_id"] = admin["id"]
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin credentials.", "error")
    return render_template("admin_login.html")


def require_admin():
    return "admin_id" in session


@app.route("/admin/dashboard")
def admin_dashboard():
    if not require_admin():
        return redirect(url_for("admin_login"))
    db = get_db()
    students = db.execute("SELECT * FROM students ORDER BY created_at DESC").fetchall()
    total = len(students)
    placed = len([s for s in students if s["prediction_result"] == "Likely to be Placed"])
    not_placed = total - placed
    open_complaints = db.execute("SELECT COUNT(*) c FROM complaints WHERE status='Open'").fetchone()["c"]
    db.close()
    stats = dict(total=total, placed=placed, not_placed=not_placed, open_complaints=open_complaints,
                 model_accuracy=MODEL_ACCURACY)
    return render_template("admin_dashboard.html", students=students, stats=stats)


@app.route("/admin/student/<int:student_id>", methods=["GET", "POST"])
def admin_student_detail(student_id):
    if not require_admin():
        return redirect(url_for("admin_login"))
    db = get_db()
    if request.method == "POST":
        remark = request.form.get("remark", "").strip()
        status = request.form.get("status", "Pending Review")
        db.execute("UPDATE students SET admin_remark=?, status=? WHERE id=?", (remark, status, student_id))
        db.commit()
        flash("Remark saved.", "success")
    student = db.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    db.close()
    if not student:
        return "Student not found", 404
    return render_template("admin_student_detail.html", student=student)


@app.route("/admin/reports")
def admin_reports():
    if not require_admin():
        return redirect(url_for("admin_login"))
    db = get_db()
    students = db.execute("SELECT * FROM students ORDER BY created_at DESC").fetchall()
    db.close()

    date_filter = request.args.get("date")
    if date_filter:
        students = [s for s in students if s["created_at"].startswith(date_filter)]

    return render_template("admin_reports.html", students=students, date_filter=date_filter or "")


@app.route("/admin/complaints", methods=["GET", "POST"])
def admin_complaints():
    if not require_admin():
        return redirect(url_for("admin_login"))
    db = get_db()
    if request.method == "POST":
        complaint_id = request.form.get("complaint_id")
        reply = request.form.get("reply", "").strip()
        status = request.form.get("status", "Open")
        db.execute(
            "UPDATE complaints SET admin_reply=?, status=? WHERE id=?", (reply, status, complaint_id)
        )
        db.commit()
    rows = db.execute("""
        SELECT c.*, s.full_name, s.placement_id
        FROM complaints c JOIN students s ON c.student_id = s.id
        ORDER BY c.created_at DESC
    """).fetchall()
    db.close()
    return render_template("admin_complaints.html", complaints=rows)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_id", None)
    return redirect(url_for("home"))


# ============ RESUME DOWNLOAD (admin only) ============

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    if not (require_admin() or require_student()):
        return redirect(url_for("home"))
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
