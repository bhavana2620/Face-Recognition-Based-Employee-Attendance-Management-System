"""
Database Management Module
Manages SQLite connection, database initialization, and data operations
for employees, attendance, and users.
"""

import sqlite3
from datetime import datetime

DATABASE_NAME = "attendance_system.db"


def get_db_connection():
    """Establish and return a connection with row dict factory."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Attendance logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees (employee_id),
            UNIQUE(employee_id, date)
        )
    """)

    # Admin user credentials
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Insert default admin if not exists (admin / admin123)
    cursor.execute("""
        INSERT OR IGNORE INTO admin_users (username, password)
        VALUES ('admin', 'admin123')
    """)

    conn.commit()
    conn.close()


def add_employee(emp_id, name, department, email, phone):
    """Insert a new employee record into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO employees (employee_id, name, department, email, phone)
            VALUES (?, ?, ?, ?, ?)
        """,
            (emp_id, name, department, email, phone),
        )
        conn.commit()
        return True, "Employee registered successfully."
    except sqlite3.IntegrityError as e:
        return False, f"Employee ID or Email already exists: {str(e)}"
    finally:
        conn.close()


def get_all_employees():
    """Retrieve all employee records."""
    conn = get_db_connection()
    employees = conn.execute(
        "SELECT * FROM employees ORDER BY employee_id ASC"
    ).fetchall()
    conn.close()
    return employees


def get_employee_by_id(emp_id):
    """Fetch a single employee by their Employee ID."""
    conn = get_db_connection()
    employee = conn.execute(
        "SELECT * FROM employees WHERE employee_id = ?", (emp_id,)
    ).fetchone()
    conn.close()
    return employee


def log_attendance(emp_id):
    """
    Log attendance for an employee for today.
    Prevents duplicates using SQLite UNIQUE constraint on (employee_id, date).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Check if employee exists
    emp = cursor.execute(
        "SELECT name FROM employees WHERE employee_id = ?", (emp_id,)
    ).fetchone()
    if not emp:
        conn.close()
        return False, "Unknown Employee"

    emp_name = emp["name"]

    try:
        cursor.execute(
            """
            INSERT INTO attendance (employee_id, date, time, status)
            VALUES (?, ?, ?, 'Present')
        """,
            (emp_id, current_date, current_time),
        )
        conn.commit()
        conn.close()
        return True, f"Attendance marked for {emp_name} ({emp_id})"
    except sqlite3.IntegrityError:
        # Already logged today
        conn.close()
        return False, f"Attendance already recorded today for {emp_name}"


def get_dashboard_metrics():
    """Calculate key analytics for the dashboard."""
    conn = get_db_connection()
    today = datetime.now().strftime("%Y-%m-%d")

    total_employees = conn.execute(
        "SELECT COUNT(*) as count FROM employees"
    ).fetchone()["count"]
    present_today = conn.execute(
        "SELECT COUNT(*) as count FROM attendance WHERE date = ?", (today,)
    ).fetchone()["count"]

    absent_today = max(0, total_employees - present_today)
    percentage = (
        round((present_today / total_employees * 100), 2)
        if total_employees > 0
        else 0.0
    )

    recent_attendance = conn.execute(
        """
        SELECT a.attendance_id, a.employee_id, e.name, e.department, a.date, a.time, a.status
        FROM attendance a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE a.date = ?
        ORDER BY a.time DESC
        LIMIT 10
    """,
        (today,),
    ).fetchall()

    conn.close()

    return {
        "total_employees": total_employees,
        "present_today": present_today,
        "absent_today": absent_today,
        "percentage": percentage,
        "recent_logs": recent_attendance,
    }


def query_attendance_reports(date=None, month=None, employee_id=None):
    """Filter attendance records by date, month (YYYY-MM), or employee ID."""
    conn = get_db_connection()
    query = """
        SELECT a.attendance_id, a.employee_id, e.name, e.department, a.date, a.time, a.status
        FROM attendance a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE 1=1
    """
    params = []

    if date:
        query += " AND a.date = ?"
        params.append(date)
    if month:
        query += " AND a.date LIKE ?"
        params.append(f"{month}%")
    if employee_id:
        query += " AND a.employee_id = ?"
        params.append(employee_id)

    query += " ORDER BY a.date DESC, a.time DESC"

    records = conn.execute(query, params).fetchall()
    conn.close()
    return records