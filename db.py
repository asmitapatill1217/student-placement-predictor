import os
import psycopg2
import psycopg2.extras
from werkzeug.security import generate_password_hash

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            placement_id TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            tenth_percent REAL,
            twelfth_percent REAL,
            degree TEXT,
            cgpa REAL,
            internships INTEGER DEFAULT 0,
            projects INTEGER DEFAULT 0,
            backlogs INTEGER DEFAULT 0,
            communication_skill INTEGER DEFAULT 5,
            extra_curricular INTEGER DEFAULT 0,
            interest TEXT,
            resume_filename TEXT,
            prediction_result TEXT,
            prediction_prob REAL,
            predicted_package REAL,
            admin_remark TEXT,
            status TEXT DEFAULT 'Pending Review',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id SERIAL PRIMARY KEY,
            student_id INTEGER NOT NULL REFERENCES students(id),
            subject TEXT,
            message TEXT NOT NULL,
            status TEXT DEFAULT 'Open',
            admin_reply TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cur.execute("SELECT * FROM admin WHERE username = %s", ("admin",))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO admin (username, password) VALUES (%s, %s)",
            ("admin", generate_password_hash("admin123")),
        )

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized (PostgreSQL/Supabase)")
