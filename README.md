# Campus Placement Predictor

A full-stack student placement prediction system built with **Python, Flask, pandas, NumPy,
scikit-learn, Matplotlib and Seaborn**.

- **Student side:** register (10th %, 12th %, degree, CGPA, internships, projects, backlogs,
  communication skill, interest, resume/CV upload) → get an instant ML placement prediction with
  a confidence score → a unique **Placement ID** is generated → track admin remarks → raise
  complaints/queries.
- **Admin (Placement Cell) side:** login → dashboard with live stats and graphs → view every
  student, open their resume, add a remark and change status (Shortlisted / Rejected / Interview
  Scheduled) → date-wise reports → complaint box with reply + status.
- **ML model:** a `RandomForestClassifier` (scikit-learn) trained on a labeled placement dataset,
  ~83% test accuracy. Graphs (pie chart, CGPA box plot, correlation heatmap, internship-vs-rate
  bar chart, package histogram) are generated with Matplotlib/Seaborn and shown on both dashboards.

## Dataset note

Kaggle wasn't reachable from the build environment used to put this together, so
`data/generate_dataset.py` creates a synthetic 800-row dataset with the **same column structure**
as the well-known Kaggle "Campus Placement" datasets (10th %, 12th %, CGPA, internships, projects,
backlogs, communication skill, extracurriculars, placement outcome, package). The correlations are
built in on purpose so the model behaves realistically (higher CGPA/internships → higher placement
chance, more backlogs → lower chance).

**To use a real Kaggle dataset instead:** download any "Campus Placement Prediction" dataset from
Kaggle, rename/remap its columns to match `data/student_placement_dataset.csv`
(`tenth_percent, twelfth_percent, degree, cgpa, internships, projects, backlogs,
communication_skill, extra_curricular, interest, placed, package_lpa`), replace the CSV, then
re-run `python model/train_model.py` to retrain on it.

## Project structure

```
placement_predictor/
├── app.py                     # Flask app: all routes
├── db.py                      # SQLite schema + connection helper
├── requirements.txt
├── vercel.json                 # Vercel serverless config (see Deployment note below)
├── data/
│   ├── generate_dataset.py     # builds the synthetic dataset
│   └── student_placement_dataset.csv
├── model/
│   ├── train_model.py          # trains the model + builds graphs
│   ├── placement_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── accuracy.txt
├── static/
│   ├── css/style.css
│   ├── graphs/                 # generated chart PNGs
│   └── uploads/                # student resumes land here
└── templates/                  # all HTML pages (Jinja2)
```

## Run it locally

```bash
cd placement_predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (only needed once — the trained model + dataset are already included)
python data/generate_dataset.py
python model/train_model.py

python db.py                    # creates placement.db with the default admin account
python app.py
```

Visit **http://127.0.0.1:5000**.

- **Default admin login:** `admin` / `admin123` — change this before going live (edit `db.py` or
  update the row in the `admin` table).
- Student accounts are created via the **Register** page.

## Pushing to GitHub

```bash
cd placement_predictor
git init
git add .
git commit -m "Campus placement predictor - Flask + ML"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`placement.db` and anything in `static/uploads/` are excluded via `.gitignore` so you don't commit
student data or resumes to a public repo.

## Deploying

**Important limitation to know before you deploy:** this app writes to a local SQLite file
(`placement.db`) and saves uploaded resumes to disk (`static/uploads/`). **Vercel's Python runtime
is serverless** — its filesystem is read-only/ephemeral at request time, so SQLite writes and
resume uploads will not persist between requests there. A `vercel.json` is included so it will
*run and let people click through it*, but registrations and uploads won't reliably survive.

For a platform that actually keeps your data and file uploads, use one of these instead — all free
to start and just as quick to set up:

- **Render** (recommended): New → Web App → connect the repo → build command
  `pip install -r requirements.txt` → start command `gunicorn app:app` → add a free persistent disk
  if you want uploads/DB to survive restarts.
- **Railway**: similar one-click deploy from a GitHub repo, persistent volume available.
- **PythonAnywhere**: good free tier for small Flask + SQLite apps.

If you do want it on Vercel specifically (e.g. because of a course requirement), swap SQLite for a
hosted database (Vercel Postgres, Supabase, or PlanetScale) and swap local file uploads for a
storage bucket (Vercel Blob, S3, or Cloudinary) — the route logic in `app.py` stays the same, only
`db.py`'s connection and the file-save lines in the `/register` route need to point at the hosted
services instead of the local disk.

## Retraining the model

If you change the dataset or want to tune the model, edit `model/train_model.py` and re-run it —
it overwrites `placement_model.pkl`, `scaler.pkl`, `accuracy.txt` and all five graph PNGs in
`static/graphs/`. Restart the Flask app afterward to pick up the new model.
