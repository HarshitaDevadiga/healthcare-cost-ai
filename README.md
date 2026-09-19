# Meridian — Healthcare Cost Prediction & Resource Planning

A full working implementation of the TEAM 8 project proposal: a cloud-native
healthcare analytics application that predicts per-patient cost from
Medicare-claims-style data and supports resource-planning decisions.

This build runs locally without a GCP account. It demonstrates the application
flow proposed for GCP, while using SQLite and FastAPI background tasks in place
of managed cloud services. See "Moving this to GCP" for the implementation work
still required for a production deployment.

```
healthcare-cost-ai/
├── backend/                  FastAPI service + ML pipeline
│   ├── app/
│   │   ├── main.py           App entrypoint, CORS, router wiring
│   │   ├── database.py       SQLAlchemy models (stand-in for Cloud SQL)
│   │   ├── schemas.py        Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── predict.py    POST /api/predict — single patient scoring
│   │   │   ├── upload.py     POST /api/upload/bulk — async batch pipeline
│   │   │   ├── jobs.py       GET  /api/jobs/{id} — poll batch job status
│   │   │   └── dashboard.py  GET  /api/dashboard/* — stats, charts, patients
│   │   └── ml/
│   │       ├── cms_data.py         Builds the training table from CMS files
│   │       ├── synthetic_data.py   Optional legacy synthetic-data generator
│   │       ├── train.py            Trains & compares Linear/RF/GBM, picks best
│   │       └── predictor.py        Loads model, scores patients, assigns risk
│   └── requirements.txt
├── data/cms/                 CMS DE-SynPUF source files
└── frontend/
    └── index.html             Single-file dashboard (no build step required)
```

## What's implemented

- **ML**: Linear Regression, Random Forest, and Gradient Boosting are trained
  on the included CMS Medicare DE-SynPUF Sample 1 data and compared using R²,
  RMSE, and MAE. The packaged run selected Gradient Boosting with an R² of
  approximately 0.245, RMSE of $7,644, and MAE of $3,574.
- **Single prediction**: `/api/predict` scores one patient and returns a
  cost estimate, risk tier (low/medium/high by percentile), and a resource
  planning recommendation.
- **Async bulk pipeline**: `/api/upload/bulk` accepts a CSV, creates a job,
  and processes it in the background in batches — standing in for
  `Cloud Storage → Pub/Sub → Cloud Run` from the architecture diagram. The
  frontend polls job status live and streams a processing feed.
- **Dashboard API**: portfolio stats, cost distribution, feature importance,
  model comparison, and a filterable patient list — all backed by a real
  database (SQLite locally; swap the connection string for Cloud SQL).
- **Frontend**: a single-file dashboard (no npm/build step) with a live
  patient predictor, drag-and-drop bulk upload with a live job console,
  a filterable patient table, and a model-insights view.

## Requirements

- Python 3.10+
- A modern browser
- No Node.js or build tools needed for the frontend

## Step-by-step: running it locally

### Windows PowerShell

From the project root:

```powershell
.\setup.ps1
.\start-app.ps1
```

The start script launches both local services and opens the dashboard. Stop
services created by the script with:

```powershell
.\stop-app.ps1
```

If Python is installed in a nonstandard location, pass its full path:

```powershell
.\setup.ps1 -PythonPath 'C:\path\to\python.exe'
```

### Manual or macOS/Linux setup

**1. Start the backend**

```bash
cd healthcare-cost-ai/backend
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

(Or just run `./run.sh`, which does all of the above for you.)

The repository includes a trained model in `backend/app/ml/artifacts/`, so it
normally loads immediately. If `model.joblib` or `metrics.json` is removed,
the next backend startup retrains from the included CMS files. Leave this
terminal running.

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Confirm it's healthy: open http://localhost:8000/api/health — you should see
`{"status":"ok"}`. Interactive API docs are at http://localhost:8000/docs.

**2. Start the frontend**

Open a **second terminal** (leave the backend running in the first):

```bash
cd healthcare-cost-ai/frontend
python3 -m http.server 5500
```

**3. Open the app**

Go to **http://localhost:5500** in your browser.

That's it — the dashboard talks to the API at `http://127.0.0.1:8000` by
default. If you serve the backend from a different host/port, open
`frontend/index.html` and set `window.API_BASE` before the main script runs,
e.g. add this near the top of the `<body>`:

```html
<script>window.API_BASE = 'http://your-backend-host:8000';</script>
```

**4. Try it out**

- **Predict Cost**: adjust the sliders and click "Run cost prediction" to
  score a single patient.
- **Bulk Upload**: click "Download a sample CSV" to get a template with the
  right columns, then drag it back onto the dropzone to watch the async
  pipeline process it batch by batch.
- **Patients**: every patient you've scored (single or bulk) shows up here,
  filterable by risk tier.
- **Model Insights**: see how Linear Regression, Random Forest, and Gradient
  Boosting compare, and which features drive predicted cost the most.

## Training data

The active pipeline in `backend/app/ml/cms_data.py` uses the included CMS
DE-SynPUF source files. It derives six 2008 features and a 2009 reimbursement
target for 95,663 beneficiaries. `synthetic_data.py` is retained as an optional
legacy generator but is not imported by the current training code.

To retrain deliberately:

```bash
cd backend
python -m app.ml.train
```

Retraining overwrites the model, scaler, metrics, and processed training-data
artifacts. Back them up first if you need to preserve the packaged run.

## Moving this to GCP (matches the proposal's architecture)

| Local (this build) | GCP (production) |
|---|---|
| SQLite (`database.py`) | Cloud SQL (PostgreSQL) — change `DATABASE_URL` |
| FastAPI `BackgroundTasks` (`upload.py`) | Pub/Sub + Cloud Run workers |
| Local file upload | Cloud Storage bucket, triggering a Cloud Function/Pub/Sub message |
| `joblib`-saved scikit-learn model | Same model, optionally served via Vertex AI Endpoints |
| SQLite dashboard queries | Periodic export/aggregate into BigQuery for Looker Studio/Power BI |
| Static `index.html` | Cloud Run (containerized) or Firebase Hosting |

This is an architectural migration path, not an active cloud integration.
Production work must still implement authentication, secrets, Cloud Storage
uploads, Pub/Sub delivery and retries, idempotent workers, Cloud SQL connection
management, observability, data governance, and deployment infrastructure.

## Team responsibilities mapped to this codebase

| Member | Owns |
|---|---|
| Dataset & Database | `cms_data.py`, CMS source files, `database.py` schema |
| Cost Prediction | `train.py`, `predictor.py`, model comparison/tuning |
| GCP Architecture | `upload.py` async pipeline → swap for real Pub/Sub + Cloud Run |
| Dashboard | `frontend/index.html` |
| Integration & Quality | End-to-end testing, `main.py` wiring, deployment |
