# 🏥 Healthcare Cost Prediction & Resource Planning

An end-to-end machine learning application for predicting patient healthcare costs from Medicare/CMS claims-style data and transforming those predictions into actionable insights for healthcare financial and resource planning.

The project combines **machine learning, healthcare analytics, REST APIs, data engineering, and an interactive web dashboard** in a production-oriented architecture.

---

## 🎯 Project Overview

Healthcare organizations manage large volumes of patient, claims, reimbursement, and utilization data. Accurately identifying patients likely to generate higher future healthcare costs can support better budgeting, resource allocation, and proactive care planning.

This project builds a machine learning system that estimates a patient's **next-year healthcare cost** using historical demographic, chronic-condition, utilization, and reimbursement information.

The application provides:

* Individual patient cost prediction
* Patient cost-risk categorization
* Bulk prediction workflows
* Healthcare utilization analytics
* Resource-planning insights
* Interactive web dashboard
* REST API using FastAPI
* Reusable trained ML artifacts
* Architecture designed for future cloud deployment

---

## 🏗️ System Architecture

```text
                 CMS Healthcare Data
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
              Machine Learning Models
                         │
                         ▼
              Trained Model Artifacts
                         │
                         ▼
                  FastAPI Backend
                  /      |       \
                 /       |        \
        Prediction    Upload     Analytics
           API          API        API
                 \       |       /
                  \      |      /
                         ▼
                   Web Dashboard
                         │
                         ▼
             Healthcare Cost Insights
```

### Planned Cloud Architecture

```text
User / Dashboard
       │
       ▼
    FastAPI
       │
       ├────────► Cloud Storage
       │
       ├────────► Pub/Sub
       │
       ▼
    Cloud Run
       │
       ▼
Machine Learning Model
       │
       ├────────► Cloud SQL
       │
       └────────► BigQuery
                         │
                         ▼
                 Analytics Dashboard
```

The cloud architecture is designed to support scalable batch processing, asynchronous prediction jobs, persistent storage, and analytics workloads.

---

## 📊 Dataset

The project uses **CMS/Medicare claims-style healthcare data**, including beneficiary summary information and inpatient/outpatient claims.

The raw datasets are intentionally **not stored in this GitHub repository** because of their size.

Examples of source information used by the model include:

* Patient demographics
* Chronic conditions
* Inpatient utilization
* Outpatient utilization
* Historical reimbursement
* Diagnosis/procedure utilization
* Previous healthcare encounters

The data-processing pipeline converts raw healthcare records into patient-level machine learning features.

---

## 🧠 Machine Learning

The prediction problem is formulated as a **regression task**:

```text
Patient History → ML Model → Predicted Next-Year Healthcare Cost
```

Multiple regression approaches were evaluated during model development, including:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting
* Extra Trees
* Histogram Gradient Boosting

The training pipeline performs:

```text
CMS Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Comparison
   ↓
Evaluation
   ↓
Best Model Selection
   ↓
Model Serialization
```

The selected model and supporting artifacts are stored using `joblib` so predictions can be served directly through the FastAPI application.

---

## 🔬 Model Features

The model incorporates several categories of patient information.

### Demographics

* Age
* Sex

### Chronic Conditions

Examples include:

* Alzheimer's disease
* Heart failure
* Chronic kidney disease
* Cancer
* COPD
* Depression
* Diabetes
* Ischemic heart disease
* Osteoporosis
* Rheumatoid arthritis
* Stroke/TIA

A `chronic_condition_count` feature summarizes the patient's overall chronic disease burden.

### Healthcare Utilization

Features include:

* Prior inpatient visits
* Prior outpatient visits
* Total prior healthcare visits
* Diagnosis utilization
* Procedure utilization

### Historical Cost Information

The system incorporates previous healthcare reimbursement information such as:

* Inpatient reimbursement
* Outpatient reimbursement
* Carrier reimbursement
* Total prior reimbursement
* Reimbursement per visit

These features allow the model to combine clinical burden with historical healthcare utilization.

---

## 📈 Model Evaluation

The final model is evaluated using standard regression metrics:

| Metric   |        Result |
| -------- | ------------: |
| **R²**   |    **0.2941** |
| **RMSE** | **$7,389.08** |
| **MAE**  | **$3,466.24** |

### Metric Interpretation

**R²** measures how much variation in future healthcare cost is explained by the model.

**RMSE** penalizes large prediction errors more heavily and is useful because healthcare cost distributions frequently contain high-cost cases.

**MAE** represents the average absolute difference between predicted and actual healthcare costs.

The project emphasizes transparent evaluation rather than presenting the model as a clinical decision-making system.

---

## ⚠️ Cost Risk Stratification

Predicted healthcare costs can also be translated into operational risk categories.

Example thresholds generated from the training data:

```text
Medium Cost Risk: $4,240.89
High Cost Risk:   $7,964.38
```

This allows financial planners or utilization-management teams to identify groups of patients who may require additional review or resource planning.

---

## 🌐 FastAPI Backend

The backend is implemented using **FastAPI** and provides APIs for machine learning inference, file uploads, asynchronous processing, dashboard analytics, and application health monitoring.

Main application:

```text
backend/app/main.py
```

The API automatically initializes the application database and loads the trained prediction model when the server starts.

### API Documentation

When the backend is running:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides interactive Swagger/OpenAPI documentation.

### Health Check

```text
GET /api/health
```

Example response:

```json
{
  "status": "ok"
}
```

---

## 🖥️ Frontend Dashboard

The application includes a web-based dashboard for interacting with the prediction system.

The frontend supports workflows such as:

* Entering patient information
* Generating healthcare-cost predictions
* Reviewing patient risk categories
* Uploading data
* Viewing analytics
* Supporting resource-planning decisions

Frontend entry point:

```text
frontend/index.html
```

---

## 📸 Application Screenshots

Add screenshots of the running application here.

### Healthcare Cost Dashboard

```text
docs/images/dashboard.png
```

### Patient Cost Prediction

```text
docs/images/prediction.png
```

### Analytics / Resource Planning

```text
docs/images/analytics.png
```

### FastAPI Documentation

```text
docs/images/api-docs.png
```

> Screenshots will be added as the interface and deployment are finalized.

---

## 🛠️ Technology Stack

| Area                    | Technology            |
| ----------------------- | --------------------- |
| **Language**            | Python                |
| **Machine Learning**    | Scikit-learn          |
| **Data Processing**     | Pandas, NumPy         |
| **Backend**             | FastAPI               |
| **API Server**          | Uvicorn               |
| **Model Storage**       | Joblib                |
| **Database**            | SQLite                |
| **Frontend**            | HTML, CSS, JavaScript |
| **API Documentation**   | Swagger / OpenAPI     |
| **Version Control**     | Git & GitHub          |
| **Cloud Platform**      | Google Cloud Platform |
| **Future Deployment**   | Cloud Run             |
| **Cloud Storage**       | Google Cloud Storage  |
| **Messaging**           | Pub/Sub               |
| **Analytics**           | BigQuery              |
| **Relational Database** | Cloud SQL             |

---

## 📁 Project Structure

```text
healthcare-cost-ai/
│
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   ├── artifacts/
│   │   │   │   ├── metrics.json
│   │   │   │   ├── model.joblib
│   │   │   │   └── scaler.joblib
│   │   │   ├── cms_data.py
│   │   │   ├── cms_halfyear_data.py
│   │   │   ├── predictor.py
│   │   │   ├── synthetic_data.py
│   │   │   └── train.py
│   │   │
│   │   ├── routers/
│   │   │   ├── dashboard.py
│   │   │   ├── jobs.py
│   │   │   ├── predict.py
│   │   │   └── upload.py
│   │   │
│   │   ├── database.py
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── requirements.txt
│   └── run.sh
│
├── data/
│   └── cms/
│
├── frontend/
│   └── index.html
│
├── .gitignore
├── README.md
├── setup.ps1
├── start-app.ps1
└── stop-app.ps1
```

---

## 🚀 Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/HarshitaDevadiga/healthcare-cost-ai.git
cd healthcare-cost-ai
```

### 2. Create a Virtual Environment

Windows:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r backend/requirements.txt
```

### 4. Add CMS Data

Place the required CMS data files inside:

```text
data/cms/
```

Raw datasets are excluded from GitHub because of their size.

### 5. Train the Model

```powershell
cd backend
python -m app.ml.train
cd ..
```

The training pipeline generates artifacts such as:

```text
model.joblib
scaler.joblib
metrics.json
```

### 6. Start the Application

From the project root:

```powershell
.\start-app.ps1
```

Once running:

```text
Dashboard:
http://127.0.0.1:5500/

API:
http://127.0.0.1:8000/

API Documentation:
http://127.0.0.1:8000/docs
```

To stop the application:

```powershell
.\stop-app.ps1
```

---

## ☁️ Google Cloud Deployment Roadmap

The local application is designed to evolve into a cloud-native healthcare analytics platform.

### Phase 1 — Containerization

Containerize the FastAPI backend using Docker.

```text
FastAPI
   ↓
Docker
   ↓
Artifact Registry
   ↓
Cloud Run
```

### Phase 2 — Cloud Data Storage

Use:

* **Cloud Storage** for raw healthcare files
* **Cloud SQL** for application and transactional data
* **BigQuery** for large-scale analytical workloads

### Phase 3 — Asynchronous Processing

Bulk healthcare prediction jobs can be processed through:

```text
Upload
   ↓
Cloud Storage
   ↓
Pub/Sub
   ↓
Cloud Run Worker
   ↓
ML Prediction
   ↓
Database / BigQuery
```

This prevents large prediction jobs from blocking interactive application requests.

### Phase 4 — Production ML

Future versions could incorporate:

* Vertex AI model deployment
* Model monitoring
* Automated retraining
* Model versioning
* Feature monitoring
* Prediction drift detection

---

## 🔮 Future Improvements

Planned enhancements include:

* Docker containerization
* Google Cloud Run deployment
* Cloud SQL integration
* BigQuery analytics
* Pub/Sub asynchronous processing
* Vertex AI integration
* Model monitoring and drift detection
* Improved feature engineering
* Hyperparameter optimization
* Explainable AI using SHAP
* CI/CD using GitHub Actions
* Authentication and role-based access
* Improved dashboard visualizations

---

## ⚕️ Disclaimer

This project is intended for **educational, research, and portfolio purposes**.

The predictions generated by this application should **not be used for medical diagnosis, treatment decisions, insurance decisions, or direct clinical decision-making**.

The application demonstrates how machine learning and cloud technologies can be applied to healthcare financial and operational analytics.

---

## 👩‍💻 Author

**Harshita Ganesh Devadiga**

Master's in Data Science
University of Houston

Interests:

* Machine Learning
* Artificial Intelligence
* Data Science
* Healthcare Analytics
* NLP / Generative AI
* Cloud Computing

GitHub: `HarshitaDevadiga`

---

## ⭐ Project Goal

The goal of this project is to demonstrate an end-to-end ML engineering workflow that goes beyond training a model:

**Healthcare Data → Feature Engineering → Machine Learning → API → Web Application → Resource Planning → Cloud Architecture**

It showcases the integration of **data science, machine learning engineering, backend development, healthcare analytics, and cloud computing** in a single portfolio project.
