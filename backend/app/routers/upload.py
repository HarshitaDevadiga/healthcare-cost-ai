import asyncio
import io
import uuid
from datetime import datetime

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .. import schemas
from ..database import Job, Patient, SessionLocal, get_db
from ..ml.predictor import get_predictor


router = APIRouter(
    prefix="/api/upload",
    tags=["upload"],
)


BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 0.35


# ============================================================
# RAW CSV INPUT COLUMNS
# ============================================================
#
# These are the 18 values supplied by the user.
#
# The following four model features are NOT required in CSV:
#
# chronic_condition_count
# total_prior_visits
# prior_reimbursement
# reimbursement_per_visit
#
# They are calculated automatically.
# ============================================================

CSV_INPUT_COLUMNS = [
    "age",
    "sex",

    "alzheimers",
    "heart_failure",
    "chronic_kidney_disease",
    "cancer",
    "copd",
    "depression",
    "diabetes",
    "ischemic_heart_disease",
    "osteoporosis",
    "rheumatoid_arthritis",
    "stroke_tia",

    "prior_inpatient_visits",
    "prior_outpatient_visits",

    "prior_inpatient_reimbursement",
    "prior_outpatient_reimbursement",
    "prior_carrier_reimbursement",
]


# ============================================================
# CHRONIC CONDITION COLUMNS
# ============================================================

CHRONIC_CONDITION_COLUMNS = [
    "alzheimers",
    "heart_failure",
    "chronic_kidney_disease",
    "cancer",
    "copd",
    "depression",
    "diabetes",
    "ischemic_heart_disease",
    "osteoporosis",
    "rheumatoid_arthritis",
    "stroke_tia",
]


# ============================================================
# DERIVE DATABASE SUMMARY VALUES
# ============================================================

def derive_database_values(
    features: dict,
) -> dict:
    """
    Calculate summary values that are stored in the existing
    SQLite Patient table.

    The ML predictor performs equivalent feature engineering
    for the complete 22-feature model input.
    """

    # --------------------------------------------------------
    # CHRONIC CONDITION COUNT
    # --------------------------------------------------------

    chronic_condition_count = sum(
        int(features.get(column, 0))
        for column in CHRONIC_CONDITION_COLUMNS
    )

    # --------------------------------------------------------
    # TOTAL PRIOR VISITS
    # --------------------------------------------------------

    total_prior_visits = (
        int(features["prior_inpatient_visits"])
        + int(features["prior_outpatient_visits"])
    )

    # --------------------------------------------------------
    # TOTAL PRIOR REIMBURSEMENT
    # --------------------------------------------------------

    prior_reimbursement = (
        float(
            features[
                "prior_inpatient_reimbursement"
            ]
        )
        + float(
            features[
                "prior_outpatient_reimbursement"
            ]
        )
        + float(
            features[
                "prior_carrier_reimbursement"
            ]
        )
    )

    # --------------------------------------------------------
    # REIMBURSEMENT PER VISIT
    # --------------------------------------------------------

    if total_prior_visits > 0:
        reimbursement_per_visit = (
            prior_reimbursement
            / total_prior_visits
        )
    else:
        reimbursement_per_visit = 0.0

    return {
        "chronic_condition_count":
            chronic_condition_count,

        "total_prior_visits":
            total_prior_visits,

        "prior_reimbursement":
            prior_reimbursement,

        "reimbursement_per_visit":
            reimbursement_per_visit,
    }


# ============================================================
# BACKGROUND BULK PROCESSING
# ============================================================

async def process_bulk_job(
    job_id: str,
    records: list[dict],
):
    """
    Simulates the cloud architecture:

    CSV Upload
        -> Cloud Storage
        -> Pub/Sub
        -> Cloud Run workers
        -> ML model
        -> Database

    Locally, batches simulate groups of Pub/Sub messages
    processed asynchronously by Cloud Run workers.
    """

    db = SessionLocal()
    predictor = get_predictor()

    try:

        job = (
            db.query(Job)
            .filter(Job.id == job_id)
            .first()
        )

        if not job:
            return

        job.status = "processing"
        db.commit()

        total = len(records)

        processed = 0
        high_risk = 0

        # ====================================================
        # PROCESS RECORDS IN BATCHES
        # ====================================================

        for start in range(
            0,
            total,
            BATCH_SIZE,
        ):

            batch = records[
                start:start + BATCH_SIZE
            ]

            # Simulated Pub/Sub -> Cloud Run latency
            await asyncio.sleep(
                BATCH_DELAY_SECONDS
            )

            for row in batch:

                # ============================================
                # VALIDATE RAW CSV INPUT
                # ============================================

                try:

                    validated = (
                        schemas.PatientFeatures
                        .model_validate(row)
                    )

                    features = (
                        validated.model_dump()
                    )

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    ValidationError,
                ):

                    # Skip invalid patient rows
                    continue

                # ============================================
                # CALCULATE DERIVED VALUES
                # ============================================

                derived = (
                    derive_database_values(
                        features
                    )
                )

                # ============================================
                # MODEL PREDICTION
                # ============================================

                try:

                    result = predictor.predict(
                        features
                    )

                except Exception as exc:

                    print(
                        "Prediction failed for row:",
                        row,
                    )

                    print(
                        "Reason:",
                        exc,
                    )

                    continue

                # ============================================
                # HIGH-RISK COUNTER
                # ============================================

                if (
                    result["risk_level"]
                    == "high"
                ):

                    high_risk += 1

                # ============================================
                # SAVE PATIENT RESULT
                # ============================================

                patient = Patient(
                    source=job_id,

                    # ----------------------------------------
                    # DEMOGRAPHICS
                    # ----------------------------------------

                    age=features["age"],

                    sex=features["sex"],

                    # ----------------------------------------
                    # DERIVED CHRONIC COUNT
                    # ----------------------------------------

                    chronic_condition_count=(
                        derived[
                            "chronic_condition_count"
                        ]
                    ),

                    # ----------------------------------------
                    # UTILIZATION
                    # ----------------------------------------

                    prior_inpatient_visits=(
                        features[
                            "prior_inpatient_visits"
                        ]
                    ),

                    prior_outpatient_visits=(
                        features[
                            "prior_outpatient_visits"
                        ]
                    ),

                    # ----------------------------------------
                    # DERIVED TOTAL REIMBURSEMENT
                    # ----------------------------------------

                    prior_reimbursement=(
                        derived[
                            "prior_reimbursement"
                        ]
                    ),

                    # ----------------------------------------
                    # MODEL OUTPUT
                    # ----------------------------------------

                    predicted_cost=(
                        result[
                            "predicted_cost"
                        ]
                    ),

                    risk_level=(
                        result[
                            "risk_level"
                        ]
                    ),

                    percentile=(
                        result[
                            "percentile"
                        ]
                    ),
                )

                db.add(patient)

                processed += 1

            # ================================================
            # SAVE BATCH + UPDATE PROGRESS
            # ================================================

            job.processed_records = (
                processed
            )

            job.high_risk_found = (
                high_risk
            )

            db.commit()

        # ====================================================
        # COMPLETE JOB
        # ====================================================

        job.status = "completed"

        job.completed_at = (
            datetime.utcnow()
        )

        db.commit()

    # ========================================================
    # JOB FAILURE
    # ========================================================

    except Exception as exc:

        db.rollback()

        job = (
            db.query(Job)
            .filter(Job.id == job_id)
            .first()
        )

        if job:

            job.status = "failed"

            job.error_message = str(
                exc
            )

            db.commit()

    finally:

        db.close()


# ============================================================
# BULK CSV UPLOAD
# ============================================================

@router.post(
    "/bulk",
    response_model=schemas.JobOut,
)
async def upload_bulk(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    # ========================================================
    # FILE TYPE VALIDATION
    # ========================================================

    if (
        not file.filename
        or not file.filename
        .lower()
        .endswith(".csv")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a .csv file."
            ),
        )

    # ========================================================
    # READ CSV
    # ========================================================

    raw = await file.read()

    try:

        df = pd.read_csv(
            io.BytesIO(raw)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not parse the CSV file."
            ),
        ) from exc

    # ========================================================
    # CLEAN COLUMN NAMES
    # ========================================================

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # ========================================================
    # CHECK ONLY THE 18 RAW INPUT COLUMNS
    # ========================================================

    missing = [
        column
        for column in CSV_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise HTTPException(
            status_code=400,
            detail=(
                "CSV is missing required "
                "input columns: "
                + ", ".join(missing)
                + ". Expected columns: "
                + ", ".join(
                    CSV_INPUT_COLUMNS
                )
            ),
        )

    # ========================================================
    # KEEP ONLY THE 18 RAW INPUT COLUMNS
    # ========================================================

    input_df = df[
        CSV_INPUT_COLUMNS
    ].copy()

    # Convert NaN to None so Pydantic can properly validate
    input_df = input_df.astype(
        object
    ).where(
        pd.notnull(input_df),
        None,
    )

    records = input_df.to_dict(
        orient="records"
    )

    # ========================================================
    # MAKE SURE CSV HAS ROWS
    # ========================================================

    if not records:

        raise HTTPException(
            status_code=400,
            detail=(
                "CSV contains no data rows."
            ),
        )

    # ========================================================
    # CREATE BULK PROCESSING JOB
    # ========================================================

    job_id = uuid.uuid4().hex[:10]

    job = Job(
        id=job_id,
        status="queued",

        filename=file.filename,

        total_records=len(records),

        processed_records=0,

        high_risk_found=0,
    )

    db.add(job)

    db.commit()

    db.refresh(job)

    # ========================================================
    # START BACKGROUND PROCESSING
    # ========================================================

    background_tasks.add_task(
        process_bulk_job,
        job_id,
        records,
    )

    return job


# ============================================================
# SAMPLE CSV
# ============================================================

@router.get(
    "/sample-csv"
)
def sample_csv():
    """
    Returns a sample CSV containing the 18 raw patient inputs
    required by the current healthcare cost model.

    The four derived model features are calculated by the
    backend and therefore are not included in the CSV.
    """

    # ========================================================
    # SAMPLE PATIENTS
    # ========================================================

    sample_rows = [

        {
            "age": 72,
            "sex": 0,

            "alzheimers": 0,
            "heart_failure": 1,
            "chronic_kidney_disease": 0,
            "cancer": 0,
            "copd": 0,
            "depression": 0,
            "diabetes": 1,
            "ischemic_heart_disease": 1,
            "osteoporosis": 0,
            "rheumatoid_arthritis": 0,
            "stroke_tia": 0,

            "prior_inpatient_visits": 1,
            "prior_outpatient_visits": 5,

            "prior_inpatient_reimbursement":
                4000,

            "prior_outpatient_reimbursement":
                1800,

            "prior_carrier_reimbursement":
                2200,
        },

        {
            "age": 75,
            "sex": 1,

            "alzheimers": 0,
            "heart_failure": 1,
            "chronic_kidney_disease": 1,
            "cancer": 0,
            "copd": 1,
            "depression": 0,
            "diabetes": 1,
            "ischemic_heart_disease": 1,
            "osteoporosis": 0,
            "rheumatoid_arthritis": 0,
            "stroke_tia": 0,

            "prior_inpatient_visits": 2,
            "prior_outpatient_visits": 8,

            "prior_inpatient_reimbursement":
                6500,

            "prior_outpatient_reimbursement":
                2400,

            "prior_carrier_reimbursement":
                3100,
        },

        {
            "age": 68,
            "sex": 0,

            "alzheimers": 0,
            "heart_failure": 0,
            "chronic_kidney_disease": 0,
            "cancer": 0,
            "copd": 0,
            "depression": 1,
            "diabetes": 1,
            "ischemic_heart_disease": 0,
            "osteoporosis": 1,
            "rheumatoid_arthritis": 0,
            "stroke_tia": 0,

            "prior_inpatient_visits": 0,
            "prior_outpatient_visits": 6,

            "prior_inpatient_reimbursement":
                0,

            "prior_outpatient_reimbursement":
                1600,

            "prior_carrier_reimbursement":
                1900,
        },

        {
            "age": 81,
            "sex": 1,

            "alzheimers": 1,
            "heart_failure": 1,
            "chronic_kidney_disease": 1,
            "cancer": 0,
            "copd": 1,
            "depression": 0,
            "diabetes": 1,
            "ischemic_heart_disease": 1,
            "osteoporosis": 0,
            "rheumatoid_arthritis": 0,
            "stroke_tia": 1,

            "prior_inpatient_visits": 3,
            "prior_outpatient_visits": 12,

            "prior_inpatient_reimbursement":
                9000,

            "prior_outpatient_reimbursement":
                3200,

            "prior_carrier_reimbursement":
                4000,
        },

        {
            "age": 70,
            "sex": 0,

            "alzheimers": 0,
            "heart_failure": 0,
            "chronic_kidney_disease": 0,
            "cancer": 1,
            "copd": 0,
            "depression": 0,
            "diabetes": 0,
            "ischemic_heart_disease": 0,
            "osteoporosis": 0,
            "rheumatoid_arthritis": 1,
            "stroke_tia": 0,

            "prior_inpatient_visits": 1,
            "prior_outpatient_visits": 4,

            "prior_inpatient_reimbursement":
                3500,

            "prior_outpatient_reimbursement":
                1400,

            "prior_carrier_reimbursement":
                1700,
        },
    ]

    # ========================================================
    # CREATE CSV
    # ========================================================

    sample_df = pd.DataFrame(
        sample_rows,
        columns=CSV_INPUT_COLUMNS,
    )

    content = sample_df.to_csv(
        index=False
    )

    return {
        "filename":
            "sample_patients_new_model.csv",

        "content":
            content,
    }