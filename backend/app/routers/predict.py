from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import schemas
from ..database import Patient, get_db
from ..ml.predictor import get_predictor


router = APIRouter(
    prefix="/api/predict",
    tags=["predict"],
)


@router.post(
    "",
    response_model=schemas.PredictionResult,
)
def predict_single(
    features: schemas.PatientFeatures,
    db: Session = Depends(get_db),
):
    # ========================================================
    # LOAD TRAINED PREDICTOR
    # ========================================================

    predictor = get_predictor()

    # ========================================================
    # CONVERT API INPUT TO DICTIONARY
    # ========================================================

    feature_dict = features.model_dump()

    # ========================================================
    # RUN COST PREDICTION
    # ========================================================

    result = predictor.predict(
        feature_dict
    )

    # ========================================================
    # DERIVE DATABASE VALUES
    # ========================================================

    chronic_condition_count = sum([
        features.alzheimers,
        features.heart_failure,
        features.chronic_kidney_disease,
        features.cancer,
        features.copd,
        features.depression,
        features.diabetes,
        features.ischemic_heart_disease,
        features.osteoporosis,
        features.rheumatoid_arthritis,
        features.stroke_tia,
    ])

    prior_reimbursement = (
        features.prior_inpatient_reimbursement
        + features.prior_outpatient_reimbursement
        + features.prior_carrier_reimbursement
    )

    # ========================================================
    # SAVE PREDICTION TO DATABASE
    # ========================================================

    patient = Patient(
        source="manual",

        # ----------------------------------------------------
        # DEMOGRAPHICS
        # ----------------------------------------------------

        age=features.age,
        sex=features.sex,

        # ----------------------------------------------------
        # DERIVED CHRONIC CONDITION COUNT
        # ----------------------------------------------------

        chronic_condition_count=chronic_condition_count,

        # ----------------------------------------------------
        # UTILIZATION
        # ----------------------------------------------------

        prior_inpatient_visits=(
            features.prior_inpatient_visits
        ),

        prior_outpatient_visits=(
            features.prior_outpatient_visits
        ),

        # ----------------------------------------------------
        # DERIVED TOTAL REIMBURSEMENT
        # ----------------------------------------------------

        prior_reimbursement=prior_reimbursement,

        # ----------------------------------------------------
        # MODEL OUTPUT
        # ----------------------------------------------------

        predicted_cost=result[
            "predicted_cost"
        ],

        risk_level=result[
            "risk_level"
        ],

        percentile=result[
            "percentile"
        ],
    )

    # ========================================================
    # COMMIT DATABASE RECORD
    # ========================================================

    db.add(patient)
    db.commit()
    db.refresh(patient)

    # ========================================================
    # RETURN PREDICTION
    # ========================================================

    return result