from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# INPUT FEATURES FOR COST PREDICTION
# ============================================================

class PatientFeatures(BaseModel):

    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    age: int = Field(
        ...,
        ge=65,
        le=100,
        description="Patient age in years (65-100)",
    )

    sex: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 = female, 1 = male",
    )

    # --------------------------------------------------------
    # INDIVIDUAL CHRONIC CONDITIONS
    # 0 = condition absent
    # 1 = condition present
    # --------------------------------------------------------

    alzheimers: int = Field(
        0,
        ge=0,
        le=1,
        description="Alzheimer's disease or related dementia",
    )

    heart_failure: int = Field(
        0,
        ge=0,
        le=1,
        description="Heart failure",
    )

    chronic_kidney_disease: int = Field(
        0,
        ge=0,
        le=1,
        description="Chronic kidney disease",
    )

    cancer: int = Field(
        0,
        ge=0,
        le=1,
        description="Cancer",
    )

    copd: int = Field(
        0,
        ge=0,
        le=1,
        description="Chronic obstructive pulmonary disease",
    )

    depression: int = Field(
        0,
        ge=0,
        le=1,
        description="Depression",
    )

    diabetes: int = Field(
        0,
        ge=0,
        le=1,
        description="Diabetes",
    )

    ischemic_heart_disease: int = Field(
        0,
        ge=0,
        le=1,
        description="Ischemic heart disease",
    )

    osteoporosis: int = Field(
        0,
        ge=0,
        le=1,
        description="Osteoporosis",
    )

    rheumatoid_arthritis: int = Field(
        0,
        ge=0,
        le=1,
        description="Rheumatoid arthritis / osteoarthritis",
    )

    stroke_tia: int = Field(
        0,
        ge=0,
        le=1,
        description="Stroke or transient ischemic attack",
    )

    # --------------------------------------------------------
    # PRIOR HEALTHCARE UTILIZATION
    # --------------------------------------------------------

    prior_inpatient_visits: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of inpatient claims in the prior year",
    )

    prior_outpatient_visits: int = Field(
        ...,
        ge=0,
        le=500,
        description="Number of outpatient claims in the prior year",
    )

    # --------------------------------------------------------
    # PRIOR-YEAR MEDICARE REIMBURSEMENT
    # --------------------------------------------------------

    prior_inpatient_reimbursement: float = Field(
        ...,
        ge=0,
        description="Prior-year inpatient Medicare reimbursement",
    )

    prior_outpatient_reimbursement: float = Field(
        ...,
        ge=0,
        description="Prior-year outpatient Medicare reimbursement",
    )

    prior_carrier_reimbursement: float = Field(
        ...,
        ge=0,
        description="Prior-year carrier/physician Medicare reimbursement",
    )


# ============================================================
# SINGLE PREDICTION RESPONSE
# ============================================================

class PredictionResult(BaseModel):
    predicted_cost: float
    risk_level: str
    percentile: int
    recommendation: str


# ============================================================
# PATIENT RECORD RETURNED FROM DATABASE
# ============================================================

class PatientOut(BaseModel):
    id: int

    age: int
    sex: int
    chronic_condition_count: int

    prior_inpatient_visits: int
    prior_outpatient_visits: int

    prior_reimbursement: float

    predicted_cost: float
    risk_level: str
    percentile: int

    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# BULK JOB RESPONSE
# ============================================================

class JobOut(BaseModel):
    id: str
    status: str

    filename: Optional[str]

    total_records: int
    processed_records: int
    high_risk_found: int

    error_message: Optional[str]

    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ============================================================
# DASHBOARD STATS
# ============================================================

class DashboardStats(BaseModel):
    model_config = {
        "protected_namespaces": ()
    }

    total_patients: int
    avg_predicted_cost: float

    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int

    model_name: str
    model_r2: float
    model_rmse: float


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


# ============================================================
# COST DISTRIBUTION
# ============================================================

class CostBucket(BaseModel):
    label: str
    count: int


# ============================================================
# MODEL COMPARISON
# ============================================================

class ModelComparisonItem(BaseModel):
    name: str
    rmse: float
    mae: float
    r2: float
    is_selected: bool