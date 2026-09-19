from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import schemas
from ..database import Patient, get_db
from ..ml.predictor import get_predictor


router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"],
)


# ============================================================
# DASHBOARD SUMMARY STATISTICS
# ============================================================

@router.get(
    "/stats",
    response_model=schemas.DashboardStats,
)
def stats(
    db: Session = Depends(get_db),
):
    predictor = get_predictor()

    metrics = predictor.model_metrics()

    best_model_name = metrics["best_model"]
    best_model_metrics = metrics["models"][best_model_name]

    total = (
        db.query(func.count(Patient.id)).scalar()
        or 0
    )

    avg_cost = (
        db.query(
            func.avg(Patient.predicted_cost)
        ).scalar()
        or 0.0
    )

    high = (
        db.query(func.count(Patient.id))
        .filter(Patient.risk_level == "high")
        .scalar()
        or 0
    )

    medium = (
        db.query(func.count(Patient.id))
        .filter(Patient.risk_level == "medium")
        .scalar()
        or 0
    )

    low = (
        db.query(func.count(Patient.id))
        .filter(Patient.risk_level == "low")
        .scalar()
        or 0
    )

    return schemas.DashboardStats(
        total_patients=int(total),
        avg_predicted_cost=round(float(avg_cost), 2),
        high_risk_count=int(high),
        medium_risk_count=int(medium),
        low_risk_count=int(low),
        model_name=best_model_name,
        model_r2=round(
            float(best_model_metrics["r2"]),
            4,
        ),
        model_rmse=round(
            float(best_model_metrics["rmse"]),
            2,
        ),
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@router.get(
    "/feature-importance",
    response_model=list[schemas.FeatureImportanceItem],
)
def feature_importance():

    predictor = get_predictor()

    importance = predictor.feature_importance()

    items = []

    # --------------------------------------------------------
    # SUPPORT DICTIONARY FORMAT
    # --------------------------------------------------------
    # Example:
    #
    # {
    #     "prior_reimbursement": 0.6172,
    #     "chronic_condition_count": 0.2072
    # }

    if isinstance(importance, dict):

        for feature, value in importance.items():

            items.append(
                schemas.FeatureImportanceItem(
                    feature=feature,
                    importance=float(value),
                )
            )

    # --------------------------------------------------------
    # SUPPORT LIST FORMAT
    # --------------------------------------------------------
    # Example:
    #
    # [
    #     {
    #         "feature": "prior_reimbursement",
    #         "importance": 0.6172
    #     }
    # ]

    elif isinstance(importance, list):

        for entry in importance:

            items.append(
                schemas.FeatureImportanceItem(
                    feature=entry["feature"],
                    importance=float(entry["importance"]),
                )
            )

    else:

        raise ValueError(
            "Unsupported feature importance format"
        )

    items.sort(
        key=lambda item: item.importance,
        reverse=True,
    )

    return items


# ============================================================
# MODEL COMPARISON
# ============================================================

@router.get(
    "/model-comparison",
    response_model=list[schemas.ModelComparisonItem],
)
def model_comparison():

    predictor = get_predictor()

    metrics = predictor.model_metrics()

    best_model_name = metrics["best_model"]

    results = []

    for model_name, model_metrics in metrics["models"].items():

        results.append(
            schemas.ModelComparisonItem(
                name=model_name,
                rmse=round(
                    float(model_metrics["rmse"]),
                    2,
                ),
                mae=round(
                    float(model_metrics["mae"]),
                    2,
                ),
                r2=round(
                    float(model_metrics["r2"]),
                    4,
                ),
                is_selected=(
                    model_name == best_model_name
                ),
            )
        )

    return results


# ============================================================
# PREDICTED COST DISTRIBUTION
# ============================================================

@router.get(
    "/cost-distribution",
    response_model=list[schemas.CostBucket],
)
def cost_distribution(
    db: Session = Depends(get_db),
):

    buckets = [
        ("$0-1k", 0, 1000),
        ("$1k-3k", 1000, 3000),
        ("$3k-7k", 3000, 7000),
        ("$7k-15k", 7000, 15000),
        ("$15k-30k", 15000, 30000),
        ("$30k+", 30000, None),
    ]

    results = []

    for label, low, high in buckets:

        query = (
            db.query(func.count(Patient.id))
            .filter(
                Patient.predicted_cost >= low
            )
        )

        if high is not None:

            query = query.filter(
                Patient.predicted_cost < high
            )

        count = query.scalar() or 0

        results.append(
            schemas.CostBucket(
                label=label,
                count=int(count),
            )
        )

    return results


# ============================================================
# PATIENT LIST
# ============================================================

@router.get(
    "/patients",
    response_model=list[schemas.PatientOut],
)
def patients(
    risk: Optional[str] = Query(
        None,
        pattern="^(high|medium|low)$",
    ),
    limit: int = Query(
        100,
        ge=1,
        le=500,
    ),
    db: Session = Depends(get_db),
):

    query = db.query(Patient)

    if risk:

        query = query.filter(
            Patient.risk_level == risk
        )

    return (
        query
        .order_by(
            Patient.created_at.desc()
        )
        .limit(limit)
        .all()
    )