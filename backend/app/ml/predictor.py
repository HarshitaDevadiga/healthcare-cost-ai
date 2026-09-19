from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
TRAINING_DATA_PATH = ARTIFACT_DIR / "training_data.csv"


# ============================================================
# MERIDIAN APPLICATION FEATURES
#
# These are the 22 features that can be produced from the
# existing 18-input Meridian form.
# ============================================================

FEATURE_COLUMNS = [
    "age",
    "sex",

    "chronic_condition_count",

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

    "prior_inpatient_reimbursement",
    "prior_outpatient_reimbursement",
    "prior_carrier_reimbursement",
    "prior_reimbursement",

    "prior_inpatient_visits",
    "prior_outpatient_visits",
    "total_prior_visits",

    "reimbursement_per_visit",
]


CHRONIC_FEATURES = [
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
# PREDICTOR
# ============================================================

class CostPredictor:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Run: python -m app.ml.train"
            )

        if not METRICS_PATH.exists():
            raise FileNotFoundError(
                f"Metrics not found: {METRICS_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
        else:
            self.scaler = None

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            self.metrics = json.load(file)

        self.best_model = self.metrics.get(
            "best_model",
            "Unknown",
        )

        self.target_transform = self.metrics.get(
            "target_transform",
            "none",
        )

        self.medium_threshold = float(
            self.metrics.get(
                "medium_threshold",
                0,
            )
        )

        self.high_threshold = float(
            self.metrics.get(
                "high_threshold",
                0,
            )
        )

        self._training_predictions = None

        self._load_training_distribution()

    # ========================================================
    # TRAINING DISTRIBUTION FOR PERCENTILES
    # ========================================================

    def _load_training_distribution(self):

        if not TRAINING_DATA_PATH.exists():
            return

        try:
            df = pd.read_csv(
                TRAINING_DATA_PATH,
                low_memory=False,
            )

            if "predicted_cost" in df.columns:
                values = pd.to_numeric(
                    df["predicted_cost"],
                    errors="coerce",
                ).dropna()

                if len(values) > 0:
                    self._training_predictions = (
                        values.to_numpy(
                            dtype=float
                        )
                    )

        except Exception:
            self._training_predictions = None

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    def _prepare_features(
        self,
        raw_features,
    ):

        data = dict(raw_features)

        # ----------------------------------------------------
        # Chronic condition count
        # ----------------------------------------------------

        chronic_condition_count = sum(
            int(data.get(feature, 0))
            for feature in CHRONIC_FEATURES
        )

        data[
            "chronic_condition_count"
        ] = chronic_condition_count

        # ----------------------------------------------------
        # Total prior visits
        # ----------------------------------------------------

        inpatient_visits = float(
            data.get(
                "prior_inpatient_visits",
                0,
            )
        )

        outpatient_visits = float(
            data.get(
                "prior_outpatient_visits",
                0,
            )
        )

        total_prior_visits = (
            inpatient_visits
            + outpatient_visits
        )

        data[
            "total_prior_visits"
        ] = total_prior_visits

        # ----------------------------------------------------
        # Total prior reimbursement
        # ----------------------------------------------------

        inpatient_reimbursement = float(
            data.get(
                "prior_inpatient_reimbursement",
                0,
            )
        )

        outpatient_reimbursement = float(
            data.get(
                "prior_outpatient_reimbursement",
                0,
            )
        )

        carrier_reimbursement = float(
            data.get(
                "prior_carrier_reimbursement",
                0,
            )
        )

        prior_reimbursement = (
            inpatient_reimbursement
            + outpatient_reimbursement
            + carrier_reimbursement
        )

        data[
            "prior_reimbursement"
        ] = prior_reimbursement

        # ----------------------------------------------------
        # Reimbursement per visit
        # ----------------------------------------------------

        if total_prior_visits > 0:
            reimbursement_per_visit = (
                prior_reimbursement
                / total_prior_visits
            )
        else:
            reimbursement_per_visit = 0.0

        data[
            "reimbursement_per_visit"
        ] = reimbursement_per_visit

        return data

    # ========================================================
    # CREATE MODEL VECTOR
    # ========================================================

    def _vectorize(
        self,
        raw_features,
    ):

        data = self._prepare_features(
            raw_features
        )

        missing = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in data
        ]

        if missing:
            raise ValueError(
                "Missing model features: "
                + ", ".join(missing)
            )

        row = {
            feature: float(data[feature])
            for feature in FEATURE_COLUMNS
        }

        return pd.DataFrame(
            [row],
            columns=FEATURE_COLUMNS,
        )

    # ========================================================
    # RAW MODEL PREDICTION
    # ========================================================

    def _predict_cost(
        self,
        X,
    ):

        if self.scaler is not None:
            model_input = (
                self.scaler.transform(X)
            )
        else:
            model_input = X

        prediction = float(
            self.model.predict(
                model_input
            )[0]
        )

        # ----------------------------------------------------
        # Reverse log target if required
        # ----------------------------------------------------

        if self.target_transform == "log1p":

            prediction = float(
                np.expm1(
                    np.clip(
                        prediction,
                        0,
                        20,
                    )
                )
            )

        # Healthcare cost cannot be negative.
        return max(
            0.0,
            prediction,
        )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def _risk_level(
        self,
        predicted_cost,
    ):

        if predicted_cost >= self.high_threshold:
            return "high"

        if predicted_cost >= self.medium_threshold:
            return "medium"

        return "low"

    # ========================================================
    # PERCENTILE
    # ========================================================

    def _percentile(
        self,
        predicted_cost,
    ):

        if (
            self._training_predictions is None
            or len(
                self._training_predictions
            ) == 0
        ):
            return 50

        percentile = (
            np.mean(
                self._training_predictions
                <= predicted_cost
            )
            * 100
        )

        return int(
            np.clip(
                round(percentile),
                1,
                99,
            )
        )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    def _recommendation(
        self,
        risk_level,
    ):

        if risk_level == "high":
            return (
                "High predicted healthcare cost. "
                "Consider proactive care management "
                "and resource planning."
            )

        if risk_level == "medium":
            return (
                "Moderate predicted healthcare cost. "
                "Monitor utilization and chronic "
                "condition management."
            )

        return (
            "Lower predicted healthcare cost. "
            "Continue routine monitoring and "
            "preventive care."
        )

    # ========================================================
    # PUBLIC PREDICTION METHOD
    # ========================================================

    def predict(
        self,
        raw_features,
    ):

        X = self._vectorize(
            raw_features
        )

        predicted_cost = (
            self._predict_cost(X)
        )

        risk_level = (
            self._risk_level(
                predicted_cost
            )
        )

        percentile = (
            self._percentile(
                predicted_cost
            )
        )

        return {
            "predicted_cost": round(
                predicted_cost,
                2,
            ),

            "risk_level":
                risk_level,

            "percentile":
                percentile,

            "recommendation":
                self._recommendation(
                    risk_level
                ),
        }

    # ========================================================
    # DASHBOARD HELPERS
    # ========================================================

    def model_metrics(self):
        return self.metrics

    def feature_importance(self):
        return self.metrics.get(
            "feature_importance",
            {},
        )


# ============================================================
# SINGLETON
# ============================================================

_predictor = None


def get_predictor():

    global _predictor

    if _predictor is None:
        _predictor = CostPredictor()

    return _predictor