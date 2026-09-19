from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from app.ml.cms_data import (
    load_cms_dataset,
    TARGET_COLUMN,
)


# ============================================================
# MERIDIAN APPLICATION MODEL FEATURES
#
# Frontend/API supplies 18 raw inputs.
#
# Backend derives:
#   chronic_condition_count
#   total_prior_visits
#   prior_reimbursement
#   reimbursement_per_visit
#
# Total model features = 22
# ============================================================

FEATURE_COLUMNS = [
    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------
    "age",
    "sex",

    # --------------------------------------------------------
    # CHRONIC CONDITIONS
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # PRIOR-YEAR REIMBURSEMENT
    # --------------------------------------------------------
    "prior_inpatient_reimbursement",
    "prior_outpatient_reimbursement",
    "prior_carrier_reimbursement",
    "prior_reimbursement",

    # --------------------------------------------------------
    # PRIOR-YEAR UTILIZATION
    # --------------------------------------------------------
    "prior_inpatient_visits",
    "prior_outpatient_visits",
    "total_prior_visits",

    # --------------------------------------------------------
    # ENGINEERED FEATURE
    # --------------------------------------------------------
    "reimbursement_per_visit",
]


# ============================================================
# PATHS
# ============================================================

ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "model.joblib"
)

SCALER_PATH = (
    ARTIFACT_DIR
    / "scaler.joblib"
)

METRICS_PATH = (
    ARTIFACT_DIR
    / "metrics.json"
)

TRAINING_DATA_PATH = (
    ARTIFACT_DIR
    / "training_data.csv"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    predictions = np.asarray(
        y_pred,
        dtype=float,
    )

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Healthcare cost cannot be negative.
    predictions = np.clip(
        predictions,
        0,
        None,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    model_name,
    metrics,
):

    print("\n" + "=" * 70)
    print(model_name.upper())
    print("=" * 70)

    print(
        "RMSE:",
        round(
            metrics["rmse"],
            2,
        ),
    )

    print(
        "MAE:",
        round(
            metrics["mae"],
            2,
        ),
    )

    print(
        "R²:",
        round(
            metrics["r2"],
            4,
        ),
    )


# ============================================================
# LOG-TARGET PREDICTION
# ============================================================

def predict_log_target(
    model,
    X,
):

    log_predictions = (
        model.predict(X)
    )

    # Prevent overflow.
    log_predictions = np.clip(
        log_predictions,
        0,
        20,
    )

    predictions = np.expm1(
        log_predictions
    )

    return np.clip(
        predictions,
        0,
        None,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
):

    feature_importance = {}

    # --------------------------------------------------------
    # TREE-BASED MODELS
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_importances_",
    ):

        values = (
            model.feature_importances_
        )

        feature_importance = {
            feature: float(value)
            for feature, value
            in zip(
                FEATURE_COLUMNS,
                values,
            )
        }

    # --------------------------------------------------------
    # LINEAR REGRESSION
    # --------------------------------------------------------

    elif hasattr(
        model,
        "coef_",
    ):

        values = np.abs(
            np.asarray(
                model.coef_
            ).reshape(-1)
        )

        total = values.sum()

        if total > 0:
            values = (
                values / total
            )

        feature_importance = {
            feature: float(value)
            for feature, value
            in zip(
                FEATURE_COLUMNS,
                values,
            )
        }

    return dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


# ============================================================
# TRAINING
# ============================================================

def train_models():

    print("=" * 70)
    print(
        "MERIDIAN HEALTHCARE COST MODEL TRAINING"
    )
    print("=" * 70)

    print(
        "Prediction objective:"
    )

    print(
        "2008 patient/claims history "
        "-> 2009 healthcare cost"
    )

    print(
        "Application-compatible features:",
        len(FEATURE_COLUMNS),
    )

    # ========================================================
    # 1. LOAD CMS DATA
    # ========================================================

    print(
        "\nLoading CMS dataset..."
    )

    df = load_cms_dataset()

    # ========================================================
    # 2. VERIFY REQUIRED FEATURES
    # ========================================================

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "CMS dataset is missing required "
            "Meridian features:\n"
            + "\n".join(
                missing_columns
            )
        )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            "Target column is missing: "
            + TARGET_COLUMN
        )

    # ========================================================
    # 3. SELECT ONLY MERIDIAN FEATURES
    # ========================================================

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    # ========================================================
    # 4. CLEAN DATA
    # ========================================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    valid_mask = (
        X.notna().all(axis=1)
        & y.notna()
        & np.isfinite(y)
        & (y >= 0)
    )

    X = X.loc[
        valid_mask
    ].copy()

    y = y.loc[
        valid_mask
    ].copy()

    print(
        "\nUsable patients:",
        f"{len(X):,}",
    )

    print(
        "Number of features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Target:",
        TARGET_COLUMN,
    )

    print(
        "\nFeatures used by Meridian:"
    )

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):
        print(
            f"{index:02d}. {feature}"
        )

    # ========================================================
    # 5. CREATE UNTOUCHED TEST SET
    #
    # 20% of the entire dataset is reserved for final testing.
    # It is not used for model selection.
    # ========================================================

    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    # ========================================================
    # 6. TRAIN / VALIDATION SPLIT
    #
    # Development data is divided into training and validation.
    # ========================================================

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_dev,
            y_dev,
            test_size=0.20,
            random_state=42,
        )
    )

    print(
        "\nTraining samples:",
        f"{len(X_train):,}",
    )

    print(
        "Validation samples:",
        f"{len(X_val):,}",
    )

    print(
        "Untouched test samples:",
        f"{len(X_test):,}",
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.describe()
    )

    # ========================================================
    # 7. BASELINE
    # ========================================================

    baseline_value = float(
        y_train.mean()
    )

    baseline_predictions = (
        np.full(
            len(y_val),
            baseline_value,
        )
    )

    baseline_metrics = (
        calculate_metrics(
            y_val,
            baseline_predictions,
        )
    )

    print_metrics(
        "Mean Cost Baseline - Validation",
        baseline_metrics,
    )

    # ========================================================
    # 8. STANDARD SCALER
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_val_scaled = (
        scaler.transform(
            X_val
        )
    )

    # ========================================================
    # STORE EXPERIMENTS
    # ========================================================

    experiments = {}
    trained_models = {}

    # ========================================================
    # 9. LINEAR REGRESSION
    # ========================================================

    print(
        "\nTraining Linear Regression..."
    )

    start = time.time()

    linear_model = (
        LinearRegression()
    )

    linear_model.fit(
        X_train_scaled,
        y_train,
    )

    predictions = (
        linear_model.predict(
            X_val_scaled
        )
    )

    predictions = np.clip(
        predictions,
        0,
        None,
    )

    metrics = calculate_metrics(
        y_val,
        predictions,
    )

    experiments[
        "Linear Regression"
    ] = metrics

    trained_models[
        "Linear Regression"
    ] = {
        "model": linear_model,
        "target_transform": "none",
        "scaler": scaler,
    }

    print_metrics(
        "Linear Regression - Validation",
        metrics,
    )

    print(
        "Training time:",
        round(
            time.time() - start,
            2,
        ),
        "seconds",
    )

    # ========================================================
    # 10. RANDOM FOREST
    # ========================================================

    print(
        "\nTraining Random Forest..."
    )

    start = time.time()

    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(
        X_train,
        y_train,
    )

    predictions = (
        rf_model.predict(
            X_val
        )
    )

    metrics = calculate_metrics(
        y_val,
        predictions,
    )

    experiments[
        "Random Forest"
    ] = metrics

    trained_models[
        "Random Forest"
    ] = {
        "model": rf_model,
        "target_transform": "none",
        "scaler": None,
    }

    print_metrics(
        "Random Forest - Validation",
        metrics,
    )

    print(
        "Training time:",
        round(
            time.time() - start,
            2,
        ),
        "seconds",
    )

    # ========================================================
    # 11. EXTRA TREES
    # ========================================================

    print(
        "\nTraining Extra Trees..."
    )

    start = time.time()

    extra_model = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=20,
        min_samples_leaf=4,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )

    extra_model.fit(
        X_train,
        y_train,
    )

    predictions = (
        extra_model.predict(
            X_val
        )
    )

    metrics = calculate_metrics(
        y_val,
        predictions,
    )

    experiments[
        "Extra Trees"
    ] = metrics

    trained_models[
        "Extra Trees"
    ] = {
        "model": extra_model,
        "target_transform": "none",
        "scaler": None,
    }

    print_metrics(
        "Extra Trees - Validation",
        metrics,
    )

    print(
        "Training time:",
        round(
            time.time() - start,
            2,
        ),
        "seconds",
    )

    # ========================================================
    # 12. GRADIENT BOOSTING
    # ========================================================

    print(
        "\nTraining Gradient Boosting..."
    )

    start = time.time()

    gb_model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.9,
        loss="squared_error",
        random_state=42,
    )

    gb_model.fit(
        X_train,
        y_train,
    )

    predictions = (
        gb_model.predict(
            X_val
        )
    )

    metrics = calculate_metrics(
        y_val,
        predictions,
    )

    experiments[
        "Gradient Boosting"
    ] = metrics

    trained_models[
        "Gradient Boosting"
    ] = {
        "model": gb_model,
        "target_transform": "none",
        "scaler": None,
    }

    print_metrics(
        "Gradient Boosting - Validation",
        metrics,
    )

    print(
        "Training time:",
        round(
            time.time() - start,
            2,
        ),
        "seconds",
    )

    # ========================================================
    # 13. HISTGRADIENTBOOSTING
    # ========================================================

    print(
        "\nTraining HistGradientBoosting..."
    )

    start = time.time()

    hist_model = (
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    hist_model.fit(
        X_train,
        y_train,
    )

    predictions = (
        hist_model.predict(
            X_val
        )
    )

    metrics = calculate_metrics(
        y_val,
        predictions,
    )

    experiments[
        "HistGradientBoosting"
    ] = metrics

    trained_models[
        "HistGradientBoosting"
    ] = {
        "model": hist_model,
        "target_transform": "none",
        "scaler": None,
    }

    print_metrics(
        "HistGradientBoosting - Validation",
        metrics,
    )

    print(
        "Training time:",
        round(
            time.time() - start,
            2,
        ),
        "seconds",
    )

    # ========================================================
    # 14. LOG-TARGET GRADIENT BOOSTING
    # ========================================================

    print(
        "\nTraining Gradient Boosting "
        "with log target..."
    )

    start = time.time()

    y_train_log = np.log1p(
        y_train
    )

    log_gb_model = (
        GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=8,
            subsample=0.9,
            loss="squared_error",
            random_state=42,
        )
    )

    log_gb_model.fit(
        X_train,
        y_train_log,
    )

    predictions = (
        predict_log_target(
            log_gb_model,
            X_val,
        )
    )

    metrics = calculate_metrics(
        y_val,
        predictions,
    )

    experiments[
        "Gradient Boosting Log Target"
    ] = metrics

    trained_models[
        "Gradient Boosting Log Target"
    ] = {
        "model": log_gb_model,
        "target_transform": "log1p",
        "scaler": None,
    }

    print_metrics(
        "Gradient Boosting Log Target "
        "- Validation",
        metrics,
    )

    print(
        "Training time:",
        round(
            time.time() - start,
            2,
        ),
        "seconds",
    )

    # ========================================================
    # 15. LOG-TARGET EXTRA TREES
    # ========================================================

    print(
        "\nTraining Extra Trees "
        "with log target..."
    )

    start = time.time()

    log_extra_model = (
        ExtraTreesRegressor(
            n_estimators=600,
            max_depth=20,
            min_samples_leaf=4,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )
    )

    log_extra_model.fit(
        X_train,
        y_train_log,
    )

    predictions = (
        predict_log_target(
            log_extra_model,
            X_val,
        )
    )

    metrics = calculate_metrics(
        y_val,
        predictions,
    )

    experiments[
        "Extra Trees Log Target"
    ] = metrics

    trained_models[
        "Extra Trees Log Target"
    ] = {
        "model": log_extra_model,
        "target_transform": "log1p",
        "scaler": None,
    }

    print_metrics(
        "Extra Trees Log Target "
        "- Validation",
        metrics,
    )

    print(
        "Training time:",
        round(
            time.time() - start,
            2,
        ),
        "seconds",
    )

    # ========================================================
    # 16. VALIDATION COMPARISON
    # ========================================================

    validation_df = (
        pd.DataFrame(
            experiments
        )
        .T
        .sort_values(
            "r2",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        validation_df.round(4)
    )

    # ========================================================
    # 17. SELECT MODEL USING VALIDATION ONLY
    # ========================================================

    best_model_name = max(
        experiments,
        key=lambda name:
            experiments[name]["r2"],
    )

    best_experiment = (
        trained_models[
            best_model_name
        ]
    )

    target_transform = (
        best_experiment[
            "target_transform"
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL SELECTED USING VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        "Selected model:",
        best_model_name,
    )

    print(
        "Validation R²:",
        round(
            experiments[
                best_model_name
            ]["r2"],
            4,
        ),
    )

    print(
        "Target transformation:",
        target_transform,
    )

    # ========================================================
    # 18. RETRAIN SELECTED MODEL ON FULL DEVELOPMENT SET
    # ========================================================

    print(
        "\nRetraining selected model "
        "on full development set..."
    )

    final_scaler = None

    # --------------------------------------------------------
    # LINEAR REGRESSION
    # --------------------------------------------------------

    if best_model_name == (
        "Linear Regression"
    ):

        final_scaler = (
            StandardScaler()
        )

        X_dev_final = (
            final_scaler.fit_transform(
                X_dev
            )
        )

        X_test_final = (
            final_scaler.transform(
                X_test
            )
        )

        final_model = (
            LinearRegression()
        )

        final_model.fit(
            X_dev_final,
            y_dev,
        )

        test_predictions = (
            final_model.predict(
                X_test_final
            )
        )

        test_predictions = np.clip(
            test_predictions,
            0,
            None,
        )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    elif best_model_name == (
        "Random Forest"
    ):

        final_model = (
            RandomForestRegressor(
                n_estimators=500,
                max_depth=18,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            )
        )

        final_model.fit(
            X_dev,
            y_dev,
        )

        test_predictions = (
            final_model.predict(
                X_test
            )
        )

    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    elif best_model_name == (
        "Extra Trees"
    ):

        final_model = (
            ExtraTreesRegressor(
                n_estimators=600,
                max_depth=20,
                min_samples_leaf=4,
                max_features=0.9,
                random_state=42,
                n_jobs=-1,
            )
        )

        final_model.fit(
            X_dev,
            y_dev,
        )

        test_predictions = (
            final_model.predict(
                X_test
            )
        )

    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    elif best_model_name == (
        "Gradient Boosting"
    ):

        final_model = (
            GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=8,
                subsample=0.9,
                loss="squared_error",
                random_state=42,
            )
        )

        final_model.fit(
            X_dev,
            y_dev,
        )

        test_predictions = (
            final_model.predict(
                X_test
            )
        )

    # --------------------------------------------------------
    # HISTGRADIENTBOOSTING
    # --------------------------------------------------------

    elif best_model_name == (
        "HistGradientBoosting"
    ):

        final_model = (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=400,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=42,
            )
        )

        final_model.fit(
            X_dev,
            y_dev,
        )

        test_predictions = (
            final_model.predict(
                X_test
            )
        )

    # --------------------------------------------------------
    # LOG GRADIENT BOOSTING
    # --------------------------------------------------------

    elif best_model_name == (
        "Gradient Boosting Log Target"
    ):

        final_model = (
            GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=8,
                subsample=0.9,
                loss="squared_error",
                random_state=42,
            )
        )

        final_model.fit(
            X_dev,
            np.log1p(
                y_dev
            ),
        )

        test_predictions = (
            predict_log_target(
                final_model,
                X_test,
            )
        )

    # --------------------------------------------------------
    # LOG EXTRA TREES
    # --------------------------------------------------------

    elif best_model_name == (
        "Extra Trees Log Target"
    ):

        final_model = (
            ExtraTreesRegressor(
                n_estimators=600,
                max_depth=20,
                min_samples_leaf=4,
                max_features=0.9,
                random_state=42,
                n_jobs=-1,
            )
        )

        final_model.fit(
            X_dev,
            np.log1p(
                y_dev
            ),
        )

        test_predictions = (
            predict_log_target(
                final_model,
                X_test,
            )
        )

    else:

        raise ValueError(
            "Unknown selected model: "
            + best_model_name
        )

    test_predictions = np.clip(
        test_predictions,
        0,
        None,
    )

    # ========================================================
    # 19. FINAL UNTOUCHED TEST RESULT
    # ========================================================

    test_metrics = (
        calculate_metrics(
            y_test,
            test_predictions,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL UNTOUCHED TEST RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Selected model:",
        best_model_name,
    )

    print_metrics(
        "Final Test",
        test_metrics,
    )

    # ========================================================
    # 20. FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        get_feature_importance(
            final_model
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )

    if feature_importance:

        for index, (
            feature,
            importance,
        ) in enumerate(
            feature_importance.items(),
            start=1,
        ):

            print(
                f"{index:02d}. "
                f"{feature}: "
                f"{importance:.4f}"
            )

    # ========================================================
    # 21. DEVELOPMENT PREDICTIONS
    #
    # Used ONLY for defining risk thresholds.
    # ========================================================

    if best_model_name == (
        "Linear Regression"
    ):

        development_predictions = (
            final_model.predict(
                X_dev_final
            )
        )

    elif target_transform == (
        "log1p"
    ):

        development_predictions = (
            predict_log_target(
                final_model,
                X_dev,
            )
        )

    else:

        development_predictions = (
            final_model.predict(
                X_dev
            )
        )

    development_predictions = (
        np.clip(
            development_predictions,
            0,
            None,
        )
    )

    # ========================================================
    # 22. RISK THRESHOLDS
    #
    # 60th percentile = medium
    # 85th percentile = high
    # ========================================================

    medium_threshold = float(
        np.percentile(
            development_predictions,
            60,
        )
    )

    high_threshold = float(
        np.percentile(
            development_predictions,
            85,
        )
    )

    print(
        "\nRisk thresholds:"
    )

    print(
        "Medium:",
        round(
            medium_threshold,
            2,
        ),
    )

    print(
        "High:",
        round(
            high_threshold,
            2,
        ),
    )

    # ========================================================
    # 23. SAVE FINAL MODEL
    # ========================================================

    print(
        "\nSaving model artifacts..."
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Always write scaler.joblib.
    #
    # It contains a StandardScaler only when required.
    # Otherwise it contains None.
    # --------------------------------------------------------

    joblib.dump(
        final_scaler,
        SCALER_PATH,
    )

    # ========================================================
    # 24. DASHBOARD MODEL RESULTS
    #
    # Keep the models expected by the existing Meridian UI.
    # ========================================================

    dashboard_models = {}

    for model_name in [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
    ]:

        if model_name in experiments:

            dashboard_models[
                model_name
            ] = experiments[
                model_name
            ]

    # --------------------------------------------------------
    # Important:
    #
    # Dashboard currently expects best_model to exist inside
    # metrics["models"].
    #
    # If Extra Trees / HistGB / log model wins, include it.
    # --------------------------------------------------------

    if (
        best_model_name
        not in dashboard_models
    ):

        dashboard_models[
            best_model_name
        ] = experiments[
            best_model_name
        ]

    # ========================================================
    # 25. SAVE METRICS
    # ========================================================

    metrics_output = {

        "best_model":
            best_model_name,

        "target_transform":
            target_transform,

        "models":
            dashboard_models,

        "validation_models":
            experiments,

        "baseline":
            baseline_metrics,

        "final_test":
            test_metrics,

        "feature_importance":
            feature_importance,

        "medium_threshold":
            medium_threshold,

        "high_threshold":
            high_threshold,

        "n_train":
            int(
                len(X_dev)
            ),

        "n_test":
            int(
                len(X_test)
            ),

        "n_total":
            int(
                len(X)
            ),

        "features":
            FEATURE_COLUMNS,

        "number_of_features":
            len(FEATURE_COLUMNS),

        "target":
            TARGET_COLUMN,

        "dataset":
            "CMS Medicare DE-SynPUF Sample 1",

        "prediction_year":
            2009,

        "feature_year":
            2008,

        "project":
            (
                "Healthcare Cost Prediction "
                "from Claims History"
            ),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_output,
            file,
            indent=4,
        )

    # ========================================================
    # 26. SAVE TRAINING DATA FOR DASHBOARD/PERCENTILES
    # ========================================================

    training_output = (
        X_dev.copy()
    )

    training_output[
        TARGET_COLUMN
    ] = y_dev.to_numpy()

    training_output[
        "predicted_cost"
    ] = development_predictions

    training_output.to_csv(
        TRAINING_DATA_PATH,
        index=False,
    )

    # ========================================================
    # 27. FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "Selected model:",
        best_model_name,
    )

    print(
        "Features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Validation R²:",
        round(
            experiments[
                best_model_name
            ]["r2"],
            4,
        ),
    )

    print(
        "Final test R²:",
        round(
            test_metrics["r2"],
            4,
        ),
    )

    print(
        "Final test RMSE:",
        round(
            test_metrics["rmse"],
            2,
        ),
    )

    print(
        "Final test MAE:",
        round(
            test_metrics["mae"],
            2,
        ),
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nScaler saved to:"
    )

    print(
        SCALER_PATH
    )

    print(
        "\nMetrics saved to:"
    )

    print(
        METRICS_PATH
    )

    print(
        "\nTraining data saved to:"
    )

    print(
        TRAINING_DATA_PATH
    )

    print(
        "\nProject remains:"
    )

    print(
        "2008 claims/patient history "
        "-> 2009 healthcare cost prediction"
    )

    print(
        "=" * 70
    )

    return {
        "best_model":
            best_model_name,

        "validation_metrics":
            experiments[
                best_model_name
            ],

        "test_metrics":
            test_metrics,

        "feature_importance":
            feature_importance,
    }


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_models()