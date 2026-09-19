from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]
CMS_DIR = BASE_DIR / "data" / "cms"

BENEFICIARY_FILE = (
    CMS_DIR
    / "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv"
)

INPATIENT_FILE = (
    CMS_DIR
    / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
)

OUTPATIENT_FILE = (
    CMS_DIR
    / "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"
)


# ============================================================
# TIME WINDOWS
# ============================================================

FEATURE_START = pd.Timestamp("2009-01-01")
FEATURE_END = pd.Timestamp("2009-06-30")

TARGET_START = pd.Timestamp("2009-07-01")
TARGET_END = pd.Timestamp("2009-12-31")


# ============================================================
# CHRONIC CONDITIONS
# ============================================================

CHRONIC_CONDITION_MAP = {
    "SP_ALZHDMTA": "alzheimers",
    "SP_CHF": "heart_failure",
    "SP_CHRNKIDN": "chronic_kidney_disease",
    "SP_CNCR": "cancer",
    "SP_COPD": "copd",
    "SP_DEPRESSN": "depression",
    "SP_DIABETES": "diabetes",
    "SP_ISCHMCHT": "ischemic_heart_disease",
    "SP_OSTEOPRS": "osteoporosis",
    "SP_RA_OA": "rheumatoid_arthritis",
    "SP_STRKETIA": "stroke_tia",
}

CONDITION_COLUMNS = list(
    CHRONIC_CONDITION_MAP.values()
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    # Demographics
    "age",
    "sex",

    # Chronic conditions
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

    # Utilization
    "inpatient_claim_count",
    "outpatient_claim_count",
    "total_claim_count",

    # Cost history
    "inpatient_cost_total",
    "inpatient_cost_mean",
    "inpatient_cost_max",
    "inpatient_cost_std",

    "outpatient_cost_total",
    "outpatient_cost_mean",
    "outpatient_cost_max",
    "outpatient_cost_std",

    "first_half_cost",

    # Diagnoses
    "inpatient_diagnosis_count",
    "inpatient_unique_diagnosis_count",

    "outpatient_diagnosis_count",
    "outpatient_unique_diagnosis_count",

    "total_diagnosis_count",

    # Procedures
    "inpatient_procedure_count",
    "inpatient_unique_procedure_count",

    "outpatient_procedure_count",
    "outpatient_unique_procedure_count",

    "total_procedure_count",

    # Duration
    "inpatient_days_mean",
    "inpatient_days_max",

    "outpatient_days_mean",
    "outpatient_days_max",

    # Intensity
    "cost_per_claim",
    "inpatient_cost_per_claim",
    "outpatient_cost_per_claim",

    "inpatient_claim_ratio",
    "outpatient_claim_ratio",

    # Recency
    "days_since_last_claim",

    # Monthly utilization
    "jan_claim_count",
    "feb_claim_count",
    "mar_claim_count",
    "apr_claim_count",
    "may_claim_count",
    "jun_claim_count",

    "jan_cost",
    "feb_cost",
    "mar_cost",
    "apr_cost",
    "may_cost",
    "jun_cost",

    # Recent trends
    "recent_3m_cost",
    "early_3m_cost",
    "recent_to_early_cost_ratio",

    "recent_3m_claims",
    "early_3m_claims",

    # Interactions
    "chronic_cost_interaction",
    "chronic_claim_interaction",
]


TARGET_COLUMN = "second_half_cost"


# ============================================================
# BASIC HELPERS
# ============================================================

def _check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required CMS file not found:\n{path}"
        )


def _numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


def _parse_date(series):

    values = (
        series
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.strip()
    )

    return pd.to_datetime(
        values,
        format="%Y%m%d",
        errors="coerce",
    )


def _safe_divide(
    numerator,
    denominator,
):

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).fillna(0)

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    ).fillna(0)

    return np.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )


def _condition_binary(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return (
        values == 1
    ).astype(int)


# ============================================================
# DETECT CLAIM COST COLUMN
# ============================================================

def _find_cost_column(df):

    candidates = [
        "CLM_PMT_AMT",
        "NCH_PRMRY_PYR_CLM_PD_AMT",
    ]

    for column in candidates:

        if column in df.columns:

            return column

    return None


# ============================================================
# DIAGNOSIS / PROCEDURE COLUMNS
# ============================================================

def _diagnosis_columns(df):

    result = []

    for column in df.columns:

        upper = column.upper()

        if (
            "ICD9_DGNS" in upper
            or "DGNS_CD" in upper
        ):
            result.append(column)

    return result


def _procedure_columns(df):

    result = []

    for column in df.columns:

        upper = column.upper()

        if (
            "ICD9_PRCDR" in upper
            or "PRCDR_CD" in upper
            or "HCPCS_CD" in upper
        ):
            result.append(column)

    return result


# ============================================================
# CLAIM-LEVEL DATA
# ============================================================

def _prepare_claims(
    df,
    claim_type,
):

    df = df.copy()

    if "CLM_FROM_DT" not in df.columns:

        raise ValueError(
            f"{claim_type}: CLM_FROM_DT not found."
        )

    df["_claim_date"] = _parse_date(
        df["CLM_FROM_DT"]
    )

    cost_column = _find_cost_column(
        df
    )

    if cost_column is None:

        print(
            f"WARNING: No claim payment column "
            f"found for {claim_type}."
        )

        df["_claim_cost"] = 0.0

    else:

        print(
            f"{claim_type} cost column:",
            cost_column,
        )

        df["_claim_cost"] = _numeric(
            df[cost_column]
        )

    return df


# ============================================================
# CLAIM COUNT
# ============================================================

def _claim_counts(
    df,
    output_column,
):

    if df.empty:

        return pd.DataFrame(
            columns=[
                "DESYNPUF_ID",
                output_column,
            ]
        )

    if "CLM_ID" in df.columns:

        result = (
            df
            .groupby(
                "DESYNPUF_ID"
            )["CLM_ID"]
            .nunique()
            .rename(
                output_column
            )
            .reset_index()
        )

    else:

        result = (
            df
            .groupby(
                "DESYNPUF_ID"
            )
            .size()
            .rename(
                output_column
            )
            .reset_index()
        )

    return result


# ============================================================
# CREATE CLAIM-LEVEL COST TABLE
# ============================================================

def _claim_level_cost(df):

    if df.empty:

        return pd.DataFrame(
            columns=[
                "DESYNPUF_ID",
                "CLM_ID",
                "_claim_date",
                "_claim_cost",
            ]
        )

    if "CLM_ID" in df.columns:

        result = (
            df
            .groupby(
                [
                    "DESYNPUF_ID",
                    "CLM_ID",
                ],
                as_index=False,
            )
            .agg(
                _claim_date=(
                    "_claim_date",
                    "min",
                ),
                _claim_cost=(
                    "_claim_cost",
                    "max",
                ),
            )
        )

    else:

        result = df[
            [
                "DESYNPUF_ID",
                "_claim_date",
                "_claim_cost",
            ]
        ].copy()

        result["CLM_ID"] = np.arange(
            len(result)
        )

    return result


# ============================================================
# COST STATISTICS
# ============================================================

def _cost_statistics(
    claim_level_df,
    prefix,
):

    columns = [
        "DESYNPUF_ID",
        f"{prefix}_cost_total",
        f"{prefix}_cost_mean",
        f"{prefix}_cost_max",
        f"{prefix}_cost_std",
    ]

    if claim_level_df.empty:

        return pd.DataFrame(
            columns=columns
        )

    result = (
        claim_level_df
        .groupby(
            "DESYNPUF_ID"
        )["_claim_cost"]
        .agg(
            ["sum", "mean", "max", "std"]
        )
        .reset_index()
    )

    result = result.rename(
        columns={
            "sum":
                f"{prefix}_cost_total",

            "mean":
                f"{prefix}_cost_mean",

            "max":
                f"{prefix}_cost_max",

            "std":
                f"{prefix}_cost_std",
        }
    )

    result[
        f"{prefix}_cost_std"
    ] = (
        result[
            f"{prefix}_cost_std"
        ]
        .fillna(0)
    )

    return result


# ============================================================
# DIAGNOSIS / PROCEDURE FEATURES
# ============================================================

def _code_features(
    df,
    code_columns,
    prefix,
):

    output_columns = [
        "DESYNPUF_ID",
        f"{prefix}_count",
        f"{prefix}_unique_count",
    ]

    if (
        df.empty
        or not code_columns
    ):

        return pd.DataFrame(
            columns=output_columns
        )

    work = df[
        ["DESYNPUF_ID"]
        + code_columns
    ].copy()

    long_df = work.melt(
        id_vars=[
            "DESYNPUF_ID"
        ],
        value_vars=code_columns,
        value_name="code",
    )

    long_df["code"] = (
        long_df["code"]
        .astype(str)
        .str.strip()
    )

    invalid = {
        "",
        "nan",
        "None",
        "0",
        "0.0",
    }

    long_df = long_df[
        ~long_df[
            "code"
        ].isin(
            invalid
        )
    ]

    if long_df.empty:

        return pd.DataFrame(
            columns=output_columns
        )

    total = (
        long_df
        .groupby(
            "DESYNPUF_ID"
        )
        .size()
        .rename(
            f"{prefix}_count"
        )
    )

    unique = (
        long_df
        .groupby(
            "DESYNPUF_ID"
        )["code"]
        .nunique()
        .rename(
            f"{prefix}_unique_count"
        )
    )

    return pd.concat(
        [
            total,
            unique,
        ],
        axis=1,
    ).reset_index()


# ============================================================
# CLAIM DURATION
# ============================================================

def _duration_features(
    df,
    prefix,
):

    output_columns = [
        "DESYNPUF_ID",
        f"{prefix}_days_mean",
        f"{prefix}_days_max",
    ]

    if (
        df.empty
        or "CLM_FROM_DT" not in df.columns
        or "CLM_THRU_DT" not in df.columns
    ):

        return pd.DataFrame(
            columns=output_columns
        )

    work = df[
        [
            "DESYNPUF_ID",
            "CLM_FROM_DT",
            "CLM_THRU_DT",
        ]
    ].copy()

    start = _parse_date(
        work["CLM_FROM_DT"]
    )

    end = _parse_date(
        work["CLM_THRU_DT"]
    )

    work["_days"] = (
        end - start
    ).dt.days + 1

    work["_days"] = (
        work["_days"]
        .clip(lower=0)
        .fillna(0)
    )

    result = (
        work
        .groupby(
            "DESYNPUF_ID"
        )["_days"]
        .agg(
            ["mean", "max"]
        )
        .reset_index()
    )

    result = result.rename(
        columns={
            "mean":
                f"{prefix}_days_mean",

            "max":
                f"{prefix}_days_max",
        }
    )

    return result


# ============================================================
# MONTHLY FEATURES
# ============================================================

def _monthly_features(
    all_claims,
):

    month_names = {
        1: "jan",
        2: "feb",
        3: "mar",
        4: "apr",
        5: "may",
        6: "jun",
    }

    patient_ids = (
        all_claims[
            ["DESYNPUF_ID"]
        ]
        .drop_duplicates()
        .copy()
    )

    result = patient_ids

    for month, name in (
        month_names.items()
    ):

        month_df = all_claims[
            all_claims[
                "_claim_date"
            ].dt.month == month
        ].copy()

        if month_df.empty:

            result[
                f"{name}_claim_count"
            ] = 0

            result[
                f"{name}_cost"
            ] = 0.0

            continue

        if "CLM_ID" in month_df.columns:

            count_df = (
                month_df
                .groupby(
                    "DESYNPUF_ID"
                )["CLM_ID"]
                .nunique()
                .rename(
                    f"{name}_claim_count"
                )
                .reset_index()
            )

        else:

            count_df = (
                month_df
                .groupby(
                    "DESYNPUF_ID"
                )
                .size()
                .rename(
                    f"{name}_claim_count"
                )
                .reset_index()
            )

        claim_level = (
            _claim_level_cost(
                month_df
            )
        )

        cost_df = (
            claim_level
            .groupby(
                "DESYNPUF_ID"
            )["_claim_cost"]
            .sum()
            .rename(
                f"{name}_cost"
            )
            .reset_index()
        )

        result = result.merge(
            count_df,
            on="DESYNPUF_ID",
            how="left",
        )

        result = result.merge(
            cost_df,
            on="DESYNPUF_ID",
            how="left",
        )

    return result


# ============================================================
# TARGET
# ============================================================

def _target_from_claims(
    inpatient_target,
    outpatient_target,
):

    inpatient_level = (
        _claim_level_cost(
            inpatient_target
        )
    )

    outpatient_level = (
        _claim_level_cost(
            outpatient_target
        )
    )

    inpatient_cost = (
        inpatient_level
        .groupby(
            "DESYNPUF_ID"
        )["_claim_cost"]
        .sum()
        .rename(
            "_target_inpatient"
        )
        .reset_index()
    )

    outpatient_cost = (
        outpatient_level
        .groupby(
            "DESYNPUF_ID"
        )["_claim_cost"]
        .sum()
        .rename(
            "_target_outpatient"
        )
        .reset_index()
    )

    target = inpatient_cost.merge(
        outpatient_cost,
        on="DESYNPUF_ID",
        how="outer",
    )

    target[
        "_target_inpatient"
    ] = target[
        "_target_inpatient"
    ].fillna(0)

    target[
        "_target_outpatient"
    ] = target[
        "_target_outpatient"
    ].fillna(0)

    target[
        TARGET_COLUMN
    ] = (
        target[
            "_target_inpatient"
        ]
        + target[
            "_target_outpatient"
        ]
    )

    return target[
        [
            "DESYNPUF_ID",
            TARGET_COLUMN,
        ]
    ]


# ============================================================
# MAIN DATASET BUILDER
# ============================================================

def load_halfyear_dataset():

    print(
        "=" * 72
    )

    print(
        "CMS HALF-YEAR HEALTHCARE COST FORECASTING"
    )

    print(
        "JAN-JUN 2009 -> JUL-DEC 2009"
    )

    print(
        "=" * 72
    )

    for path in [
        BENEFICIARY_FILE,
        INPATIENT_FILE,
        OUTPATIENT_FILE,
    ]:
        _check_file(path)

    # ========================================================
    # LOAD
    # ========================================================

    print(
        "\nLoading CMS files..."
    )

    bene = pd.read_csv(
        BENEFICIARY_FILE,
        low_memory=False,
    )

    inpatient = pd.read_csv(
        INPATIENT_FILE,
        low_memory=False,
    )

    outpatient = pd.read_csv(
        OUTPATIENT_FILE,
        low_memory=False,
    )

    print(
        "Beneficiaries:",
        f"{len(bene):,}",
    )

    print(
        "Inpatient rows:",
        f"{len(inpatient):,}",
    )

    print(
        "Outpatient rows:",
        f"{len(outpatient):,}",
    )

    # ========================================================
    # PREPARE CLAIMS
    # ========================================================

    inpatient = _prepare_claims(
        inpatient,
        "Inpatient",
    )

    outpatient = _prepare_claims(
        outpatient,
        "Outpatient",
    )

    # ========================================================
    # SPLIT TEMPORALLY
    # ========================================================

    inpatient_features = inpatient[
        (
            inpatient[
                "_claim_date"
            ] >= FEATURE_START
        )
        & (
            inpatient[
                "_claim_date"
            ] <= FEATURE_END
        )
    ].copy()

    outpatient_features = outpatient[
        (
            outpatient[
                "_claim_date"
            ] >= FEATURE_START
        )
        & (
            outpatient[
                "_claim_date"
            ] <= FEATURE_END
        )
    ].copy()

    inpatient_target = inpatient[
        (
            inpatient[
                "_claim_date"
            ] >= TARGET_START
        )
        & (
            inpatient[
                "_claim_date"
            ] <= TARGET_END
        )
    ].copy()

    outpatient_target = outpatient[
        (
            outpatient[
                "_claim_date"
            ] >= TARGET_START
        )
        & (
            outpatient[
                "_claim_date"
            ] <= TARGET_END
        )
    ].copy()

    print(
        "\nFIRST HALF:"
    )

    print(
        "Inpatient rows:",
        f"{len(inpatient_features):,}",
    )

    print(
        "Outpatient rows:",
        f"{len(outpatient_features):,}",
    )

    print(
        "\nSECOND HALF:"
    )

    print(
        "Inpatient rows:",
        f"{len(inpatient_target):,}",
    )

    print(
        "Outpatient rows:",
        f"{len(outpatient_target):,}",
    )

    # ========================================================
    # DEMOGRAPHICS
    # ========================================================

    birth_values = (
        bene[
            "BENE_BIRTH_DT"
        ]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.strip()
    )

    birth_date = pd.to_datetime(
        birth_values,
        format="%Y%m%d",
        errors="coerce",
    )

    reference = pd.Timestamp(
        "2009-06-30"
    )

    bene["age"] = np.floor(
        (
            reference
            - birth_date
        ).dt.days
        / 365.25
    )

    sex_raw = pd.to_numeric(
        bene[
            "BENE_SEX_IDENT_CD"
        ],
        errors="coerce",
    )

    bene["sex"] = (
        sex_raw == 1
    ).astype(int)

    # ========================================================
    # CHRONIC CONDITIONS
    # ========================================================

    for source, feature in (
        CHRONIC_CONDITION_MAP.items()
    ):

        if source in bene.columns:

            bene[
                feature
            ] = _condition_binary(
                bene[source]
            )

        else:

            bene[
                feature
            ] = 0

    bene[
        "chronic_condition_count"
    ] = bene[
        CONDITION_COLUMNS
    ].sum(
        axis=1
    )

    # ========================================================
    # BASE PATIENT TABLE
    # ========================================================

    base_columns = [
        "DESYNPUF_ID",
        "age",
        "sex",
        "chronic_condition_count",
    ] + CONDITION_COLUMNS

    patient_df = bene[
        base_columns
    ].copy()

    # ========================================================
    # CLAIM COUNTS
    # ========================================================

    inpatient_count = _claim_counts(
        inpatient_features,
        "inpatient_claim_count",
    )

    outpatient_count = _claim_counts(
        outpatient_features,
        "outpatient_claim_count",
    )

    # ========================================================
    # CLAIM COSTS
    # ========================================================

    inpatient_level = (
        _claim_level_cost(
            inpatient_features
        )
    )

    outpatient_level = (
        _claim_level_cost(
            outpatient_features
        )
    )

    inpatient_cost = (
        _cost_statistics(
            inpatient_level,
            "inpatient",
        )
    )

    outpatient_cost = (
        _cost_statistics(
            outpatient_level,
            "outpatient",
        )
    )

    # ========================================================
    # DIAGNOSIS FEATURES
    # ========================================================

    inpatient_diag_columns = (
        _diagnosis_columns(
            inpatient_features
        )
    )

    outpatient_diag_columns = (
        _diagnosis_columns(
            outpatient_features
        )
    )

    print(
        "\nDiagnosis columns:"
    )

    print(
        "Inpatient:",
        len(
            inpatient_diag_columns
        ),
    )

    print(
        "Outpatient:",
        len(
            outpatient_diag_columns
        ),
    )

    inpatient_diag = (
        _code_features(
            inpatient_features,
            inpatient_diag_columns,
            "inpatient_diagnosis",
        )
    )

    outpatient_diag = (
        _code_features(
            outpatient_features,
            outpatient_diag_columns,
            "outpatient_diagnosis",
        )
    )

    # ========================================================
    # PROCEDURE FEATURES
    # ========================================================

    inpatient_proc_columns = (
        _procedure_columns(
            inpatient_features
        )
    )

    outpatient_proc_columns = (
        _procedure_columns(
            outpatient_features
        )
    )

    print(
        "\nProcedure columns:"
    )

    print(
        "Inpatient:",
        len(
            inpatient_proc_columns
        ),
    )

    print(
        "Outpatient:",
        len(
            outpatient_proc_columns
        ),
    )

    inpatient_proc = (
        _code_features(
            inpatient_features,
            inpatient_proc_columns,
            "inpatient_procedure",
        )
    )

    outpatient_proc = (
        _code_features(
            outpatient_features,
            outpatient_proc_columns,
            "outpatient_procedure",
        )
    )

    # ========================================================
    # DURATION
    # ========================================================

    inpatient_duration = (
        _duration_features(
            inpatient_features,
            "inpatient",
        )
    )

    outpatient_duration = (
        _duration_features(
            outpatient_features,
            "outpatient",
        )
    )

    # ========================================================
    # MONTHLY FEATURES
    # ========================================================

    # Add a claim-type prefix to CLM_ID so an inpatient claim
    # and outpatient claim cannot accidentally share the same
    # identifier after concatenation.

    inpatient_monthly = (
        inpatient_features.copy()
    )

    outpatient_monthly = (
        outpatient_features.copy()
    )

    if "CLM_ID" in inpatient_monthly.columns:

        inpatient_monthly[
            "CLM_ID"
        ] = (
            "IP_"
            + inpatient_monthly[
                "CLM_ID"
            ].astype(str)
        )

    if "CLM_ID" in outpatient_monthly.columns:

        outpatient_monthly[
            "CLM_ID"
        ] = (
            "OP_"
            + outpatient_monthly[
                "CLM_ID"
            ].astype(str)
        )

    combined_first_half = pd.concat(
        [
            inpatient_monthly,
            outpatient_monthly,
        ],
        ignore_index=True,
        sort=False,
    )

    monthly = _monthly_features(
        combined_first_half
    )

    # ========================================================
    # TARGET
    # ========================================================

    target_df = _target_from_claims(
        inpatient_target,
        outpatient_target,
    )

    # ========================================================
    # MERGE
    # ========================================================

    feature_frames = [
        inpatient_count,
        outpatient_count,
        inpatient_cost,
        outpatient_cost,
        inpatient_diag,
        outpatient_diag,
        inpatient_proc,
        outpatient_proc,
        inpatient_duration,
        outpatient_duration,
        monthly,
    ]

    for frame in feature_frames:

        patient_df = patient_df.merge(
            frame,
            on="DESYNPUF_ID",
            how="left",
        )

    # Keep all 2009 beneficiaries. Patients without a
    # second-half claim receive a valid second-half cost of 0.

    patient_df = patient_df.merge(
        target_df,
        on="DESYNPUF_ID",
        how="left",
    )

    patient_df[
        TARGET_COLUMN
    ] = (
        patient_df[
            TARGET_COLUMN
        ]
        .fillna(0)
    )

    # ========================================================
    # ENSURE RAW FEATURES EXIST
    # ========================================================

    required_raw = [
        "inpatient_claim_count",
        "outpatient_claim_count",

        "inpatient_cost_total",
        "inpatient_cost_mean",
        "inpatient_cost_max",
        "inpatient_cost_std",

        "outpatient_cost_total",
        "outpatient_cost_mean",
        "outpatient_cost_max",
        "outpatient_cost_std",

        "inpatient_diagnosis_count",
        "inpatient_unique_diagnosis_count",

        "outpatient_diagnosis_count",
        "outpatient_unique_diagnosis_count",

        "inpatient_procedure_count",
        "inpatient_unique_procedure_count",

        "outpatient_procedure_count",
        "outpatient_unique_procedure_count",

        "inpatient_days_mean",
        "inpatient_days_max",

        "outpatient_days_mean",
        "outpatient_days_max",

        "jan_claim_count",
        "feb_claim_count",
        "mar_claim_count",
        "apr_claim_count",
        "may_claim_count",
        "jun_claim_count",

        "jan_cost",
        "feb_cost",
        "mar_cost",
        "apr_cost",
        "may_cost",
        "jun_cost",
    ]

    for column in required_raw:

        if column not in patient_df.columns:

            patient_df[
                column
            ] = 0

        patient_df[
            column
        ] = (
            pd.to_numeric(
                patient_df[column],
                errors="coerce",
            )
            .fillna(0)
        )

    # ========================================================
    # DERIVED UTILIZATION
    # ========================================================

    patient_df[
        "total_claim_count"
    ] = (
        patient_df[
            "inpatient_claim_count"
        ]
        + patient_df[
            "outpatient_claim_count"
        ]
    )

    patient_df[
        "first_half_cost"
    ] = (
        patient_df[
            "inpatient_cost_total"
        ]
        + patient_df[
            "outpatient_cost_total"
        ]
    )

    # ========================================================
    # DIAGNOSIS / PROCEDURE TOTALS
    # ========================================================

    patient_df[
        "total_diagnosis_count"
    ] = (
        patient_df[
            "inpatient_diagnosis_count"
        ]
        + patient_df[
            "outpatient_diagnosis_count"
        ]
    )

    patient_df[
        "total_procedure_count"
    ] = (
        patient_df[
            "inpatient_procedure_count"
        ]
        + patient_df[
            "outpatient_procedure_count"
        ]
    )

    # ========================================================
    # COST INTENSITY
    # ========================================================

    patient_df[
        "cost_per_claim"
    ] = _safe_divide(
        patient_df[
            "first_half_cost"
        ],
        patient_df[
            "total_claim_count"
        ],
    )

    patient_df[
        "inpatient_cost_per_claim"
    ] = _safe_divide(
        patient_df[
            "inpatient_cost_total"
        ],
        patient_df[
            "inpatient_claim_count"
        ],
    )

    patient_df[
        "outpatient_cost_per_claim"
    ] = _safe_divide(
        patient_df[
            "outpatient_cost_total"
        ],
        patient_df[
            "outpatient_claim_count"
        ],
    )

    # ========================================================
    # CLAIM RATIOS
    # ========================================================

    patient_df[
        "inpatient_claim_ratio"
    ] = _safe_divide(
        patient_df[
            "inpatient_claim_count"
        ],
        patient_df[
            "total_claim_count"
        ],
    )

    patient_df[
        "outpatient_claim_ratio"
    ] = _safe_divide(
        patient_df[
            "outpatient_claim_count"
        ],
        patient_df[
            "total_claim_count"
        ],
    )

    # ========================================================
    # RECENCY
    # ========================================================

    last_claim = (
        combined_first_half
        .groupby(
            "DESYNPUF_ID"
        )["_claim_date"]
        .max()
    )

    recency = (
        FEATURE_END
        - last_claim
    ).dt.days

    recency_df = (
        recency
        .rename(
            "days_since_last_claim"
        )
        .reset_index()
    )

    patient_df = patient_df.merge(
        recency_df,
        on="DESYNPUF_ID",
        how="left",
    )

    # No first-half claims: use full six-month window.
    patient_df[
        "days_since_last_claim"
    ] = (
        patient_df[
            "days_since_last_claim"
        ]
        .fillna(181)
        .clip(
            lower=0,
            upper=181,
        )
    )

    # ========================================================
    # RECENT TREND
    # ========================================================

    patient_df[
        "early_3m_cost"
    ] = (
        patient_df[
            "jan_cost"
        ]
        + patient_df[
            "feb_cost"
        ]
        + patient_df[
            "mar_cost"
        ]
    )

    patient_df[
        "recent_3m_cost"
    ] = (
        patient_df[
            "apr_cost"
        ]
        + patient_df[
            "may_cost"
        ]
        + patient_df[
            "jun_cost"
        ]
    )

    patient_df[
        "early_3m_claims"
    ] = (
        patient_df[
            "jan_claim_count"
        ]
        + patient_df[
            "feb_claim_count"
        ]
        + patient_df[
            "mar_claim_count"
        ]
    )

    patient_df[
        "recent_3m_claims"
    ] = (
        patient_df[
            "apr_claim_count"
        ]
        + patient_df[
            "may_claim_count"
        ]
        + patient_df[
            "jun_claim_count"
        ]
    )

    patient_df[
        "recent_to_early_cost_ratio"
    ] = _safe_divide(
        patient_df[
            "recent_3m_cost"
        ],
        patient_df[
            "early_3m_cost"
        ],
    )

    # Cap extreme ratios when denominator is tiny.
    patient_df[
        "recent_to_early_cost_ratio"
    ] = (
        patient_df[
            "recent_to_early_cost_ratio"
        ]
        .clip(
            lower=0,
            upper=20,
        )
    )

    # ========================================================
    # INTERACTIONS
    # ========================================================

    patient_df[
        "chronic_cost_interaction"
    ] = (
        patient_df[
            "chronic_condition_count"
        ]
        * np.log1p(
            patient_df[
                "first_half_cost"
            ].clip(lower=0)
        )
    )

    patient_df[
        "chronic_claim_interaction"
    ] = (
        patient_df[
            "chronic_condition_count"
        ]
        * patient_df[
            "total_claim_count"
        ]
    )

    # ========================================================
    # CLEAN
    # ========================================================

    for column in (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    ):

        if column not in patient_df.columns:

            patient_df[column] = 0

        patient_df[column] = (
            pd.to_numeric(
                patient_df[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    # ========================================================
    # AGE RANGE
    # ========================================================

    patient_df = patient_df[
        (
            patient_df[
                "age"
            ] >= 65
        )
        & (
            patient_df[
                "age"
            ] <= 100
        )
    ].copy()

    patient_df[
        "age"
    ] = patient_df[
        "age"
    ].astype(int)

    # ========================================================
    # BINARY
    # ========================================================

    for column in (
        ["sex"]
        + CONDITION_COLUMNS
    ):

        patient_df[column] = (
            patient_df[column]
            .clip(
                lower=0,
                upper=1,
            )
            .astype(int)
        )

    # ========================================================
    # NON-NEGATIVE
    # ========================================================

    for column in (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    ):

        if column != "sex":

            patient_df[column] = (
                patient_df[column]
                .clip(lower=0)
            )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    patient_df = (
        patient_df
        .drop_duplicates(
            subset=[
                "DESYNPUF_ID"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # FINAL ORDER
    # ========================================================

    patient_df = patient_df[
        ["DESYNPUF_ID"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    ].copy()

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "HALF-YEAR DATASET CREATED"
    )

    print(
        "=" * 72
    )

    print(
        "Patients:",
        f"{len(patient_df):,}",
    )

    print(
        "Features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Shape:",
        patient_df.shape,
    )

    print(
        "\nTarget:"
    )

    print(
        patient_df[
            TARGET_COLUMN
        ].describe()
    )

    zero_target_rate = (
        patient_df[
            TARGET_COLUMN
        ]
        .eq(0)
        .mean()
        * 100
    )

    print(
        "\nPatients with $0 "
        "second-half IP/OP claims:",
        f"{zero_target_rate:.2f}%",
    )

    first_half_users = (
        patient_df[
            "total_claim_count"
        ] > 0
    ).mean() * 100

    print(
        "Patients with at least one "
        "first-half claim:",
        f"{first_half_users:.2f}%",
    )

    print(
        "\nMissing values:",
        int(
            patient_df[
                FEATURE_COLUMNS
                + [TARGET_COLUMN]
            ]
            .isna()
            .sum()
            .sum()
        ),
    )

    print(
        "\nTop correlations:"
    )

    correlations = (
        patient_df[
            FEATURE_COLUMNS
            + [TARGET_COLUMN]
        ]
        .corr(
            numeric_only=True
        )[TARGET_COLUMN]
        .drop(
            TARGET_COLUMN
        )
        .abs()
        .sort_values(
            ascending=False
        )
    )

    print(
        correlations
        .head(20)
        .to_string()
    )

    return patient_df


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    df = load_halfyear_dataset()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PREPROCESSING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 72
    )