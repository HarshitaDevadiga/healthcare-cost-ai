from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
CMS_DIR = DATA_DIR / "cms"


# ============================================================
# CMS FILES
# ============================================================

BENEFICIARY_2008_FILE = (
    CMS_DIR
    / "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"
)

BENEFICIARY_2009_FILE = (
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


CONDITION_FEATURES = list(
    CHRONIC_CONDITION_MAP.values()
)


# ============================================================
# FINAL MODEL FEATURES
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
    # BENEFICIARY REIMBURSEMENT HISTORY
    # --------------------------------------------------------

    "prior_inpatient_reimbursement",
    "prior_outpatient_reimbursement",
    "prior_carrier_reimbursement",
    "prior_reimbursement",

    # --------------------------------------------------------
    # CLAIM UTILIZATION
    # --------------------------------------------------------

    "prior_inpatient_visits",
    "prior_outpatient_visits",
    "total_prior_visits",

    # --------------------------------------------------------
    # INPATIENT CLAIM COST FEATURES
    # --------------------------------------------------------

    "inpatient_claim_cost_total",
    "inpatient_claim_cost_mean",
    "inpatient_claim_cost_max",
    "inpatient_claim_cost_std",

    # --------------------------------------------------------
    # OUTPATIENT CLAIM COST FEATURES
    # --------------------------------------------------------

    "outpatient_claim_cost_total",
    "outpatient_claim_cost_mean",
    "outpatient_claim_cost_max",
    "outpatient_claim_cost_std",

    # --------------------------------------------------------
    # DIAGNOSIS FEATURES
    # --------------------------------------------------------

    "inpatient_diagnosis_count",
    "inpatient_unique_diagnosis_count",

    "outpatient_diagnosis_count",
    "outpatient_unique_diagnosis_count",

    "total_diagnosis_count",
    "total_unique_diagnosis_count",

    # --------------------------------------------------------
    # PROCEDURE FEATURES
    # --------------------------------------------------------

    "inpatient_procedure_count",
    "inpatient_unique_procedure_count",

    "outpatient_procedure_count",
    "outpatient_unique_procedure_count",

    "total_procedure_count",
    "total_unique_procedure_count",

    # --------------------------------------------------------
    # CLAIM DURATION
    # --------------------------------------------------------

    "inpatient_days_mean",
    "inpatient_days_max",
    "outpatient_days_mean",
    "outpatient_days_max",

    # --------------------------------------------------------
    # ENGINEERED UTILIZATION/COST FEATURES
    # --------------------------------------------------------

    "reimbursement_per_visit",

    "inpatient_reimbursement_per_visit",
    "outpatient_reimbursement_per_visit",

    "inpatient_visit_ratio",
    "outpatient_visit_ratio",

    "chronic_cost_interaction",
    "chronic_visit_interaction",
]


TARGET_COLUMN = "next_year_cost"


# ============================================================
# HELPERS
# ============================================================

def _check_file(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"CMS file was not found:\n{path}"
        )


def _numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


def _claim_year(df):

    possible_columns = [
        "CLM_FROM_DT",
        "CLM_THRU_DT",
    ]

    for column in possible_columns:

        if column in df.columns:

            values = (
                df[column]
                .astype(str)
                .str.replace(
                    ".0",
                    "",
                    regex=False,
                )
                .str.strip()
            )

            dates = pd.to_datetime(
                values,
                format="%Y%m%d",
                errors="coerce",
            )

            return dates.dt.year

    raise ValueError(
        "Could not determine claim year."
    )


def _parse_claim_date(series):

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


def _condition_to_binary(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return (
        numeric == 1
    ).astype(int)


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


# ============================================================
# FIND DIAGNOSIS COLUMNS
# ============================================================

def _diagnosis_columns(df):

    columns = []

    for column in df.columns:

        upper = column.upper()

        if (
            "ICD9_DGNS" in upper
            or "DGNS_CD" in upper
        ):
            columns.append(column)

    return columns


# ============================================================
# FIND PROCEDURE COLUMNS
# ============================================================

def _procedure_columns(df):

    columns = []

    for column in df.columns:

        upper = column.upper()

        if (
            "ICD9_PRCDR" in upper
            or "PRCDR_CD" in upper
            or "HCPCS_CD" in upper
        ):
            columns.append(column)

    return columns


# ============================================================
# COUNT CODE INFORMATION PER PATIENT
# ============================================================

def _code_features(
    df,
    columns,
    prefix,
):

    if not columns:

        return pd.DataFrame(
            columns=[
                "DESYNPUF_ID",
                f"{prefix}_count",
                f"{prefix}_unique_count",
            ]
        )

    work = df[
        ["DESYNPUF_ID"] + columns
    ].copy()

    long_df = work.melt(
        id_vars="DESYNPUF_ID",
        value_vars=columns,
        value_name="code",
    )

    long_df["code"] = (
        long_df["code"]
        .astype(str)
        .str.strip()
    )

    invalid_values = {
        "",
        "nan",
        "None",
        "0",
        "0.0",
    }

    long_df = long_df[
        ~long_df["code"].isin(
            invalid_values
        )
    ]

    if long_df.empty:

        return pd.DataFrame(
            columns=[
                "DESYNPUF_ID",
                f"{prefix}_count",
                f"{prefix}_unique_count",
            ]
        )

    total = (
        long_df
        .groupby("DESYNPUF_ID")
        .size()
        .rename(
            f"{prefix}_count"
        )
    )

    unique = (
        long_df
        .groupby("DESYNPUF_ID")[
            "code"
        ]
        .nunique()
        .rename(
            f"{prefix}_unique_count"
        )
    )

    result = pd.concat(
        [
            total,
            unique,
        ],
        axis=1,
    ).reset_index()

    return result


# ============================================================
# CLAIM COUNT
# ============================================================

def _claim_count(
    df,
    output_column,
):

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
# CLAIM COST STATISTICS
# ============================================================

def _claim_cost_features(
    df,
    prefix,
):

    possible_cost_columns = [
        "CLM_PMT_AMT",
        "NCH_PRMRY_PYR_CLM_PD_AMT",
    ]

    cost_column = None

    for column in possible_cost_columns:

        if column in df.columns:
            cost_column = column
            break

    output_columns = [
        "DESYNPUF_ID",
        f"{prefix}_claim_cost_total",
        f"{prefix}_claim_cost_mean",
        f"{prefix}_claim_cost_max",
        f"{prefix}_claim_cost_std",
    ]

    if cost_column is None:

        return pd.DataFrame(
            columns=output_columns
        )

    work = df[
        [
            "DESYNPUF_ID",
            cost_column,
        ]
    ].copy()

    work["_claim_cost"] = _numeric(
        work[cost_column]
    )

    # --------------------------------------------------------
    # If multiple rows correspond to the same claim,
    # aggregate to claim level first.
    # --------------------------------------------------------

    if "CLM_ID" in df.columns:

        work["CLM_ID"] = df[
            "CLM_ID"
        ]

        claim_level = (
            work
            .groupby(
                [
                    "DESYNPUF_ID",
                    "CLM_ID",
                ],
                as_index=False,
            )["_claim_cost"]
            .max()
        )

    else:

        claim_level = work[
            [
                "DESYNPUF_ID",
                "_claim_cost",
            ]
        ].copy()

    result = (
        claim_level
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
                f"{prefix}_claim_cost_total",

            "mean":
                f"{prefix}_claim_cost_mean",

            "max":
                f"{prefix}_claim_cost_max",

            "std":
                f"{prefix}_claim_cost_std",
        }
    )

    result[
        f"{prefix}_claim_cost_std"
    ] = result[
        f"{prefix}_claim_cost_std"
    ].fillna(0)

    return result


# ============================================================
# CLAIM DURATION FEATURES
# ============================================================

def _duration_features(
    df,
    prefix,
):

    mean_column = (
        f"{prefix}_days_mean"
    )

    max_column = (
        f"{prefix}_days_max"
    )

    if (
        "CLM_FROM_DT" not in df.columns
        or "CLM_THRU_DT" not in df.columns
    ):

        return pd.DataFrame(
            columns=[
                "DESYNPUF_ID",
                mean_column,
                max_column,
            ]
        )

    work = df[
        [
            "DESYNPUF_ID",
            "CLM_FROM_DT",
            "CLM_THRU_DT",
        ]
    ].copy()

    start = _parse_claim_date(
        work["CLM_FROM_DT"]
    )

    end = _parse_claim_date(
        work["CLM_THRU_DT"]
    )

    work["_duration"] = (
        end - start
    ).dt.days

    # Include the starting day.
    work["_duration"] = (
        work["_duration"] + 1
    )

    work["_duration"] = (
        work["_duration"]
        .clip(lower=0)
        .fillna(0)
    )

    result = (
        work
        .groupby(
            "DESYNPUF_ID"
        )["_duration"]
        .agg(
            ["mean", "max"]
        )
        .reset_index()
    )

    result = result.rename(
        columns={
            "mean": mean_column,
            "max": max_column,
        }
    )

    return result


# ============================================================
# LOAD CMS DATASET
# ============================================================

def load_cms_dataset():

    print("=" * 70)
    print("LOADING CMS DE-SynPUF DATA")
    print("=" * 70)

    # ========================================================
    # VERIFY FILES
    # ========================================================

    required_files = [
        BENEFICIARY_2008_FILE,
        BENEFICIARY_2009_FILE,
        INPATIENT_FILE,
        OUTPATIENT_FILE,
    ]

    for path in required_files:
        _check_file(path)

    # ========================================================
    # LOAD FILES
    # ========================================================

    print(
        "\nLoading beneficiary files..."
    )

    bene_2008 = pd.read_csv(
        BENEFICIARY_2008_FILE,
        low_memory=False,
    )

    bene_2009 = pd.read_csv(
        BENEFICIARY_2009_FILE,
        low_memory=False,
    )

    print(
        "Loading inpatient claims..."
    )

    inpatient = pd.read_csv(
        INPATIENT_FILE,
        low_memory=False,
    )

    print(
        "Loading outpatient claims..."
    )

    outpatient = pd.read_csv(
        OUTPATIENT_FILE,
        low_memory=False,
    )

    print(
        "\nRaw files loaded."
    )

    print(
        "2008 beneficiaries:",
        f"{len(bene_2008):,}",
    )

    print(
        "2009 beneficiaries:",
        f"{len(bene_2009):,}",
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
    # AGE
    # ========================================================

    birth_values = (
        bene_2008[
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

    reference_date = pd.Timestamp(
        "2008-12-31"
    )

    age_years = (
        (
            reference_date
            - birth_date
        ).dt.days
        / 365.25
    )

    bene_2008["age"] = np.floor(
        pd.to_numeric(
            age_years,
            errors="coerce",
        )
    )

    # ========================================================
    # SEX
    # ========================================================

    sex_raw = pd.to_numeric(
        bene_2008[
            "BENE_SEX_IDENT_CD"
        ],
        errors="coerce",
    )

    bene_2008["sex"] = (
        sex_raw == 1
    ).astype(int)

    # ========================================================
    # CHRONIC CONDITIONS
    # ========================================================

    print(
        "\nCreating chronic-condition features..."
    )

    for source, feature in (
        CHRONIC_CONDITION_MAP.items()
    ):

        if source in bene_2008.columns:

            bene_2008[
                feature
            ] = _condition_to_binary(
                bene_2008[source]
            )

        else:

            bene_2008[
                feature
            ] = 0

    bene_2008[
        "chronic_condition_count"
    ] = bene_2008[
        CONDITION_FEATURES
    ].sum(axis=1)

    # ========================================================
    # 2008 REIMBURSEMENT
    # ========================================================

    reimbursement_mapping = {
        "MEDREIMB_IP":
            "prior_inpatient_reimbursement",

        "MEDREIMB_OP":
            "prior_outpatient_reimbursement",

        "MEDREIMB_CAR":
            "prior_carrier_reimbursement",
    }

    for source, feature in (
        reimbursement_mapping.items()
    ):

        if source in bene_2008.columns:

            bene_2008[
                feature
            ] = _numeric(
                bene_2008[source]
            )

        else:

            bene_2008[
                feature
            ] = 0.0

    bene_2008[
        "prior_reimbursement"
    ] = (
        bene_2008[
            "prior_inpatient_reimbursement"
        ]
        + bene_2008[
            "prior_outpatient_reimbursement"
        ]
        + bene_2008[
            "prior_carrier_reimbursement"
        ]
    )

    # ========================================================
    # 2009 TARGET
    # ========================================================

    print(
        "Creating 2009 target..."
    )

    bene_2009[
        TARGET_COLUMN
    ] = 0.0

    target_found = 0

    for column in [
        "MEDREIMB_IP",
        "MEDREIMB_OP",
        "MEDREIMB_CAR",
    ]:

        if column in bene_2009.columns:

            bene_2009[
                TARGET_COLUMN
            ] += _numeric(
                bene_2009[column]
            )

            target_found += 1

    if target_found == 0:

        raise ValueError(
            "No 2009 reimbursement "
            "columns were found."
        )

    target_df = bene_2009[
        [
            "DESYNPUF_ID",
            TARGET_COLUMN,
        ]
    ].copy()

    # ========================================================
    # FILTER CLAIMS TO 2008 ONLY
    # ========================================================

    inpatient = inpatient.copy()
    outpatient = outpatient.copy()

    inpatient[
        "_claim_year"
    ] = _claim_year(
        inpatient
    )

    outpatient[
        "_claim_year"
    ] = _claim_year(
        outpatient
    )

    inpatient_2008 = inpatient[
        inpatient[
            "_claim_year"
        ] == 2008
    ].copy()

    outpatient_2008 = outpatient[
        outpatient[
            "_claim_year"
        ] == 2008
    ].copy()

    print(
        "\n2008 inpatient rows:",
        f"{len(inpatient_2008):,}",
    )

    print(
        "2008 outpatient rows:",
        f"{len(outpatient_2008):,}",
    )

    # ========================================================
    # UNIQUE CLAIM COUNTS
    # ========================================================

    print(
        "\nCreating unique claim counts..."
    )

    inpatient_visits = _claim_count(
        inpatient_2008,
        "prior_inpatient_visits",
    )

    outpatient_visits = _claim_count(
        outpatient_2008,
        "prior_outpatient_visits",
    )

    # ========================================================
    # CLAIM COST STATISTICS
    # ========================================================

    print(
        "Creating claim cost statistics..."
    )

    inpatient_cost = (
        _claim_cost_features(
            inpatient_2008,
            "inpatient",
        )
    )

    outpatient_cost = (
        _claim_cost_features(
            outpatient_2008,
            "outpatient",
        )
    )

    # ========================================================
    # DIAGNOSIS FEATURES
    # ========================================================

    print(
        "Creating diagnosis features..."
    )

    inpatient_diagnosis_columns = (
        _diagnosis_columns(
            inpatient_2008
        )
    )

    outpatient_diagnosis_columns = (
        _diagnosis_columns(
            outpatient_2008
        )
    )

    print(
        "Inpatient diagnosis columns:",
        len(
            inpatient_diagnosis_columns
        ),
    )

    print(
        "Outpatient diagnosis columns:",
        len(
            outpatient_diagnosis_columns
        ),
    )

    inpatient_diagnosis = (
        _code_features(
            inpatient_2008,
            inpatient_diagnosis_columns,
            "inpatient_diagnosis",
        )
    )

    outpatient_diagnosis = (
        _code_features(
            outpatient_2008,
            outpatient_diagnosis_columns,
            "outpatient_diagnosis",
        )
    )

    # ========================================================
    # PROCEDURE FEATURES
    # ========================================================

    print(
        "Creating procedure features..."
    )

    inpatient_procedure_columns = (
        _procedure_columns(
            inpatient_2008
        )
    )

    outpatient_procedure_columns = (
        _procedure_columns(
            outpatient_2008
        )
    )

    print(
        "Inpatient procedure columns:",
        len(
            inpatient_procedure_columns
        ),
    )

    print(
        "Outpatient procedure columns:",
        len(
            outpatient_procedure_columns
        ),
    )

    inpatient_procedure = (
        _code_features(
            inpatient_2008,
            inpatient_procedure_columns,
            "inpatient_procedure",
        )
    )

    outpatient_procedure = (
        _code_features(
            outpatient_2008,
            outpatient_procedure_columns,
            "outpatient_procedure",
        )
    )

    # ========================================================
    # CLAIM DURATION
    # ========================================================

    print(
        "Creating claim duration features..."
    )

    inpatient_duration = (
        _duration_features(
            inpatient_2008,
            "inpatient",
        )
    )

    outpatient_duration = (
        _duration_features(
            outpatient_2008,
            "outpatient",
        )
    )

    # ========================================================
    # PATIENT-LEVEL BASE
    # ========================================================

    base_columns = [
        "DESYNPUF_ID",
        "age",
        "sex",
        "chronic_condition_count",
    ]

    base_columns += (
        CONDITION_FEATURES
    )

    base_columns += [
        "prior_inpatient_reimbursement",
        "prior_outpatient_reimbursement",
        "prior_carrier_reimbursement",
        "prior_reimbursement",
    ]

    patient_df = bene_2008[
        base_columns
    ].copy()

    # ========================================================
    # MERGE TARGET
    # ========================================================

    patient_df = patient_df.merge(
        target_df,
        on="DESYNPUF_ID",
        how="inner",
    )

    # ========================================================
    # MERGE ALL CLAIM FEATURES
    # ========================================================

    feature_frames = [
        inpatient_visits,
        outpatient_visits,
        inpatient_cost,
        outpatient_cost,
        inpatient_diagnosis,
        outpatient_diagnosis,
        inpatient_procedure,
        outpatient_procedure,
        inpatient_duration,
        outpatient_duration,
    ]

    for feature_df in feature_frames:

        patient_df = (
            patient_df.merge(
                feature_df,
                on="DESYNPUF_ID",
                how="left",
            )
        )

    # ========================================================
    # ENSURE CLAIM FEATURES EXIST
    # ========================================================

    claim_feature_columns = [
        "prior_inpatient_visits",
        "prior_outpatient_visits",

        "inpatient_claim_cost_total",
        "inpatient_claim_cost_mean",
        "inpatient_claim_cost_max",
        "inpatient_claim_cost_std",

        "outpatient_claim_cost_total",
        "outpatient_claim_cost_mean",
        "outpatient_claim_cost_max",
        "outpatient_claim_cost_std",

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
    ]

    for column in claim_feature_columns:

        if column not in patient_df.columns:
            patient_df[column] = 0

        patient_df[column] = (
            pd.to_numeric(
                patient_df[column],
                errors="coerce",
            )
            .fillna(0)
        )

    # ========================================================
    # TOTAL UTILIZATION
    # ========================================================

    patient_df[
        "total_prior_visits"
    ] = (
        patient_df[
            "prior_inpatient_visits"
        ]
        + patient_df[
            "prior_outpatient_visits"
        ]
    )

    # ========================================================
    # TOTAL DIAGNOSIS FEATURES
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
        "total_unique_diagnosis_count"
    ] = (
        patient_df[
            "inpatient_unique_diagnosis_count"
        ]
        + patient_df[
            "outpatient_unique_diagnosis_count"
        ]
    )

    # ========================================================
    # TOTAL PROCEDURE FEATURES
    # ========================================================

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

    patient_df[
        "total_unique_procedure_count"
    ] = (
        patient_df[
            "inpatient_unique_procedure_count"
        ]
        + patient_df[
            "outpatient_unique_procedure_count"
        ]
    )

    # ========================================================
    # COST PER VISIT
    # ========================================================

    patient_df[
        "reimbursement_per_visit"
    ] = _safe_divide(
        patient_df[
            "prior_reimbursement"
        ],
        patient_df[
            "total_prior_visits"
        ],
    )

    patient_df[
        "inpatient_reimbursement_per_visit"
    ] = _safe_divide(
        patient_df[
            "prior_inpatient_reimbursement"
        ],
        patient_df[
            "prior_inpatient_visits"
        ],
    )

    patient_df[
        "outpatient_reimbursement_per_visit"
    ] = _safe_divide(
        patient_df[
            "prior_outpatient_reimbursement"
        ],
        patient_df[
            "prior_outpatient_visits"
        ],
    )

    # ========================================================
    # UTILIZATION RATIOS
    # ========================================================

    patient_df[
        "inpatient_visit_ratio"
    ] = _safe_divide(
        patient_df[
            "prior_inpatient_visits"
        ],
        patient_df[
            "total_prior_visits"
        ],
    )

    patient_df[
        "outpatient_visit_ratio"
    ] = _safe_divide(
        patient_df[
            "prior_outpatient_visits"
        ],
        patient_df[
            "total_prior_visits"
        ],
    )

    # ========================================================
    # INTERACTION FEATURES
    # ========================================================

    patient_df[
        "chronic_cost_interaction"
    ] = (
        patient_df[
            "chronic_condition_count"
        ]
        * np.log1p(
            patient_df[
                "prior_reimbursement"
            ].clip(lower=0)
        )
    )

    patient_df[
        "chronic_visit_interaction"
    ] = (
        patient_df[
            "chronic_condition_count"
        ]
        * patient_df[
            "total_prior_visits"
        ]
    )

    # ========================================================
    # NUMERIC CLEANING
    # ========================================================

    print(
        "\nCleaning final features..."
    )

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
        )

    # ========================================================
    # REMOVE INVALID AGE/TARGET
    # ========================================================

    patient_df = (
        patient_df.dropna(
            subset=[
                "age",
                TARGET_COLUMN,
            ]
        )
    )

    # ========================================================
    # MEDICARE AGE RANGE
    # ========================================================

    patient_df = patient_df[
        (
            patient_df["age"] >= 65
        )
        & (
            patient_df["age"] <= 100
        )
    ].copy()

    patient_df["age"] = (
        patient_df["age"]
        .astype(int)
    )

    # ========================================================
    # FILL MISSING FEATURES
    # ========================================================

    for column in FEATURE_COLUMNS:

        patient_df[column] = (
            patient_df[column]
            .fillna(0)
        )

    # ========================================================
    # NON-NEGATIVE COST / COUNT FEATURES
    # ========================================================

    nonnegative_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in (
            ["sex"]
            + CONDITION_FEATURES
        )
    ]

    for column in (
        nonnegative_columns
        + [TARGET_COLUMN]
    ):

        patient_df[column] = (
            patient_df[column]
            .clip(lower=0)
        )

    # ========================================================
    # BINARY FEATURES
    # ========================================================

    binary_columns = [
        "sex",
    ] + CONDITION_FEATURES

    for column in binary_columns:

        patient_df[column] = (
            patient_df[column]
            .fillna(0)
            .clip(
                lower=0,
                upper=1,
            )
            .astype(int)
        )

    # ========================================================
    # COUNT FEATURES
    # ========================================================

    count_columns = [
        "chronic_condition_count",
        "prior_inpatient_visits",
        "prior_outpatient_visits",
        "total_prior_visits",

        "inpatient_diagnosis_count",
        "inpatient_unique_diagnosis_count",

        "outpatient_diagnosis_count",
        "outpatient_unique_diagnosis_count",

        "total_diagnosis_count",
        "total_unique_diagnosis_count",

        "inpatient_procedure_count",
        "inpatient_unique_procedure_count",

        "outpatient_procedure_count",
        "outpatient_unique_procedure_count",

        "total_procedure_count",
        "total_unique_procedure_count",
    ]

    for column in count_columns:

        patient_df[column] = (
            patient_df[column]
            .round()
            .astype(int)
        )

    # ========================================================
    # REMOVE DUPLICATE PATIENTS
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
    # FINAL COLUMN ORDER
    # ========================================================

    final_columns = (
        ["DESYNPUF_ID"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    patient_df = (
        patient_df[
            final_columns
        ].copy()
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RICH CMS MACHINE LEARNING DATASET CREATED"
    )

    print(
        "=" * 70
    )

    print(
        "\nPatients:",
        f"{len(patient_df):,}",
    )

    print(
        "Features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Target:",
        TARGET_COLUMN,
    )

    print(
        "\nFeature list:"
    )

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):

        print(
            f"{index:02d}. {feature}"
        )

    print(
        "\nTarget statistics:"
    )

    print(
        patient_df[
            TARGET_COLUMN
        ].describe()
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
        "\nFirst five rows:"
    )

    print(
        patient_df.head()
        .to_string(
            index=False
        )
    )

    return patient_df


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    dataset = load_cms_dataset()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL DATASET"
    )

    print(
        "=" * 70
    )

    print(
        "Shape:",
        dataset.shape,
    )

    print(
        "Patients:",
        f"{len(dataset):,}",
    )

    print(
        "ML features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "\nCMS preprocessing "
        "completed successfully."
    )