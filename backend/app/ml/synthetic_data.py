"""
Generates a synthetic Medicare-claims-like dataset that mirrors the structure
described in the project proposal (demographics, chronic conditions, prior
utilization, and reimbursement history) so the pipeline can be trained and
demoed without requiring a manual download of the real CMS files.

Swap `generate_dataset()` for a loader that reads the real CMS Synthetic
Public Use Files (see the DataSet link in the proposal) once you have them
on disk -- the column names below are the contract the rest of the app
expects, and are intentionally aligned with the CMS DE-SynPUF fields.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "sex",  # 0 = female, 1 = male
    "chronic_condition_count",  # count out of 11 CMS chronic condition flags
    "prior_inpatient_visits",
    "prior_outpatient_visits",
    "prior_er_visits",
    "prior_rx_count",  # distinct prescription drug events, prior year
    "prior_reimbursement",  # prior year total reimbursement, USD
]

TARGET_COLUMN = "next_year_cost"


def generate_dataset(n_patients: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(65, 95, size=n_patients)
    sex = rng.integers(0, 2, size=n_patients)
    chronic = rng.poisson(2.1, size=n_patients).clip(0, 11)
    inpatient = rng.poisson(0.35 + chronic * 0.12, size=n_patients).clip(0, 12)
    outpatient = rng.poisson(3.0 + chronic * 0.9, size=n_patients).clip(0, 60)
    er_visits = rng.poisson(0.25 + chronic * 0.08, size=n_patients).clip(0, 10)
    rx_count = rng.poisson(4.0 + chronic * 1.6, size=n_patients).clip(0, 45)
    prior_reimbursement = (
        800
        + chronic * 950
        + inpatient * 4200
        + outpatient * 180
        + er_visits * 650
        + rx_count * 95
        + rng.normal(0, 900, size=n_patients)
    ).clip(0, None)

    # Non-linear next-year cost: base + chronic burden + utilization momentum
    # + a heavy tail for a small number of catastrophic / high-cost patients.
    base_cost = (
        650
        + (age - 65) * 38
        + chronic**1.6 * 480
        + inpatient**1.4 * 2600
        + outpatient * 165
        + er_visits**1.3 * 520
        + rx_count * 110
        + prior_reimbursement * 0.42
    )

    noise = rng.normal(0, 420, size=n_patients)
    catastrophic_mask = rng.random(n_patients) < 0.02
    catastrophic_boost = catastrophic_mask * rng.uniform(8000, 25000, size=n_patients)

    next_year_cost = (base_cost + noise + catastrophic_boost).clip(200, None)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "chronic_condition_count": chronic,
            "prior_inpatient_visits": inpatient,
            "prior_outpatient_visits": outpatient,
            "prior_er_visits": er_visits,
            "prior_rx_count": rx_count,
            "prior_reimbursement": prior_reimbursement.round(2),
            TARGET_COLUMN: next_year_cost.round(2),
        }
    )
    return df


if __name__ == "__main__":
    data = generate_dataset()
    print(data.describe())
