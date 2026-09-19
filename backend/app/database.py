"""
SQLite is used here as a local stand-in for the Cloud SQL
(PostgreSQL) instance described in the project architecture.

Because the schema is defined through SQLAlchemy, the application
can later be connected to PostgreSQL / Google Cloud SQL by changing
the DATABASE_URL environment variable.
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./healthcare_cost.db",
)

connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# ============================================================
# PATIENT TABLE
# ============================================================

class Patient(Base):
    __tablename__ = "patients"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    # "manual" for individual predictions.
    # For bulk uploads, this can contain the job ID.
    source = Column(
        String,
        default="manual",
    )

    # --------------------------------------------------------
    # MODEL INPUT FEATURES
    # --------------------------------------------------------

    age = Column(
        Integer
    )

    sex = Column(
        Integer
    )

    chronic_condition_count = Column(
        Integer
    )

    prior_inpatient_visits = Column(
        Integer
    )

    prior_outpatient_visits = Column(
        Integer
    )

    prior_reimbursement = Column(
        Float
    )

    # --------------------------------------------------------
    # MODEL OUTPUT
    # --------------------------------------------------------

    predicted_cost = Column(
        Float
    )

    risk_level = Column(
        String
    )

    percentile = Column(
        Integer
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


# ============================================================
# BULK PROCESSING JOB TABLE
# ============================================================

class Job(Base):
    __tablename__ = "jobs"

    id = Column(
        String,
        primary_key=True,
        index=True,
    )

    status = Column(
        String,
        default="queued",
    )

    filename = Column(
        String,
        nullable=True,
    )

    total_records = Column(
        Integer,
        default=0,
    )

    processed_records = Column(
        Integer,
        default=0,
    )

    high_risk_found = Column(
        Integer,
        default=0,
    )

    error_message = Column(
        String,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    completed_at = Column(
        DateTime,
        nullable=True,
    )


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

def init_db():
    Base.metadata.create_all(
        bind=engine
    )


# ============================================================
# DATABASE SESSION
# ============================================================

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()