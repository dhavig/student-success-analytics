"""Shared pytest fixtures."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def project_root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def small_cohort():
    """A reproducible 200-student cohort generated in-memory for fast tests."""
    from etl.generate_data import (GenConfig, compute_retention,
                                   generate_students, generate_term_enrollment,
                                   write_program_dim)

    cfg = GenConfig(n_students=200, cohort_years=(2022, 2023), seed=7)
    students = generate_students(cfg)
    terms = generate_term_enrollment(students, cfg)
    retention = compute_retention(students, terms, cfg)
    programs = write_program_dim()
    return {"students": students, "terms": terms,
            "retention": retention, "programs": programs}
