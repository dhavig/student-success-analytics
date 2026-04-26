"""Tests for the synthetic data generator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etl.generate_data import PROGRAMS


def test_students_shape_and_columns(small_cohort):
    students = small_cohort["students"]
    assert len(students) == 200
    expected = {
        "student_id", "first_name", "last_name", "cohort_year",
        "gender", "race_ethnicity", "first_generation", "pell_eligible",
        "hs_gpa", "sat_total", "program_code",
        "institutional_aid", "federal_aid", "unmet_need",
        "distance_from_home_mi",
    }
    assert expected.issubset(set(students.columns))


def test_student_ids_are_unique(small_cohort):
    students = small_cohort["students"]
    assert students["student_id"].is_unique


def test_demographic_value_ranges(small_cohort):
    students = small_cohort["students"]
    assert students["hs_gpa"].between(2.0, 4.5).all()
    assert students["sat_total"].between(800, 1600).all()
    assert students["unmet_need"].ge(0).all()
    assert students["institutional_aid"].ge(0).all()
    assert set(students["first_generation"].unique()) <= {"Yes", "No"}
    assert set(students["pell_eligible"].unique()) <= {"Yes", "No"}


def test_program_codes_are_valid(small_cohort):
    valid = {p[0] for p in PROGRAMS}
    assert set(small_cohort["students"]["program_code"]).issubset(valid)


def test_terms_two_per_student(small_cohort):
    terms = small_cohort["terms"]
    counts = terms.groupby("student_id").size()
    assert (counts == 2).all()
    assert set(terms["term_season"].unique()) == {"Fall", "Spring"}


def test_term_gpa_in_valid_range(small_cohort):
    terms = small_cohort["terms"]
    assert terms["term_gpa"].between(0.0, 4.0).all()
    assert terms["credits_attempted"].between(12, 18).all()
    assert (terms["credits_earned"] <= terms["credits_attempted"]).all()


def test_retention_outcome_is_binary(small_cohort):
    retention = small_cohort["retention"]
    assert set(retention["returned_year_2"].unique()) <= {0, 1}
    assert retention["retention_probability_true"].between(0, 1).all()


def test_overall_retention_rate_is_realistic(small_cohort):
    rate = small_cohort["retention"]["returned_year_2"].mean()
    # Synthetic generator targets ~80%; allow generous band for sample noise.
    assert 0.65 <= rate <= 0.95, f"Retention rate {rate:.2%} outside realistic band"


def test_generative_signal_is_learnable(small_cohort):
    """First-term GPA should correlate positively with year-2 retention."""
    terms = small_cohort["terms"]
    retention = small_cohort["retention"]
    fall = terms[terms["term_seq"] == 1][["student_id", "term_gpa"]]
    df = fall.merge(retention[["student_id", "returned_year_2"]], on="student_id")
    corr = df["term_gpa"].corr(df["returned_year_2"])
    assert corr > 0.15, f"GPA-retention correlation too weak: {corr:.3f}"


def test_programs_dim_is_complete(small_cohort):
    programs = small_cohort["programs"]
    assert len(programs) == len(PROGRAMS)
    assert set(programs.columns) == {"program_code", "program_name", "college"}
