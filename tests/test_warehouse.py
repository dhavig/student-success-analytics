"""Tests for the DuckDB warehouse loader."""
from __future__ import annotations

import os

import duckdb
import pandas as pd
import pytest


SCHEMA_FILE = "sql/schema.sql"


def _build_warehouse(tmp_path, project_root, cohort) -> str:
    """Build a warehouse in tmp_path from an in-memory cohort."""
    db_path = str(tmp_path / "warehouse.duckdb")

    students = cohort["students"]
    terms = cohort["terms"].copy()
    retention = cohort["retention"]
    programs = cohort["programs"]

    student_dim = students.drop(columns=["first_name", "last_name"])
    terms["term_key"] = terms["term_year"].astype(str) + "-" + terms["term_season"]
    term_dim = (terms[["term_key", "term_year", "term_season", "term_seq"]]
                .drop_duplicates().reset_index(drop=True))

    con = duckdb.connect(db_path)
    with open(os.path.join(project_root, SCHEMA_FILE)) as fh:
        con.execute(fh.read())

    con.register("student_dim_df", student_dim)
    con.register("program_df", programs)
    con.register("term_dim_df", term_dim)
    con.register("term_fact_df", terms[[
        "student_id", "term_key", "term_year", "term_season", "term_seq",
        "term_gpa", "credits_attempted", "credits_earned",
        "lms_logins_per_week", "advising_meetings", "on_campus_housing",
    ]])
    con.register("retention_df", retention)

    con.execute("INSERT INTO dim_student        SELECT * FROM student_dim_df;")
    con.execute("INSERT INTO dim_program        SELECT * FROM program_df;")
    con.execute("INSERT INTO dim_term           SELECT * FROM term_dim_df;")
    con.execute("INSERT INTO fact_term_enrollment SELECT * FROM term_fact_df;")
    con.execute("INSERT INTO fact_retention     SELECT * FROM retention_df;")
    con.close()
    return db_path


@pytest.fixture(scope="module")
def warehouse_db(tmp_path_factory, project_root, small_cohort):
    tmp_path = tmp_path_factory.mktemp("wh")
    return _build_warehouse(tmp_path, project_root, small_cohort)


def test_all_expected_tables_exist(warehouse_db):
    con = duckdb.connect(warehouse_db, read_only=True)
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    con.close()
    assert {"dim_student", "dim_program", "dim_term",
            "fact_term_enrollment", "fact_retention"}.issubset(tables)


def test_row_counts_match_inputs(warehouse_db, small_cohort):
    con = duckdb.connect(warehouse_db, read_only=True)
    n_students = con.execute("SELECT COUNT(*) FROM dim_student").fetchone()[0]
    n_terms = con.execute("SELECT COUNT(*) FROM fact_term_enrollment").fetchone()[0]
    n_retention = con.execute("SELECT COUNT(*) FROM fact_retention").fetchone()[0]
    con.close()

    assert n_students == len(small_cohort["students"])
    assert n_terms == len(small_cohort["terms"])
    assert n_retention == len(small_cohort["retention"])


def test_referential_integrity_student_to_term(warehouse_db):
    """Every term row must reference an existing student."""
    con = duckdb.connect(warehouse_db, read_only=True)
    orphans = con.execute("""
        SELECT COUNT(*) FROM fact_term_enrollment t
        LEFT JOIN dim_student s USING (student_id)
        WHERE s.student_id IS NULL
    """).fetchone()[0]
    con.close()
    assert orphans == 0


def test_referential_integrity_program(warehouse_db):
    con = duckdb.connect(warehouse_db, read_only=True)
    orphans = con.execute("""
        SELECT COUNT(*) FROM dim_student s
        LEFT JOIN dim_program p USING (program_code)
        WHERE p.program_code IS NULL
    """).fetchone()[0]
    con.close()
    assert orphans == 0


def test_no_pii_in_warehouse(warehouse_db):
    """Names must not be loaded into the analytic warehouse."""
    con = duckdb.connect(warehouse_db, read_only=True)
    cols = {row[1] for row in con.execute("PRAGMA table_info('dim_student')").fetchall()}
    con.close()
    assert "first_name" not in cols
    assert "last_name" not in cols


def test_analytics_query_runs(warehouse_db, project_root):
    """The first reference query in sql/analytics_queries.sql should execute."""
    con = duckdb.connect(warehouse_db, read_only=True)
    df = con.execute("""
        SELECT cohort_year, COUNT(*) AS n,
               AVG(returned_year_2) AS retention_rate
        FROM fact_retention
        GROUP BY cohort_year
        ORDER BY cohort_year
    """).df()
    con.close()
    assert len(df) > 0
    assert df["retention_rate"].between(0, 1).all()
