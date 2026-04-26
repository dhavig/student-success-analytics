"""
Load generated CSVs into a DuckDB star-schema warehouse.

Usage:
    python etl/load_warehouse.py

Reads:  data/students.csv, data/term_enrollment.csv, data/retention.csv,
        data/programs.csv
Writes: data/warehouse.duckdb
"""

from __future__ import annotations

import os
import sys

import duckdb
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SQL_DIR = os.path.join(ROOT, "sql")
DB_PATH = os.path.join(DATA_DIR, "warehouse.duckdb")
SCHEMA_PATH = os.path.join(SQL_DIR, "schema.sql")

REQUIRED_FILES = ["students.csv", "term_enrollment.csv", "retention.csv", "programs.csv"]


def _check_inputs() -> None:
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        sys.exit(
            "Missing input CSVs in ./data: "
            + ", ".join(missing)
            + "\nRun `python etl/generate_data.py` first."
        )


def _term_key(year: int, season: str) -> str:
    return f"{year}-{season}"


def main() -> None:
    _check_inputs()

    students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    terms = pd.read_csv(os.path.join(DATA_DIR, "term_enrollment.csv"))
    retention = pd.read_csv(os.path.join(DATA_DIR, "retention.csv"))
    programs = pd.read_csv(os.path.join(DATA_DIR, "programs.csv"))

    # Names are PII-like; not loaded into the warehouse for analytics.
    student_dim = students.drop(columns=["first_name", "last_name"])

    terms = terms.copy()
    terms["term_key"] = [_term_key(y, s) for y, s in zip(terms["term_year"], terms["term_season"])]
    term_dim = (terms[["term_key", "term_year", "term_season", "term_seq"]]
                .drop_duplicates()
                .reset_index(drop=True))

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)

    with open(SCHEMA_PATH) as fh:
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

    counts = {
        t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ["dim_student", "dim_program", "dim_term",
                  "fact_term_enrollment", "fact_retention"]
    }
    con.close()

    print(f"Built warehouse at {DB_PATH}")
    for t, n in counts.items():
        print(f"  {t:25s} {n:>8,} rows")


if __name__ == "__main__":
    main()
