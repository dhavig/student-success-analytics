"""
Build a denormalized, Tableau-ready CSV from the warehouse + model + NLP
artifacts. One row per student, with every dimension a Tableau viz needs to
filter or color by, plus the model's risk score and the NLP sentiment
aggregates.

Usage:
    python scripts/export_for_tableau.py

Outputs:
    data/tableau_student_view.csv
    data/tableau_term_view.csv     -- one row per (student, term)

Then in Tableau Public / Desktop:
    Connect to Text File -> tableau_student_view.csv
    Build the four worksheets described in docs/TABLEAU_GUIDE.md.
"""

from __future__ import annotations

import os

import duckdb
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "warehouse.duckdb")
ART_DIR = os.path.join(ROOT, "artifacts")
DATA_DIR = os.path.join(ROOT, "data")


def main() -> None:
    if not os.path.exists(DB_PATH):
        raise SystemExit("Run the warehouse first: python etl/load_warehouse.py")

    con = duckdb.connect(DB_PATH, read_only=True)

    student_view = con.execute("""
        WITH first_year AS (
            SELECT
                student_id,
                AVG(term_gpa)            AS mean_term_gpa,
                SUM(credits_earned)      AS total_credits_earned,
                AVG(lms_logins_per_week) AS mean_lms,
                SUM(advising_meetings)   AS total_advising,
                MAX(on_campus_housing)   AS ever_on_campus
            FROM fact_term_enrollment
            GROUP BY student_id
        )
        SELECT
            s.student_id,
            s.cohort_year,
            s.gender,
            s.race_ethnicity,
            s.first_generation,
            s.pell_eligible,
            s.hs_gpa,
            s.sat_total,
            s.program_code,
            p.program_name,
            p.college,
            s.institutional_aid,
            s.federal_aid,
            s.unmet_need,
            s.distance_from_home_mi,
            f.mean_term_gpa,
            f.total_credits_earned,
            f.mean_lms,
            f.total_advising,
            f.ever_on_campus,
            r.returned_year_2,
            r.retention_probability_true
        FROM dim_student s
        JOIN dim_program p USING (program_code)
        JOIN first_year f USING (student_id)
        JOIN fact_retention r USING (student_id)
    """).df()

    risk_path = os.path.join(ART_DIR, "risk_scores.csv")
    if os.path.exists(risk_path):
        risk = pd.read_csv(risk_path)[
            ["student_id", "retention_probability", "risk_score", "risk_band"]
        ]
        student_view = student_view.merge(risk, on="student_id", how="left")

    sent_path = os.path.join(ART_DIR, "evaluation_sentiment.csv")
    if os.path.exists(sent_path):
        sent = pd.read_csv(sent_path)[
            ["student_id", "n_evals", "mean_rating",
             "mean_sentiment", "share_negative"]
        ]
        student_view = student_view.merge(sent, on="student_id", how="left")

    term_view = con.execute("""
        SELECT
            t.student_id,
            t.term_year,
            t.term_season,
            t.term_seq,
            t.term_gpa,
            t.credits_attempted,
            t.credits_earned,
            t.lms_logins_per_week,
            t.advising_meetings,
            t.on_campus_housing,
            s.cohort_year,
            s.gender,
            s.race_ethnicity,
            s.first_generation,
            s.pell_eligible,
            s.program_code,
            p.program_name,
            p.college,
            r.returned_year_2
        FROM fact_term_enrollment t
        JOIN dim_student s USING (student_id)
        JOIN dim_program p USING (program_code)
        JOIN fact_retention r USING (student_id)
    """).df()
    con.close()

    student_path = os.path.join(DATA_DIR, "tableau_student_view.csv")
    term_path = os.path.join(DATA_DIR, "tableau_term_view.csv")
    student_view.to_csv(student_path, index=False)
    term_view.to_csv(term_path, index=False)

    print(f"Wrote {len(student_view):,} rows -> {student_path}")
    print(f"Wrote {len(term_view):,} rows   -> {term_path}")
    print(f"Columns in student view: {len(student_view.columns)}")


if __name__ == "__main__":
    main()
