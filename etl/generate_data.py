"""
Synthetic student data generator.

Produces a realistic mid-sized private university first-time-in-college (FTIC)
cohort with demographics, academic preparation, financial aid, term-level
enrollment, GPA, engagement signals, and a year-2 retention outcome.

Outputs CSVs under ./data/ that the warehouse loader consumes.

All data is fully synthetic. Distributions are inspired by publicly reported
IPEDS and Common Data Set patterns for similar institutions but do not
represent any real student.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from faker import Faker

RNG_SEED = 42
N_STUDENTS = 5000
COHORT_YEARS = [2020, 2021, 2022, 2023]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

PROGRAMS = [
    ("BUS", "Business Administration", "Quinlan School of Business"),
    ("NUR", "Nursing", "Marcella Niehoff School of Nursing"),
    ("BIO", "Biology", "College of Arts and Sciences"),
    ("PSY", "Psychology", "College of Arts and Sciences"),
    ("CS",  "Computer Science", "College of Arts and Sciences"),
    ("ENG", "English", "College of Arts and Sciences"),
    ("COM", "Communication", "School of Communication"),
    ("EDU", "Education", "School of Education"),
    ("SOC", "Sociology", "College of Arts and Sciences"),
    ("POL", "Political Science", "College of Arts and Sciences"),
]

RACE_ETHNICITY = [
    ("White", 0.50),
    ("Hispanic/Latino", 0.20),
    ("Asian", 0.13),
    ("Black/African American", 0.08),
    ("Two or More Races", 0.05),
    ("International", 0.03),
    ("Unknown", 0.01),
]

GENDERS = [("Female", 0.58), ("Male", 0.41), ("Nonbinary", 0.01)]


@dataclass
class GenConfig:
    n_students: int = N_STUDENTS
    cohort_years: tuple = tuple(COHORT_YEARS)
    seed: int = RNG_SEED


def _weighted_choice(rng: np.random.Generator, items: list[tuple], n: int) -> np.ndarray:
    values, weights = zip(*items)
    weights = np.array(weights) / sum(weights)
    return rng.choice(values, size=n, p=weights)


def generate_students(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    fake = Faker()
    Faker.seed(cfg.seed)

    n = cfg.n_students
    student_ids = [f"S{100000 + i}" for i in range(n)]
    cohort_year = rng.choice(cfg.cohort_years, size=n)

    gender = _weighted_choice(rng, GENDERS, n)
    race_ethnicity = _weighted_choice(rng, RACE_ETHNICITY, n)
    first_gen = rng.choice(["Yes", "No"], size=n, p=[0.28, 0.72])
    pell_eligible = rng.choice(["Yes", "No"], size=n, p=[0.32, 0.68])

    # Academic preparation
    hs_gpa = np.clip(rng.normal(3.55, 0.35, n), 2.0, 4.5)
    sat_total = np.clip(rng.normal(1230, 130, n), 800, 1600).astype(int)

    # Program assignment
    prog_idx = rng.integers(0, len(PROGRAMS), size=n)
    program_code = [PROGRAMS[i][0] for i in prog_idx]

    # Financial aid
    cost_of_attendance = 62000
    institutional_aid = np.clip(
        rng.normal(28000, 9000, n) + (pell_eligible == "Yes") * 4000, 0, 55000
    ).round(-2)
    federal_aid = np.where(pell_eligible == "Yes",
                           np.clip(rng.normal(7000, 1500, n), 0, 14000),
                           np.clip(rng.normal(2500, 1500, n), 0, 8000)).round(-2)
    unmet_need = np.clip(cost_of_attendance - institutional_aid - federal_aid
                         - rng.normal(15000, 7000, n), 0, 50000).round(-2)

    # Distance from home (proxy for residential vs commuter dynamics)
    distance_from_home_mi = np.clip(rng.lognormal(4.5, 1.1, n), 1, 3000).astype(int)

    # Names for realism (not used in modeling)
    first_name = [fake.first_name() for _ in range(n)]
    last_name = [fake.last_name() for _ in range(n)]

    df = pd.DataFrame({
        "student_id": student_ids,
        "first_name": first_name,
        "last_name": last_name,
        "cohort_year": cohort_year,
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "first_generation": first_gen,
        "pell_eligible": pell_eligible,
        "hs_gpa": hs_gpa.round(2),
        "sat_total": sat_total,
        "program_code": program_code,
        "institutional_aid": institutional_aid,
        "federal_aid": federal_aid,
        "unmet_need": unmet_need,
        "distance_from_home_mi": distance_from_home_mi,
    })
    return df


def generate_term_enrollment(students: pd.DataFrame, cfg: GenConfig) -> pd.DataFrame:
    """Generate fall and spring term records for the first year only."""
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []

    for _, s in students.iterrows():
        for term_seq, term_label in [(1, "Fall"), (2, "Spring")]:
            # First-term GPA driven by HS GPA, SAT, with noise and aid stress
            base = 0.70 * (s["hs_gpa"] - 3.5) + 0.0020 * (s["sat_total"] - 1200)
            stress = -0.000003 * s["unmet_need"] - 0.10 * (s["first_generation"] == "Yes")
            noise = rng.normal(0, 0.30)
            term_gpa = np.clip(3.05 + base + stress + noise, 0.0, 4.0)

            credits_attempted = int(rng.choice([12, 13, 14, 15, 16, 17, 18],
                                               p=[0.05, 0.05, 0.10, 0.45, 0.20, 0.10, 0.05]))
            # Credits earned correlates with GPA
            earn_ratio = np.clip(0.6 + 0.10 * term_gpa + rng.normal(0, 0.05), 0.3, 1.0)
            credits_earned = int(round(credits_attempted * earn_ratio))

            # Engagement signals
            lms_logins_per_week = max(0, int(rng.normal(8 + 1.5 * term_gpa, 3)))
            advising_meetings = int(rng.choice([0, 1, 2, 3], p=[0.35, 0.40, 0.20, 0.05]))
            on_campus_housing = int((s["distance_from_home_mi"] > 50)
                                    and rng.random() < 0.85)

            rows.append({
                "student_id": s["student_id"],
                "term_year": int(s["cohort_year"]) + (0 if term_seq == 1 else 1),
                "term_season": term_label,
                "term_seq": term_seq,
                "term_gpa": round(float(term_gpa), 2),
                "credits_attempted": credits_attempted,
                "credits_earned": credits_earned,
                "lms_logins_per_week": lms_logins_per_week,
                "advising_meetings": advising_meetings,
                "on_campus_housing": on_campus_housing,
            })

    return pd.DataFrame(rows)


def compute_retention(students: pd.DataFrame, terms: pd.DataFrame, cfg: GenConfig) -> pd.DataFrame:
    """Generate the year-2 retention outcome with a realistic generative process."""
    rng = np.random.default_rng(cfg.seed + 2)

    first_year = (terms.groupby("student_id")
                       .agg(mean_term_gpa=("term_gpa", "mean"),
                            total_credits_earned=("credits_earned", "sum"),
                            mean_lms=("lms_logins_per_week", "mean"),
                            total_advising=("advising_meetings", "sum"),
                            ever_on_campus=("on_campus_housing", "max"))
                       .reset_index())

    df = students.merge(first_year, on="student_id", how="left")

    # Logit of retention probability. Intercept tuned so overall rate ~ 80%,
    # consistent with mid-sized private university first-year retention.
    logit = (
        0.7
        + 1.60 * (df["mean_term_gpa"] - 2.7)
        + 0.040 * (df["total_credits_earned"] - 24)
        + 0.040 * (df["mean_lms"] - 6)
        + 0.20  * df["total_advising"]
        + 0.40  * df["ever_on_campus"]
        - 0.000035 * df["unmet_need"]
        - 0.30 * (df["first_generation"] == "Yes").astype(int)
        - 0.15 * (df["pell_eligible"] == "Yes").astype(int)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    returned = (rng.random(len(df)) < prob).astype(int)

    out = df[["student_id", "cohort_year"]].copy()
    out["retention_probability_true"] = prob.round(4)
    out["returned_year_2"] = returned
    return out


def write_program_dim() -> pd.DataFrame:
    return pd.DataFrame(PROGRAMS, columns=["program_code", "program_name", "college"])


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg = GenConfig()

    print(f"Generating {cfg.n_students:,} students across cohorts {list(cfg.cohort_years)}...")
    students = generate_students(cfg)
    terms = generate_term_enrollment(students, cfg)
    retention = compute_retention(students, terms, cfg)
    programs = write_program_dim()

    students.to_csv(os.path.join(DATA_DIR, "students.csv"), index=False)
    terms.to_csv(os.path.join(DATA_DIR, "term_enrollment.csv"), index=False)
    retention.to_csv(os.path.join(DATA_DIR, "retention.csv"), index=False)
    programs.to_csv(os.path.join(DATA_DIR, "programs.csv"), index=False)

    print(f"  students.csv          {len(students):,} rows")
    print(f"  term_enrollment.csv   {len(terms):,} rows")
    print(f"  retention.csv         {len(retention):,} rows")
    print(f"  programs.csv          {len(programs):,} rows")
    print(f"\nOverall year-2 retention rate: {retention['returned_year_2'].mean():.1%}")


if __name__ == "__main__":
    main()
