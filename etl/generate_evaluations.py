"""
Synthetic course-evaluation generator.

Produces realistic free-text comments and 1-5 star ratings for first-year
students, tied to their academic and demographic attributes so that:

  - Higher-GPA / retained students lean positive in tone.
  - Lower-GPA / non-retained students lean negative, with topic clusters
    around financial stress, advising gaps, workload, and belonging.
  - Topic vocabulary is consistent enough for LDA to recover structure.

Outputs:  data/course_evaluations.csv

Schema:
    student_id, term_year, term_season, course_code, rating, comment
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

COURSE_CODES_BY_PROGRAM = {
    "BUS": ["ECON-101", "ACCT-201", "MGMT-110", "MKTG-220"],
    "NUR": ["NURS-110", "BIOL-201", "CHEM-103", "PSYC-101"],
    "BIO": ["BIOL-110", "CHEM-110", "MATH-160", "BIOL-220"],
    "PSY": ["PSYC-101", "PSYC-220", "STAT-150", "BIOL-101"],
    "CS":  ["CMSC-150", "CMSC-260", "MATH-160", "MATH-260"],
    "ENG": ["ENGL-110", "ENGL-220", "PHIL-130", "HIST-110"],
    "COM": ["COMM-100", "COMM-220", "ENGL-110", "PSYC-101"],
    "EDU": ["EDUC-100", "PSYC-101", "ENGL-110", "MATH-150"],
    "SOC": ["SOCL-101", "PSYC-101", "STAT-150", "ENGL-110"],
    "POL": ["POLS-101", "HIST-110", "ECON-101", "ENGL-110"],
}

# Phrase banks tagged by intended topic. LDA should rediscover these clusters.
TOPIC_PHRASES = {
    "instruction_positive": [
        "the professor explained concepts clearly",
        "lectures were engaging and well organized",
        "instructor was passionate about the subject",
        "office hours were helpful and welcoming",
        "the material was challenging but rewarding",
    ],
    "instruction_negative": [
        "lecture pace was too fast to follow",
        "the professor seemed unprepared most days",
        "office hours were inaccessible",
        "feedback on assignments was unclear",
        "I felt lost during most lectures",
    ],
    "workload": [
        "the workload was overwhelming",
        "too many assignments due in the same week",
        "I could not balance this course with my other classes",
        "weekly readings were excessive",
        "exams covered material we never discussed",
    ],
    "financial_stress": [
        "I was stressed about tuition and books all semester",
        "working two jobs made it hard to attend office hours",
        "buying the required textbook was a real burden",
        "I had to choose between this class and a paid shift",
        "the lab fees were not mentioned in advance",
    ],
    "advising_support": [
        "my advisor never returned my emails",
        "I needed help and could not find it",
        "academic advising was confusing and slow",
        "I wish someone had checked in on me earlier",
        "the writing center was a lifesaver this semester",
        "tutoring helped me catch up after a rough start",
    ],
    "belonging": [
        "I felt like I did not fit in with the other students",
        "the class environment was welcoming and inclusive",
        "I made friends in study groups",
        "commuting made it hard to feel connected to campus",
        "I considered transferring during midterms",
    ],
    "career_value": [
        "this course made me excited about my major",
        "I can see how this applies to my future career",
        "the projects were relevant to real industry work",
        "I am rethinking whether this major is right for me",
        "my internship interest grew because of this class",
    ],
    "neutral": [
        "the course was okay overall",
        "average semester nothing remarkable",
        "some parts were interesting",
        "I learned a few useful things",
        "the syllabus was straightforward",
    ],
}

POSITIVE_TOPICS = ["instruction_positive", "career_value"]
NEGATIVE_TOPICS = ["instruction_negative", "workload", "financial_stress",
                   "advising_support", "belonging"]
NEUTRAL_TOPICS = ["neutral"]


def _topic_weights_for(student: pd.Series, returned: int, rng: np.random.Generator) -> dict[str, float]:
    """Return a probability distribution over topics for one student."""
    base = {t: 1.0 for t in TOPIC_PHRASES}

    # Positive lean for retained / strong students
    if returned == 1:
        for t in POSITIVE_TOPICS: base[t] *= 3.0
        for t in NEUTRAL_TOPICS:  base[t] *= 1.5
    else:
        for t in NEGATIVE_TOPICS: base[t] *= 3.0

    # Financial stress signal scales with unmet need
    if student["unmet_need"] > 20000:
        base["financial_stress"] *= 2.5
    elif student["unmet_need"] > 10000:
        base["financial_stress"] *= 1.5

    # First-gen lean toward advising and belonging
    if student["first_generation"] == "Yes":
        base["advising_support"] *= 1.8
        base["belonging"] *= 1.6

    # Low HS GPA --> instruction_negative + workload
    if student["hs_gpa"] < 3.2:
        base["instruction_negative"] *= 1.6
        base["workload"] *= 1.6

    # Smooth a touch with noise
    for t in base:
        base[t] *= (0.85 + 0.30 * rng.random())

    s = sum(base.values())
    return {t: v / s for t, v in base.items()}


def _sample_comment(weights: dict[str, float], rng: np.random.Generator) -> tuple[str, str]:
    """Pick a topic, then assemble a 1-2 phrase comment from that topic."""
    topics = list(weights.keys())
    probs = np.array([weights[t] for t in topics])
    primary = rng.choice(topics, p=probs / probs.sum())

    n_phrases = 1 if rng.random() < 0.55 else 2
    pool = TOPIC_PHRASES[primary][:]
    rng.shuffle(pool)
    phrases = pool[:n_phrases]

    # 30% chance of mixing one phrase from a different topic for realism
    if n_phrases == 2 and rng.random() < 0.30:
        other = rng.choice([t for t in topics if t != primary])
        phrases[1] = rng.choice(TOPIC_PHRASES[other])

    text = ". ".join(p.capitalize() for p in phrases) + "."
    return primary, text


def _rating_for_topic(primary: str, rng: np.random.Generator) -> int:
    """Map primary topic to a plausible 1-5 rating."""
    if primary in POSITIVE_TOPICS:
        return int(rng.choice([4, 5], p=[0.35, 0.65]))
    if primary in NEGATIVE_TOPICS:
        return int(rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25]))
    return int(rng.choice([2, 3, 4], p=[0.15, 0.55, 0.30]))


def main() -> None:
    students_path = os.path.join(DATA_DIR, "students.csv")
    retention_path = os.path.join(DATA_DIR, "retention.csv")
    if not (os.path.exists(students_path) and os.path.exists(retention_path)):
        raise SystemExit("Missing data/students.csv or data/retention.csv. "
                         "Run `python etl/generate_data.py` first.")

    students = pd.read_csv(students_path)
    retention = pd.read_csv(retention_path)[["student_id", "returned_year_2"]]
    df = students.merge(retention, on="student_id")

    rng = np.random.default_rng(101)
    rows = []
    for _, s in df.iterrows():
        n_evals = int(rng.choice([2, 3, 4], p=[0.20, 0.55, 0.25]))
        course_pool = COURSE_CODES_BY_PROGRAM.get(s["program_code"], ["ELEC-101"])
        weights = _topic_weights_for(s, int(s["returned_year_2"]), rng)

        for i in range(n_evals):
            term_year = int(s["cohort_year"]) + (0 if i < 2 else 1)
            term_season = "Fall" if i % 2 == 0 else "Spring"
            course = course_pool[i % len(course_pool)]
            primary, comment = _sample_comment(weights, rng)
            rating = _rating_for_topic(primary, rng)
            rows.append({
                "student_id": s["student_id"],
                "term_year": term_year,
                "term_season": term_season,
                "course_code": course,
                "rating": rating,
                "comment": comment,
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "course_evaluations.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} course evaluations to {out_path}")
    print(f"  Mean rating:   {out['rating'].mean():.2f}")
    print(f"  Comment chars: median {out['comment'].str.len().median():.0f}, "
          f"max {out['comment'].str.len().max()}")


if __name__ == "__main__":
    main()
