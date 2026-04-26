"""Tests for the course-evaluation generator and NLP analyzer."""
from __future__ import annotations

import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _csv(name: str) -> pd.DataFrame:
    p = os.path.join(ROOT, "data", name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not generated; run etl/generate_evaluations.py")
    return pd.read_csv(p)


def _artifact(name: str) -> pd.DataFrame:
    p = os.path.join(ROOT, "artifacts", name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not generated; run nlp/analyze_evaluations.py")
    return pd.read_csv(p)


def test_evaluations_have_expected_schema():
    evals = _csv("course_evaluations.csv")
    expected = {"student_id", "term_year", "term_season",
                "course_code", "rating", "comment"}
    assert expected.issubset(set(evals.columns))


def test_ratings_in_range_and_non_empty_comments():
    evals = _csv("course_evaluations.csv")
    assert evals["rating"].between(1, 5).all()
    assert evals["comment"].str.len().min() > 5
    assert evals["comment"].str.len().max() < 500


def test_each_student_has_2_to_4_evaluations():
    evals = _csv("course_evaluations.csv")
    counts = evals.groupby("student_id").size()
    assert counts.min() >= 2
    assert counts.max() <= 4


def test_sentiment_artifact_has_expected_columns():
    df = _artifact("evaluation_sentiment.csv")
    for c in ["student_id", "n_evals", "mean_rating", "mean_sentiment",
              "min_sentiment", "share_negative"]:
        assert c in df.columns
    assert df["mean_sentiment"].between(-1, 1).all()


def test_topics_artifact_has_six_topics():
    topics = _artifact("evaluation_topics.csv")
    assert len(topics) == 6
    assert {"topic", "label", "top_terms"}.issubset(set(topics.columns))


def test_sentiment_predicts_retention():
    """Negative-sentiment students must have lower retention than positive."""
    s = _artifact("sentiment_vs_retention.csv")
    rates = dict(zip(s["sentiment_band"], s["retention_rate"]))
    assert rates["Very Negative"] < rates["Positive"], (
        f"Expected Very Negative ({rates['Very Negative']:.2%}) < "
        f"Positive ({rates['Positive']:.2%})"
    )
