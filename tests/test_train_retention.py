"""Tests for the retention model training pipeline.

These tests exercise the training functions on a small in-memory cohort
without touching the on-disk warehouse or artifacts/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.train_retention import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                                    PROTECTED_ATTRIBUTES, TARGET,
                                    build_pipeline, evaluate, fairness_audit)


@pytest.fixture(scope="module")
def modeling_table(small_cohort) -> pd.DataFrame:
    """Build the same modeling table the training script builds, in memory."""
    students = small_cohort["students"].drop(columns=["first_name", "last_name"])
    terms = small_cohort["terms"]
    retention = small_cohort["retention"]

    first_year = (terms.groupby("student_id")
                       .agg(mean_term_gpa=("term_gpa", "mean"),
                            total_credits_earned=("credits_earned", "sum"),
                            mean_lms=("lms_logins_per_week", "mean"),
                            total_advising=("advising_meetings", "sum"),
                            ever_on_campus=("on_campus_housing", "max"))
                       .reset_index())

    df = (students.merge(first_year, on="student_id")
                  .merge(retention[["student_id", "returned_year_2"]],
                         on="student_id"))
    return df


def test_pipeline_builds():
    pipe = build_pipeline()
    assert "pre" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_pipeline_fits_and_predicts(modeling_table):
    pipe = build_pipeline()
    X = modeling_table[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = modeling_table[TARGET].values
    pipe.fit(X, y)

    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_evaluate_returns_expected_keys(modeling_table):
    pipe = build_pipeline()
    X = modeling_table[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = modeling_table[TARGET].values
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]

    metrics = evaluate(y, proba)
    expected = {"roc_auc", "pr_auc", "brier", "accuracy_at_0.5",
                "true_positive", "false_positive", "true_negative",
                "false_negative", "n", "positive_rate"}
    assert expected.issubset(set(metrics.keys()))
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["brier"] <= 1


def test_protected_attributes_excluded_from_features():
    """Critical fairness invariant: protected attributes must not be inputs."""
    feature_set = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    for attr in PROTECTED_ATTRIBUTES:
        assert attr not in feature_set, \
            f"Protected attribute {attr!r} leaked into model features"


def test_fairness_audit_covers_protected_attributes(modeling_table):
    pipe = build_pipeline()
    X = modeling_table[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = modeling_table[TARGET].values
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]

    meta = modeling_table[PROTECTED_ATTRIBUTES].reset_index(drop=True)
    audit = fairness_audit(meta, y, proba)

    assert not audit.empty
    assert {"attribute", "group", "n", "false_negative_rate",
            "false_positive_rate", "roc_auc"}.issubset(set(audit.columns))
    # FNR/FPR are rates in [0, 1].
    assert audit["false_negative_rate"].between(0, 1).all()
    assert audit["false_positive_rate"].between(0, 1).all()


@pytest.mark.slow
def test_model_beats_naive_baseline(modeling_table):
    """Trained model should beat the always-predict-majority-class baseline."""
    from sklearn.model_selection import train_test_split

    X = modeling_table[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = modeling_table[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                              test_size=0.3, random_state=0)
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    metrics = evaluate(y_te, proba)
    # Even on the small 200-row cohort the model should clear AUC=0.5.
    assert metrics["roc_auc"] > 0.55, \
        f"Model AUC {metrics['roc_auc']:.3f} no better than chance"
