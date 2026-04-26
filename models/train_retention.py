"""
Train and evaluate a first-year retention risk model.

- Pulls a modeling table from the DuckDB warehouse.
- Trains an XGBoost classifier with calibrated probabilities.
- Reports overall metrics (ROC-AUC, PR-AUC, Brier).
- Computes SHAP values for global and local explainability.
- Runs a fairness audit across gender, first-generation status, Pell
  eligibility, and race/ethnicity.
- Writes:
    artifacts/retention_model.joblib
    artifacts/feature_importance.csv
    artifacts/risk_scores.csv
    artifacts/fairness_audit.csv
    artifacts/metrics.json
    artifacts/shap_summary.png
"""

from __future__ import annotations

import json
import os
import warnings

import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "warehouse.duckdb")
ART_DIR = os.path.join(ROOT, "artifacts")

NUMERIC_FEATURES = [
    "hs_gpa", "sat_total", "institutional_aid", "federal_aid",
    "unmet_need", "distance_from_home_mi",
    "mean_term_gpa", "total_credits_earned",
    "mean_lms", "total_advising", "ever_on_campus",
]
CATEGORICAL_FEATURES = ["program_code"]
PROTECTED_ATTRIBUTES = ["gender", "race_ethnicity", "first_generation", "pell_eligible"]
TARGET = "returned_year_2"


def load_modeling_table() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Warehouse not found at {DB_PATH}. "
            "Run `python etl/generate_data.py` then `python etl/load_warehouse.py`."
        )

    query = """
    WITH first_year AS (
        SELECT
            student_id,
            AVG(term_gpa)                AS mean_term_gpa,
            SUM(credits_earned)          AS total_credits_earned,
            AVG(lms_logins_per_week)     AS mean_lms,
            SUM(advising_meetings)       AS total_advising,
            MAX(on_campus_housing)       AS ever_on_campus
        FROM fact_term_enrollment
        GROUP BY student_id
    )
    SELECT
        s.student_id,
        s.cohort_year,
        s.gender, s.race_ethnicity, s.first_generation, s.pell_eligible,
        s.hs_gpa, s.sat_total, s.program_code,
        s.institutional_aid, s.federal_aid, s.unmet_need,
        s.distance_from_home_mi,
        f.mean_term_gpa, f.total_credits_earned,
        f.mean_lms, f.total_advising, f.ever_on_campus,
        r.returned_year_2
    FROM dim_student s
    JOIN first_year f USING (student_id)
    JOIN fact_retention r USING (student_id);
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(query).df()
    con.close()
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    base = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    # Calibration stabilizes probabilities for the advisor outreach use case.
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy_at_0.5": float((y_pred == y_true).mean()),
        "true_positive": int(tp), "false_positive": int(fp),
        "true_negative": int(tn), "false_negative": int(fn),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def fairness_audit(meta: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    y_pred = (y_prob >= 0.5).astype(int)
    for attr in PROTECTED_ATTRIBUTES:
        for value, idx in meta.groupby(attr).groups.items():
            idx = list(idx)
            if len(idx) < 30:
                continue
            yt = y_true[idx]
            yp = y_pred[idx]
            ypb = y_prob[idx]
            try:
                auc = roc_auc_score(yt, ypb) if len(np.unique(yt)) > 1 else np.nan
            except ValueError:
                auc = np.nan
            rows.append({
                "attribute": attr,
                "group": str(value),
                "n": len(idx),
                "actual_retention_rate": float(yt.mean()),
                "predicted_positive_rate": float(yp.mean()),
                "selection_rate_at_risk": float((ypb < 0.5).mean()),
                "roc_auc": auc,
                "false_negative_rate": float(((yp == 0) & (yt == 1)).sum() / max((yt == 1).sum(), 1)),
                "false_positive_rate": float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)),
            })
    audit = pd.DataFrame(rows).sort_values(["attribute", "group"]).reset_index(drop=True)
    return audit


def shap_summary(pipe: Pipeline, X_sample: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """Compute SHAP on the underlying XGBoost (using one of the calibrated estimators)."""
    import shap

    pre = pipe.named_steps["pre"]
    cal = pipe.named_steps["clf"]
    booster = cal.calibrated_classifiers_[0].estimator
    feature_names = (NUMERIC_FEATURES
                     + list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)))

    X_trans = pre.transform(X_sample)
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X_trans)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_trans, feature_names=feature_names, show=False, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

    importance = (pd.DataFrame({"feature": feature_names,
                                "mean_abs_shap": np.abs(sv).mean(axis=0)})
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True))
    return importance


def main() -> None:
    os.makedirs(ART_DIR, exist_ok=True)

    df = load_modeling_table()
    print(f"Modeling table: {len(df):,} rows, {df[TARGET].mean():.1%} year-2 retention")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values
    meta = df[["student_id", "cohort_year"] + PROTECTED_ATTRIBUTES].reset_index(drop=True)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.25, stratify=y, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_prob_test = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_prob_test)
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:22s} {v}")

    audit = fairness_audit(meta_test.reset_index(drop=True), y_test, y_prob_test)
    print("\nFairness audit (test set):")
    print(audit.to_string(index=False))

    # Score the full population for the dashboard.
    full_prob = pipe.predict_proba(X)[:, 1]
    risk_scores = df[["student_id", "cohort_year"] + PROTECTED_ATTRIBUTES + ["program_code"]].copy()
    risk_scores["retention_probability"] = full_prob.round(4)
    # Risk = 1 - probability of returning. Bands chosen for advisor triage.
    risk_scores["risk_score"] = (1 - full_prob).round(4)
    risk_scores["risk_band"] = pd.cut(
        risk_scores["risk_score"],
        bins=[-0.01, 0.20, 0.40, 0.60, 1.01],
        labels=["Low", "Moderate", "Elevated", "High"],
    )
    risk_scores["actual_returned_year_2"] = y

    importance = shap_summary(pipe, X_train.sample(min(800, len(X_train)), random_state=42),
                              os.path.join(ART_DIR, "shap_summary.png"))

    joblib.dump(pipe, os.path.join(ART_DIR, "retention_model.joblib"))
    risk_scores.to_csv(os.path.join(ART_DIR, "risk_scores.csv"), index=False)
    audit.to_csv(os.path.join(ART_DIR, "fairness_audit.csv"), index=False)
    importance.to_csv(os.path.join(ART_DIR, "feature_importance.csv"), index=False)
    with open(os.path.join(ART_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nArtifacts written to {ART_DIR}")


if __name__ == "__main__":
    main()
