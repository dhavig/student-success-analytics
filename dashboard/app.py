"""
Streamlit dashboard for the Student Success Analytics platform.

Three tabs:
  1. Cohort Overview      -- leadership view of retention trends.
  2. Advisor Outreach     -- operational list of high-risk students.
  3. Model Transparency   -- metrics, feature importance, fairness audit.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "warehouse.duckdb")
ART_DIR = os.path.join(ROOT, "artifacts")

st.set_page_config(
    page_title="Student Success Analytics",
    page_icon=None,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_warehouse_tables() -> dict[str, pd.DataFrame]:
    if not os.path.exists(DB_PATH):
        return {}
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = {
        "students": con.execute("SELECT * FROM dim_student").df(),
        "programs": con.execute("SELECT * FROM dim_program").df(),
        "terms":    con.execute("SELECT * FROM fact_term_enrollment").df(),
        "retention": con.execute("SELECT * FROM fact_retention").df(),
    }
    con.close()
    return tables


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict:
    out = {}
    risk = os.path.join(ART_DIR, "risk_scores.csv")
    fi = os.path.join(ART_DIR, "feature_importance.csv")
    fa = os.path.join(ART_DIR, "fairness_audit.csv")
    metrics = os.path.join(ART_DIR, "metrics.json")
    shap_png = os.path.join(ART_DIR, "shap_summary.png")
    if os.path.exists(risk):
        out["risk_scores"] = pd.read_csv(risk)
    if os.path.exists(fi):
        out["feature_importance"] = pd.read_csv(fi)
    if os.path.exists(fa):
        out["fairness_audit"] = pd.read_csv(fa)
    if os.path.exists(metrics):
        with open(metrics) as fh:
            out["metrics"] = json.load(fh)
    if os.path.exists(shap_png):
        out["shap_png"] = shap_png
    return out


def header() -> None:
    st.title("Student Success Analytics")
    st.caption(
        "Synthetic-data demo of an institutional research analytics platform: "
        "warehoused student data, a calibrated retention-risk model, and dashboards "
        "for leadership and advisors."
    )


def warehouse_missing_warning() -> None:
    st.warning(
        "No warehouse found yet. Run the pipeline first:\n\n"
        "```bash\n"
        "python etl/generate_data.py\n"
        "python etl/load_warehouse.py\n"
        "python models/train_retention.py\n"
        "```"
    )


def cohort_overview(tables: dict[str, pd.DataFrame]) -> None:
    students = tables["students"]
    retention = tables["retention"]
    programs = tables["programs"]
    # Drop duplicate cohort_year on the retention side; students has the same value.
    joined = (retention.drop(columns=["cohort_year"])
                       .merge(students, on="student_id")
                       .merge(programs, on="program_code"))

    cohorts = sorted(joined["cohort_year"].unique())
    selected = st.multiselect("Cohort years", cohorts, default=cohorts)
    view = joined[joined["cohort_year"].isin(selected)] if selected else joined

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in view", f"{len(view):,}")
    c2.metric("Retention rate", f"{view['returned_year_2'].mean():.1%}")
    c3.metric("Pell-eligible", f"{(view['pell_eligible'] == 'Yes').mean():.1%}")
    c4.metric("First-generation", f"{(view['first_generation'] == 'Yes').mean():.1%}")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Retention by cohort")
        cohort = (view.groupby("cohort_year")["returned_year_2"]
                      .agg(["mean", "count"])
                      .reset_index()
                      .rename(columns={"mean": "retention_rate", "count": "n"}))
        fig = px.bar(cohort, x="cohort_year", y="retention_rate",
                     text=cohort["retention_rate"].map(lambda v: f"{v:.0%}"),
                     hover_data=["n"])
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Retention by college")
        college = (view.groupby("college")["returned_year_2"]
                       .agg(["mean", "count"])
                       .reset_index()
                       .rename(columns={"mean": "retention_rate", "count": "n"})
                       .sort_values("retention_rate"))
        fig = px.bar(college, x="retention_rate", y="college", orientation="h",
                     text=college["retention_rate"].map(lambda v: f"{v:.0%}"),
                     hover_data=["n"])
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Retention gaps by student characteristic")
    gap_cols = st.columns(3)
    for col, attr, label in zip(
        gap_cols,
        ["first_generation", "pell_eligible", "gender"],
        ["First-generation", "Pell-eligible", "Gender"],
    ):
        with col:
            g = (view.groupby(attr)["returned_year_2"]
                     .agg(["mean", "count"])
                     .reset_index()
                     .rename(columns={"mean": "retention_rate", "count": "n"}))
            fig = px.bar(g, x=attr, y="retention_rate",
                         text=g["retention_rate"].map(lambda v: f"{v:.0%}"),
                         hover_data=["n"], title=label)
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)


def advisor_outreach(tables: dict[str, pd.DataFrame], artifacts: dict) -> None:
    if "risk_scores" not in artifacts:
        st.warning("No risk scores yet. Run `python models/train_retention.py`.")
        return

    risk = artifacts["risk_scores"].merge(
        tables["students"][["student_id", "hs_gpa", "unmet_need"]],
        on="student_id", how="left",
    )
    programs = tables["programs"]
    risk = risk.merge(programs, on="program_code", how="left")

    cohorts = sorted(risk["cohort_year"].unique())
    f1, f2, f3 = st.columns(3)
    sel_cohorts = f1.multiselect("Cohort years", cohorts, default=[max(cohorts)])
    bands = ["High", "Elevated", "Moderate", "Low"]
    sel_bands = f2.multiselect("Risk band", bands, default=["High", "Elevated"])
    sel_colleges = f3.multiselect("College", sorted(risk["college"].unique()), default=[])

    view = risk.copy()
    if sel_cohorts:
        view = view[view["cohort_year"].isin(sel_cohorts)]
    if sel_bands:
        view = view[view["risk_band"].isin(sel_bands)]
    if sel_colleges:
        view = view[view["college"].isin(sel_colleges)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Students flagged", f"{len(view):,}")
    c2.metric("Avg. risk score", f"{view['risk_score'].mean():.2f}" if len(view) else "—")
    c3.metric("Pell-eligible share",
              f"{(view['pell_eligible'] == 'Yes').mean():.0%}" if len(view) else "—")

    st.divider()
    st.subheader("Outreach list")
    cols = ["student_id", "cohort_year", "program_code", "college",
            "risk_band", "risk_score", "retention_probability",
            "first_generation", "pell_eligible", "hs_gpa", "unmet_need"]
    st.dataframe(view[cols].sort_values("risk_score", ascending=False),
                 use_container_width=True, height=420)

    st.download_button(
        "Download outreach list (CSV)",
        view[cols].to_csv(index=False).encode("utf-8"),
        file_name="advisor_outreach.csv",
        mime="text/csv",
    )

    st.caption(
        "Intended use: prioritize proactive advisor outreach. Not a sole basis "
        "for any consequential decision. See Model Card for limitations."
    )


def model_transparency(artifacts: dict) -> None:
    if "metrics" not in artifacts:
        st.warning("No model artifacts yet. Run `python models/train_retention.py`.")
        return

    m = artifacts["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c3.metric("Brier score", f"{m['brier']:.3f}")
    c4.metric("Test n", f"{m['n']:,}")

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Top features (mean |SHAP|)")
        fi = artifacts["feature_importance"].head(15)
        fig = px.bar(fi.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=460)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("SHAP summary")
        if "shap_png" in artifacts:
            st.image(artifacts["shap_png"], use_container_width=True)

    st.divider()
    st.subheader("Fairness audit")
    audit = artifacts["fairness_audit"].copy()
    for col in ["actual_retention_rate", "predicted_positive_rate",
                "selection_rate_at_risk", "false_negative_rate", "false_positive_rate"]:
        audit[col] = audit[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    audit["roc_auc"] = audit["roc_auc"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    st.dataframe(audit, use_container_width=True, height=360)
    st.caption(
        "Selection rate at risk = share of group flagged at probability < 0.5. "
        "Inspect FNR and FPR parity across protected groups before deploying."
    )


def main() -> None:
    header()
    tables = load_warehouse_tables()
    if not tables:
        warehouse_missing_warning()
        return
    artifacts = load_artifacts()

    tab1, tab2, tab3 = st.tabs(["Cohort Overview", "Advisor Outreach", "Model Transparency"])
    with tab1:
        cohort_overview(tables)
    with tab2:
        advisor_outreach(tables, artifacts)
    with tab3:
        model_transparency(artifacts)


if __name__ == "__main__":
    main()
