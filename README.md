# Student Success Analytics

End-to-end institutional research analytics platform: synthetic student data → warehouse → predictive retention model → interactive dashboard.

Built as a portfolio project aligned to the Senior Data Scientist role at Loyola University Chicago's Office of Institutional Research & Analysis.

## What's inside

| Layer | Tech | Purpose |
|---|---|---|
| **Data engineering** | Python, DuckDB, SQL | Generates synthetic student records and loads them into a star-schema warehouse. |
| **Advanced analytics** | scikit-learn, XGBoost, SHAP | First-year retention risk model with explainability and a fairness audit across demographic slices. |
| **Business intelligence** | Streamlit, Plotly | Interactive dashboard: cohort retention trends, risk score distribution, advisor outreach list. |
| **Documentation** | Markdown | README case study + Model Card describing assumptions, metrics, and responsible-use guidance. |

## Project structure

```
student-success-analytics/
├── etl/
│   ├── generate_data.py       # Synthetic student/enrollment/aid generator
│   └── load_warehouse.py      # Builds DuckDB star schema and loads data
├── sql/
│   ├── schema.sql             # Star schema DDL (fact + dim tables)
│   └── analytics_queries.sql  # Example IR-style analytic queries
├── models/
│   └── train_retention.py     # XGBoost model + SHAP + fairness audit
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── data/                      # Generated CSVs and DuckDB file (gitignored)
├── artifacts/                 # Trained model, plots, metrics (gitignored)
├── notebooks/                 # Optional exploratory notebooks
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate synthetic data and build the warehouse
python etl/generate_data.py
python etl/load_warehouse.py

# 3. Train the retention model
python models/train_retention.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

## Case study: First-year retention risk

**Business question.** Which first-year students are at elevated risk of not returning for their second year, and what factors are driving that risk?

**Approach.**
1. Generate a realistic synthetic cohort (~5,000 students) modeled on a mid-sized private university — demographics, academic preparation, financial aid, first-term GPA, engagement signals.
2. Load into a DuckDB star schema (`fact_term_enrollment`, `dim_student`, `dim_program`, `dim_term`).
3. Train a gradient-boosted classifier predicting `returned_year_2`. Evaluate with ROC-AUC, PR-AUC, and calibration.
4. Explain individual predictions with SHAP. Audit subgroup performance (gender, first-generation status, Pell eligibility, race/ethnicity) for parity.
5. Surface results in a dashboard with three views: leadership summary, advisor outreach list, and model transparency.

**Why this matters for Loyola IR.** This mirrors a real-world OIRA workflow — translating warehoused student data into both strategic insight (cohort dashboards) and operational impact (risk-flagged students for advisor outreach), with the responsible-AI guardrails the role explicitly calls out.

## Responsible use

All data here is synthetic. The fairness audit in `models/train_retention.py` is a starting point, not a sign-off; in production any retention-risk model would require Title IX, FERPA, and IRB review, ongoing drift monitoring, and a human-in-the-loop intervention design. See `MODEL_CARD.md`.

## License

MIT — for portfolio and educational use.
