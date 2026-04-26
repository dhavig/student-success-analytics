"""
Build notebooks/01_ir_memo_walkthrough.ipynb programmatically using nbformat.

The notebook reads as an IR memo: business question -> warehouse query ->
analysis -> finding -> recommendation. Designed as a teaching artifact to
demonstrate the JD's "cross-train OIRA's research analysts" capacity-building
responsibility.
"""

from __future__ import annotations

import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "01_ir_memo_walkthrough.ipynb")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def main() -> None:
    cells = []

    cells.append(md("""# IR Memo Walkthrough — First-Year Retention Risk

**Audience.** OIRA research analysts and program leadership.
**Purpose.** A reproducible walkthrough of the Student Success Analytics pipeline, written as an IR memo so analysts can adapt the pattern to their own questions.

This notebook follows the analytic pattern OIRA uses every day:

1. **Business question** — what does the provost or dean want to know?
2. **Warehouse query** — pull a clean modeling table from the star schema.
3. **Analysis** — descriptive, then predictive.
4. **Finding** — the headline number plus the equity lens.
5. **Recommendation** — operational next step.

> All data is synthetic. See `MODEL_CARD.md` for assumptions and the responsible-use checklist.
"""))

    cells.append(md("""## 0. Setup

Load the warehouse and the model artifacts. Re-run the pipeline first if these don't exist:

```bash
python etl/generate_data.py
python etl/generate_evaluations.py
python etl/load_warehouse.py
python models/train_retention.py
python nlp/analyze_evaluations.py
```
"""))

    cells.append(code("""import os, sys
ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT)

import duckdb
import pandas as pd
import matplotlib.pyplot as plt

DB = os.path.join(ROOT, "data", "warehouse.duckdb")
ART = os.path.join(ROOT, "artifacts")
con = duckdb.connect(DB, read_only=True)
print("Tables in warehouse:")
for r in con.execute("SHOW TABLES").fetchall():
    print(" -", r[0])"""))

    cells.append(md("""## 1. Business question

> *What is the year-2 retention rate for our most recent FTIC cohort, and how does it differ for first-generation and Pell-eligible students?*

This question comes up in every cabinet briefing. Headline numbers in 30 seconds, equity gaps in the next 60.
"""))

    cells.append(code("""headline = con.execute(\"\"\"
    SELECT
        COUNT(*)                              AS cohort_n,
        ROUND(AVG(returned_year_2) * 100, 1)  AS retention_pct
    FROM fact_retention
    WHERE cohort_year = (SELECT MAX(cohort_year) FROM fact_retention)
\"\"\").df()
headline"""))

    cells.append(code("""gaps = con.execute(\"\"\"
    SELECT
        s.first_generation,
        s.pell_eligible,
        COUNT(*)                              AS n,
        ROUND(AVG(r.returned_year_2) * 100, 1) AS retention_pct
    FROM fact_retention r
    JOIN dim_student s USING (student_id)
    WHERE r.cohort_year = (SELECT MAX(cohort_year) FROM fact_retention)
    GROUP BY 1, 2
    ORDER BY 1, 2
\"\"\").df()
gaps"""))

    cells.append(md("""**Read-out.** The headline is fine, but the intersection of first-gen + Pell-eligible is meaningfully below the cohort average. That's the row you'd brief to the VP for Student Success.
"""))

    cells.append(md("""## 2. Predictive lens

> *Which students should advisors call this week?*

The trained risk model is on disk. Load the four-tier band and look at its predictive lift.
"""))

    cells.append(code("""risk = pd.read_csv(os.path.join(ART, "risk_scores.csv"))
band_lift = (risk.groupby("risk_band", observed=True)
                  .agg(n=("student_id", "count"),
                       retention_rate=("actual_returned_year_2", "mean"))
                  .reset_index())
band_lift["risk_band"] = pd.Categorical(
    band_lift["risk_band"], categories=["Low", "Moderate", "Elevated", "High"], ordered=True)
band_lift = band_lift.sort_values("risk_band")
band_lift"""))

    cells.append(code("""fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(band_lift["risk_band"].astype(str), band_lift["retention_rate"],
       color=["#FFC629", "#E0A82E", "#C77018", "#7A0019"])
ax.set_ylim(0, 1)
ax.set_ylabel("Year-2 retention rate")
ax.set_title("Predictive lift across risk bands")
for i, (n, r) in enumerate(zip(band_lift["n"], band_lift["retention_rate"])):
    ax.text(i, r + 0.02, f"{r:.0%}\\nn={n}", ha="center")
plt.tight_layout(); plt.show()"""))

    cells.append(md("""**Read-out.** The High band is small but very actionable. A single advisor outreach campaign covering ~25 students could move an attainable share of them. The Elevated band (~200) is the larger volume opportunity.
"""))

    cells.append(md("""## 3. Voice-of-student lens

> *Are students writing things in course evaluations that the numeric warehouse misses?*

This is where the unstructured-text pipeline pays off.
"""))

    cells.append(code("""sent_v_ret = pd.read_csv(os.path.join(ART, "sentiment_vs_retention.csv"))
sent_v_ret"""))

    cells.append(code("""topics = pd.read_csv(os.path.join(ART, "evaluation_topics.csv"))
topics[["topic", "label", "top_terms"]]"""))

    cells.append(md("""**Read-out.** Negative-sentiment students retain at materially lower rates than positive-sentiment peers — a signal that pure numeric features (GPA, aid) capture only partially. Topic 3 (financial stress) and topic 4 (advising support) are the recurring narratives behind that gap.
"""))

    cells.append(md("""## 4. Recommendation (operational)

1. **This week.** Hand the High + Elevated risk band lists from `artifacts/risk_scores.csv` to the FTIC advising team for outreach.
2. **This term.** Pilot a paired intervention for first-gen + Pell students whose course evaluations include financial-stress phrases — a 1:1 financial-aid counseling appointment, plus a check-in from the academic advisor.
3. **Next year.** Add survey-instrument items the model card flags as missing (mental health, basic-needs insecurity) and re-train; re-audit fairness before any operational use.

## 5. For research analysts adapting this pattern

The same five-step pattern works for any IR question:

1. Frame the business question in one sentence.
2. Pull a flat modeling table from the warehouse with one CTE per logical step.
3. Compute headline + equity-cut numbers.
4. Add a predictive or text lens if the question warrants it.
5. Close with a one-paragraph recommendation that names the operational owner.

For exercises, see `sql/analytics_queries.sql` — six reference queries that map to the most common cabinet asks.
"""))

    cells.append(code("""con.close()"""))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        nbf.write(nb, fh)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
