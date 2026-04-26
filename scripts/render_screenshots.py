"""
Render polished standalone PNGs of the dashboard's key visualizations
for embedding in the README. Uses matplotlib so it works without a browser.

Run after the warehouse and model artifacts exist.

Outputs to docs/:
    01_cohort_overview.png     KPI bar -- retention by cohort and college
    02_retention_gaps.png      Gaps by first-gen, Pell, gender
    03_risk_distribution.png   Risk score histogram + band lift
    04_feature_importance.png  Top features by mean |SHAP|
    05_fairness_audit.png      Per-group fairness table render
"""

from __future__ import annotations

import json
import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "warehouse.duckdb")
ART_DIR = os.path.join(ROOT, "artifacts")
DOCS_DIR = os.path.join(ROOT, "docs")

# Loyola-inspired palette
MAROON = "#7A0019"
GOLD = "#FFC629"
DARK = "#2E2E2E"
GREY = "#9A9A9A"
LIGHT = "#EEEEEE"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelcolor": DARK,
    "axes.edgecolor": GREY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def load() -> dict:
    con = duckdb.connect(DB_PATH, read_only=True)
    students = con.execute("SELECT * FROM dim_student").df()
    retention = con.execute("SELECT * FROM fact_retention").df()
    programs = con.execute("SELECT * FROM dim_program").df()
    con.close()

    risk = pd.read_csv(os.path.join(ART_DIR, "risk_scores.csv"))
    fi = pd.read_csv(os.path.join(ART_DIR, "feature_importance.csv"))
    fa = pd.read_csv(os.path.join(ART_DIR, "fairness_audit.csv"))
    with open(os.path.join(ART_DIR, "metrics.json")) as fh:
        metrics = json.load(fh)

    joined = (retention.drop(columns=["cohort_year"])
                       .merge(students, on="student_id")
                       .merge(programs, on="program_code"))
    return {"risk": risk, "fi": fi, "fa": fa, "metrics": metrics, "joined": joined}


def _bar_label_pct(ax, bars, values):
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.0%}", ha="center", va="bottom",
                fontsize=11, color=DARK)


def render_cohort_overview(d: dict, out: str) -> None:
    j = d["joined"]
    cohort = (j.groupby("cohort_year")["returned_year_2"].mean().reset_index()
               .rename(columns={"returned_year_2": "rate"}))
    college = (j.groupby("college")["returned_year_2"].mean().reset_index()
                .rename(columns={"returned_year_2": "rate"})
                .sort_values("rate"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [0.9, 1.4]})
    fig.suptitle("Cohort Overview - Year-2 Retention", fontsize=17, fontweight="bold",
                 color=DARK, x=0.02, ha="left", y=1.02)

    bars = axes[0].bar(cohort["cohort_year"].astype(str), cohort["rate"],
                       color=MAROON, width=0.65)
    _bar_label_pct(axes[0], bars, cohort["rate"])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_yticks(np.arange(0, 1.01, 0.2))
    axes[0].set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
    axes[0].set_title("Retention rate by cohort", loc="left", color=DARK)
    axes[0].set_xlabel("Cohort year", color=DARK)
    axes[0].grid(axis="y", color=LIGHT, linewidth=1)
    axes[0].set_axisbelow(True)

    bars = axes[1].barh(college["college"], college["rate"], color=GOLD,
                        edgecolor=DARK, linewidth=0.4)
    for bar, v in zip(bars, college["rate"]):
        axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{v:.0%}", va="center", fontsize=11, color=DARK)
    axes[1].set_xlim(0, 1.08)
    axes[1].set_xticks(np.arange(0, 1.01, 0.2))
    axes[1].set_xticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
    axes[1].set_title("Retention rate by college", loc="left", color=DARK)
    axes[1].grid(axis="x", color=LIGHT, linewidth=1)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def render_gaps(d: dict, out: str) -> None:
    j = d["joined"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Retention gaps by student characteristic", fontsize=17,
                 fontweight="bold", color=DARK, x=0.02, ha="left", y=1.04)

    for ax, attr, label, color in zip(
        axes,
        ["first_generation", "pell_eligible", "gender"],
        ["First-generation", "Pell-eligible", "Gender"],
        [MAROON, GOLD, DARK],
    ):
        g = (j.groupby(attr)["returned_year_2"].mean().reset_index()
              .rename(columns={"returned_year_2": "rate"}))
        bars = ax.bar(g[attr].astype(str), g["rate"], color=color, width=0.55)
        _bar_label_pct(ax, bars, g["rate"])
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
        ax.set_title(label, loc="left", color=DARK)
        ax.grid(axis="y", color=LIGHT, linewidth=1)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def render_risk(d: dict, out: str) -> None:
    risk = d["risk"]
    band_order = ["Low", "Moderate", "Elevated", "High"]
    band_stats = (risk.groupby("risk_band", observed=True)["actual_returned_year_2"]
                       .agg(["mean", "count"]).reset_index()
                       .rename(columns={"mean": "retention_rate", "count": "n"}))
    band_stats["risk_band"] = pd.Categorical(band_stats["risk_band"],
                                              categories=band_order, ordered=True)
    band_stats = band_stats.sort_values("risk_band")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("Risk score distribution and predictive lift", fontsize=17,
                 fontweight="bold", color=DARK, x=0.02, ha="left", y=1.02)

    axes[0].hist(risk["risk_score"], bins=40, color=MAROON, edgecolor="white")
    axes[0].set_title("Risk score distribution (1 - P(return))", loc="left", color=DARK)
    axes[0].set_xlabel("Risk score")
    axes[0].set_ylabel("Students")
    axes[0].grid(axis="y", color=LIGHT, linewidth=1)
    axes[0].set_axisbelow(True)
    for x, label in [(0.20, "Moderate"), (0.40, "Elevated"), (0.60, "High")]:
        axes[0].axvline(x, color=GREY, linestyle="--", linewidth=1)
        axes[0].text(x, axes[0].get_ylim()[1] * 0.95, f"  {label}",
                     ha="left", va="top", fontsize=9, color=GREY)

    band_colors = [GOLD, "#E0A82E", "#C77018", MAROON]
    bars = axes[1].bar(band_stats["risk_band"].astype(str), band_stats["retention_rate"],
                       color=band_colors, width=0.55)
    for bar, r, n in zip(bars, band_stats["retention_rate"], band_stats["n"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f"{r:.0%}\nn={n:,}", ha="center", va="bottom",
                     fontsize=11, color=DARK)
    axes[1].set_ylim(0, 1.18)
    axes[1].set_yticks(np.arange(0, 1.01, 0.2))
    axes[1].set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
    axes[1].set_title("Actual year-2 retention rate by predicted risk band",
                      loc="left", color=DARK)
    axes[1].grid(axis="y", color=LIGHT, linewidth=1)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def render_features(d: dict, out: str) -> None:
    fi = d["fi"].head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(fi["feature"], fi["mean_abs_shap"], color=MAROON,
                   edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, fi["mean_abs_shap"]):
        ax.text(bar.get_width() + max(fi["mean_abs_shap"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=11, color=DARK)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top retention drivers (mean |SHAP value|)",
                 loc="left", color=DARK, pad=14)
    ax.grid(axis="x", color=LIGHT, linewidth=1)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def render_fairness(d: dict, out: str) -> None:
    fa = d["fa"].copy()
    table = pd.DataFrame({
        "Attribute": fa["attribute"],
        "Group": fa["group"],
        "n": fa["n"].astype(int),
        "Actual retention %": (fa["actual_retention_rate"] * 100).round(1),
        "ROC-AUC": fa["roc_auc"].round(3),
        "FNR %": (fa["false_negative_rate"] * 100).round(2),
        "FPR %": (fa["false_positive_rate"] * 100).round(1),
    })

    fig, ax = plt.subplots(figsize=(14, 1.6 + 0.55 * len(table)))
    ax.axis("off")
    ax.set_title("Fairness audit: per-group performance",
                 loc="left", color=DARK, fontsize=20, fontweight="bold",
                 pad=24, x=0.0)

    cell_text = [[str(v) for v in row] for row in table.values]
    tab = ax.table(cellText=cell_text, colLabels=list(table.columns),
                   cellLoc="left", colLoc="left", loc="center",
                   bbox=[0.0, 0.0, 1.0, 0.92])
    tab.auto_set_font_size(False)
    tab.set_fontsize(13)
    n_cols = len(table.columns)
    for col in range(n_cols):
        cell = tab[0, col]
        cell.set_facecolor(MAROON)
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")
    for row in range(1, len(table) + 1):
        for col in range(n_cols):
            cell = tab[row, col]
            cell.set_edgecolor(LIGHT)
            cell.set_facecolor("#FAFAFA" if row % 2 else "white")

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    d = load()
    render_cohort_overview(d, os.path.join(DOCS_DIR, "01_cohort_overview.png"))
    render_gaps(d, os.path.join(DOCS_DIR, "02_retention_gaps.png"))
    render_risk(d, os.path.join(DOCS_DIR, "03_risk_distribution.png"))
    render_features(d, os.path.join(DOCS_DIR, "04_feature_importance.png"))
    render_fairness(d, os.path.join(DOCS_DIR, "05_fairness_audit.png"))


if __name__ == "__main__":
    main()
