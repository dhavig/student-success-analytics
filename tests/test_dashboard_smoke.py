"""Smoke tests for the Streamlit dashboard module.

These don't run the Streamlit server; they verify the app module imports
cleanly and exposes the expected functions, so regressions in helpers or
imports get caught in CI.
"""
from __future__ import annotations

import importlib


def test_dashboard_module_imports():
    mod = importlib.import_module("dashboard.app")
    for name in ["main", "cohort_overview", "advisor_outreach",
                 "model_transparency", "load_warehouse_tables", "load_artifacts"]:
        assert hasattr(mod, name), f"dashboard.app missing {name}"


def test_dashboard_constants_point_to_repo_paths():
    """Catch regressions where DB_PATH / ART_DIR get pointed somewhere wrong."""
    import os

    import dashboard.app as app

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(app.__file__)))
    assert os.path.commonpath([app.DB_PATH, repo_root]) == repo_root
    assert os.path.commonpath([app.ART_DIR, repo_root]) == repo_root
