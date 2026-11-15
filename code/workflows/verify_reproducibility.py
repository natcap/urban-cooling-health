#!/usr/bin/env python3
"""Run lightweight, non-model reproducibility checks for the current manuscript package."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).parent / "configs"
FIGURE_CONFIG = CONFIG_DIR / "manuscript_figure_package_2026-11-13.csv"
FIGURE_BUILDER = Path(__file__).parent / "build_current_figure_package.py"
RUN_REGISTRY = CONFIG_DIR / "run_registry.csv"


def check_registry() -> list[str]:
    failures: list[str] = []
    required = {
        "run_id", "run_date", "status", "scope", "software",
        "configuration_or_source", "output_root", "primary_scripts", "figures",
    }
    with RUN_REGISTRY.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            return [f"run registry lacks required columns: {RUN_REGISTRY}"]
        rows = list(reader)
    if not rows:
        return ["run registry has no rows"]
    identifiers = [row["run_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        failures.append("run registry contains duplicate run_id values")
    for row in rows:
        for script in [item.strip() for item in row["primary_scripts"].split(";")]:
            if script and not (REPO_ROOT / script).is_file():
                failures.append(f"registry script missing for {row['run_id']}: {script}")
    return failures


def check_environment() -> list[str]:
    failures: list[str] = []
    if sys.version_info < (3, 10):
        failures.append("Python 3.10 or newer is required for the lightweight checks")
    if not shutil.which("Rscript"):
        failures.append("Rscript is not on PATH; R figure rendering cannot be checked")
    for path in (
        REPO_ROOT / "code/Urban_Cooling_Modeling_Runs/environment-invest-3.20.2.yml",
        REPO_ROOT / "code/health_assessment/environment-health.yml",
        REPO_ROOT / ".Renviron.example",
    ):
        if not path.is_file():
            failures.append(f"required environment guidance is missing: {path.relative_to(REPO_ROOT)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("URBAN_COOLING_DATA_ROOT"))
    parser.add_argument("--skip-figures", action="store_true", help="Do not verify copied current figures")
    parser.add_argument("--require-data-root", action="store_true", help="Fail if the external data root is absent")
    args = parser.parse_args()

    failures = check_registry() + check_environment()
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    if args.require_data_root and (data_root is None or not data_root.is_dir()):
        failures.append("URBAN_COOLING_DATA_ROOT is required and must point to an existing directory")
    if not args.skip_figures:
        if data_root is None or not data_root.is_dir():
            failures.append("Figure-package verification needs URBAN_COOLING_DATA_ROOT")
        else:
            result = subprocess.run(
                [sys.executable, str(FIGURE_BUILDER), "--config", str(FIGURE_CONFIG),
                 "--data-root", str(data_root), "--verify"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            if result.returncode:
                failures.append("figure-package verification failed: " + result.stderr.strip())
            else:
                print(result.stdout.strip())
    if failures:
        print("Reproducibility checks failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Reproducibility checks passed: registry, paths, environment guidance and figure package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
