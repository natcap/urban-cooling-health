"""Portable tests for the curated manuscript-figure package workflow."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "code/workflows/build_current_figure_package.py"


class CurrentFigurePackageTest(unittest.TestCase):
    def test_build_then_verify_detects_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            source.write_bytes(b"first-figure")
            config = root / "package.csv"
            with config.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target, fieldnames=["asset", "role", "status", "source", "generator"]
                )
                writer.writeheader()
                writer.writerow({
                    "asset": "figure_01.png", "role": "Main Figure 1", "status": "selected",
                    "source": str(source), "generator": "example.py",
                })
            output = root / "current"
            build = subprocess.run(
                [sys.executable, str(BUILDER), "--config", str(config), "--output", str(output)],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            verify = subprocess.run(
                [sys.executable, str(BUILDER), "--config", str(config), "--output", str(output), "--verify"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            self.assertEqual(verify.returncode, 0, verify.stderr)
            source.write_bytes(b"changed-figure")
            stale = subprocess.run(
                [sys.executable, str(BUILDER), "--config", str(config), "--output", str(output), "--verify"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(stale.returncode, 0)
            refreshed = subprocess.run(
                [sys.executable, str(BUILDER), "--config", str(config), "--output", str(output), "--refresh"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            self.assertEqual(refreshed.returncode, 0, refreshed.stderr)

    def test_repository_registry_has_valid_scripts(self) -> None:
        environment = os.environ.copy()
        environment["URBAN_COOLING_DATA_ROOT"] = str(REPO_ROOT)
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "code/workflows/verify_reproducibility.py"), "--skip-figures"],
            cwd=REPO_ROOT, env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
