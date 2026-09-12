"""Tests for command construction in the unified UCM pipeline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
RUNNER = WORKFLOW_DIR / "run_ucm_pipeline.py"
EXAMPLE_CONFIG = WORKFLOW_DIR / "ucm_pipeline.example.json"


class PipelineCommandTest(unittest.TestCase):
    """Verify the workflow plan without importing or running InVEST."""

    def test_dry_run_uses_one_mixed_temperature_ucm_command(self) -> None:
        with tempfile.TemporaryDirectory() as data_root:
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--config",
                    str(EXAMPLE_CONFIG),
                    "--data-root",
                    data_root,
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("[ucm]", result.stdout)
        self.assertIn("--temperatures 25 28", result.stdout)
        self.assertIn("--valuation-temperatures 25", result.stdout)
        self.assertIn("[hothaps]", result.stdout)
        self.assertIn("[summarize]", result.stdout)
        self.assertNotIn("[compare]", result.stdout)
        self.assertNotIn("[figure4]", result.stdout)
        self.assertNotIn("[figure5]", result.stdout)

    def test_placeholder_date_blocks_an_accidental_production_run(self) -> None:
        with tempfile.TemporaryDirectory() as data_root:
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--config",
                    str(EXAMPLE_CONFIG),
                    "--data-root",
                    data_root,
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("Replace YYYY-MM-DD", result.stderr)


if __name__ == "__main__":
    unittest.main()
