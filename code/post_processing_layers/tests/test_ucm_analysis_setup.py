"""Tests for the shared R notebook setup."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SETUP_SCRIPT = Path(__file__).resolve().parents[2] / "ucm_analysis_setup.R"
REPO_ROOT = SETUP_SCRIPT.parents[1]


@unittest.skipUnless(shutil.which("Rscript"), "Rscript is required")
class UcmAnalysisSetupTest(unittest.TestCase):
    def test_resolves_paths_and_exports_legacy_names(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            data_root = Path(workspace)
            (data_root / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs").mkdir(
                parents=True
            )
            (data_root / "2_postprocess_intermediate/UCM_official_runs").mkdir(
                parents=True
            )
            environment = os.environ.copy()
            environment["URBAN_COOLING_DATA_ROOT"] = str(data_root)
            environment["URBAN_COOLING_RUN_DATE"] = "20260913"
            result = subprocess.run(
                [
                    "Rscript",
                    "-e",
                    (
                        "source(commandArgs(TRUE)[1]); "
                        "x <- ucm_analysis_setup(export_legacy_names=TRUE); "
                        "stopifnot(dir.g == x$paths$data_root, "
                        "year_baseline == 2021L, ymd == '20260913', "
                        "exists('scenario_colors')); cat('SETUP_OK')"
                    ),
                    str(SETUP_SCRIPT),
                ],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("SETUP_OK", result.stdout)

    def test_missing_data_root_has_actionable_error(self) -> None:
        environment = os.environ.copy()
        environment.pop("URBAN_COOLING_DATA_ROOT", None)
        # Isolate the test from a developer's user-level ~/.Renviron.
        environment["R_ENVIRON_USER"] = "/dev/null"
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                "source(commandArgs(TRUE)[1]); ucm_analysis_setup()",
                str(SETUP_SCRIPT),
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("Set URBAN_COOLING_DATA_ROOT", result.stderr)


if __name__ == "__main__":
    unittest.main()
