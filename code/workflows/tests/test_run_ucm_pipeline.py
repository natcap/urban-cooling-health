"""Tests for command construction in the unified UCM pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
RUNNER = WORKFLOW_DIR / "run_ucm_pipeline.py"
EXAMPLE_CONFIG = WORKFLOW_DIR / "ucm_pipeline.example.json"
REPOSITORY_ROOT = WORKFLOW_DIR.parents[1]
HEALTH_CONFIG = (
    REPOSITORY_ROOT
    / "code/health_assessment/health-analysis-revised-equal-area.example.json"
)
CANONICAL_SCENARIOS = {
    "baseline", "allbuilt", "treerisk", "treeopp",
    "green10", "green20", "green30", "target10", "target20", "target30",
}


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

    def test_revised_configs_share_portable_root_and_scenario_keys(self) -> None:
        ucm = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        health = json.loads(HEALTH_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(set(ucm["scenarios"]), CANONICAL_SCENARIOS)
        self.assertEqual(health["data_root"], "${URBAN_COOLING_DATA_ROOT}")
        self.assertEqual(health["design"]["population_year"], 2021)
        self.assertEqual(health["design"]["mortality_year"], 2021)
        self.assertEqual(set(ucm["temperatures_c"]), {25, 28})
        self.assertEqual(set(ucm["valuation_temperatures_c"]), {25})
        self.assertEqual(
            set(health["scenarios"]),
            {
                f"{name}_{temperature}c"
                for name in CANONICAL_SCENARIOS - {"baseline"}
                for temperature in (25, 28)
            },
        )
        for relative_path in (
            *ucm["scenarios"].values(),
            ucm["borough_vector"],
            health["population_2021"],
            *(cause["mortality_raster"] for cause in health["causes"]),
        ):
            self.assertFalse(Path(relative_path).is_absolute(), relative_path)
            self.assertNotIn(":/", relative_path)
            self.assertNotIn("/Users/", relative_path)

    @unittest.skipUnless(shutil.which("Rscript"), "Rscript is required")
    def test_canonical_scenarios_have_matching_r_labels_and_colours(self) -> None:
        result = subprocess.run(
            [
                "Rscript", "-e",
                (
                    "source(commandArgs(TRUE)[1]); "
                    "keys <- strsplit(commandArgs(TRUE)[2], ',', fixed=TRUE)[[1]]; "
                    "stopifnot(all(keys %in% names(scenario_labels)), "
                    "all(keys %in% names(scenario_colors))); "
                    "cat(paste(scenario_labels[keys], collapse='|'))"
                ),
                str(REPOSITORY_ROOT / "code/func_colors.R"),
                ",".join(sorted(CANONICAL_SCENARIOS)),
            ],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(
            result.stdout,
            "AllBuilt|Baseline|Green10|Green20|Green30|Target10|Target20|"
            "Target30|TreeOpp|TreeRisk",
        )

    def test_release_inventory_excludes_raw_rasters_and_local_secrets(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPOSITORY_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        forbidden = [
            path for path in tracked
            if (
                path.startswith("data/")
                and path.lower().endswith((".tif", ".tiff"))
            ) or path in {"code/google_api_key.env", "code/deploy_on_shinyapps.io.R"}
        ]
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
