"""Integration test for the production Figure 4 builder."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "plot_figure4_citywide.R"
SCENARIOS = (
    "allbuilt", "treerisk", "treeopp",
    "green10", "target10", "green20", "target20", "green30", "target30",
)


@unittest.skipUnless(shutil.which("Rscript"), "Rscript is required")
class Figure4BuilderTest(unittest.TestCase):
    """Confirm that reviewed inputs produce the documented figure artifacts."""

    def test_builds_citywide_and_borough_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            city_path = root / "city.csv"
            borough_path = root / "borough.csv"
            health_root = root / "health"
            output_dir = root / "figure4"

            city_rows = []
            borough_rows = []
            for scenario_index, scenario in enumerate(("baseline", *SCENARIOS)):
                city_rows.append({
                    "scenario": scenario,
                    "temperature_c": 25,
                    "energy_change_vs_baseline": scenario_index * 1_000_000,
                    "productivity_change_vs_baseline_pp": scenario_index / 10,
                })
                for borough_index in range(33):
                    borough_rows.append({
                        "scenario": scenario,
                        "temperature_c": 25,
                        "borough": f"Borough {borough_index}",
                        "borough_code": f"E{borough_index:03d}",
                        "energy_savings": 10_000_000 + scenario_index * 100_000 + borough_index,
                        "workability_fraction": 0.8 + scenario_index / 1000,
                    })
                if scenario != "baseline":
                    health_dir = health_root / f"{scenario}_25c"
                    health_dir.mkdir(parents=True)
                    with (health_dir / "city_totals_deterministic.csv").open("w", newline="") as table:
                        writer = csv.DictWriter(table, fieldnames=("cause", "deaths_averted"))
                        writer.writeheader()
                        writer.writerow({"cause": "all_cause", "deaths_averted": scenario_index * 10})
                    with (health_dir / "city_total_draws_by_cause.csv").open("w", newline="") as table:
                        writer = csv.DictWriter(table, fieldnames=("draw", "all_cause"))
                        writer.writeheader()
                        for draw in range(20):
                            writer.writerow({"draw": draw, "all_cause": -(scenario_index * 10 + draw / 10)})

            for path, rows in ((city_path, city_rows), (borough_path, borough_rows)):
                with path.open("w", newline="") as table:
                    writer = csv.DictWriter(table, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            result = subprocess.run(
                [
                    "Rscript", str(SCRIPT),
                    "--citywide-summary", str(city_path),
                    "--borough-summary", str(borough_path),
                    "--health-root", str(health_root),
                    "--output-dir", str(output_dir),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            for name in (
                "figure4_citywide_cobenefits.png",
                "figure4_citywide_cobenefits.pdf",
                "figure4_citywide_cobenefits.svg",
                "figure4_citywide_cobenefits_data.csv",
                "figure4_citywide_cobenefits_manifest.json",
                "figure4_borough_sensitivity.png",
                "figure4_borough_sensitivity_data.csv",
                "extended_data_citywide_vs_borough.png",
                "extended_data_citywide_vs_borough_data.csv",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            # The manuscript figure directly labels each estimand with its
            # documented panel-specific precision.
            svg = (output_dir / "figure4_citywide_cobenefits.svg").read_text()
            self.assertIn("+1.0", svg)
            self.assertIn("+0.10", svg)
            self.assertIn("+90", svg)


if __name__ == "__main__":
    unittest.main()
