"""Integration test for the nine-scenario Figure 5 builder."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "plot_figure5_borough_cobenefits.R"
SCENARIOS = (
    "allbuilt", "treerisk", "treeopp",
    "green10", "target10", "green20", "target20", "green30", "target30",
)


@unittest.skipUnless(shutil.which("Rscript"), "Rscript is required")
class Figure5BuilderTest(unittest.TestCase):
    """Confirm that all manuscript scenarios reach the mapped-data artifact."""

    def test_builds_all_nine_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            borough_path = root / "borough.csv"
            health_root = root / "health"
            vector_path = root / "boroughs.shp"
            output_dir = root / "figure5"

            rows = []
            for scenario_index, scenario in enumerate(("baseline", *SCENARIOS)):
                for borough_index in range(33):
                    rows.append({
                        "scenario": scenario,
                        "temperature_c": 25,
                        "borough": f"Borough {borough_index}",
                        "borough_code": f"E{borough_index:03d}",
                        "energy_savings": 1_000_000 + scenario_index * 1_000,
                        "workability_fraction": 0.8 + scenario_index / 1000,
                        "capacity_method": "daily_tmax_sensitivity",
                    })
            with borough_path.open("w", newline="") as table:
                writer = csv.DictWriter(table, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            fixture_code = f"""
            suppressPackageStartupMessages({{library(sf); library(terra)}})
            bbox <- st_bbox(c(xmin=0, ymin=0, xmax=11, ymax=3), crs=st_crs(27700))
            geometry <- st_make_grid(st_as_sfc(bbox), n=c(11, 3))
            boroughs <- st_sf(
              NAME=sprintf('Borough %d', 0:32),
              GSS_CODE=sprintf('E%03d', 0:32), geometry=geometry
            )
            st_write(boroughs, {str(vector_path)!r}, quiet=TRUE)
            template <- rast(ncols=110, nrows=30, xmin=0, xmax=11,
                             ymin=0, ymax=3, crs='EPSG:27700')
            values(template) <- -0.01
            for (scenario in c({', '.join(repr(s) for s in SCENARIOS)})) {{
              folder <- file.path({str(health_root)!r}, paste0(scenario, '_25c'))
              dir.create(folder, recursive=TRUE)
              writeRaster(template, file.path(folder, 'Excess_all_cause.tif'),
                          overwrite=TRUE)
            }}
            """
            fixture = subprocess.run(
                ["Rscript", "-e", fixture_code],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, fixture.returncode, fixture.stderr)

            result = subprocess.run(
                [
                    "Rscript", str(SCRIPT),
                    "--borough-summary", str(borough_path),
                    "--health-root", str(health_root),
                    "--borough-vector", str(vector_path),
                    "--output-dir", str(output_dir),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

            with (output_dir / "figure5_borough_cobenefits_20261110_v3_data.csv").open() as table:
                mapped = list(csv.DictReader(table))
            self.assertEqual(set(SCENARIOS), {row["scenario"] for row in mapped})
            self.assertEqual(33 * 9 * 3, len(mapped))

            with (health_root / "borough_results.csv").open("w", newline="") as table:
                writer = csv.DictWriter(table, fieldnames=("scenario", "cause", "reconstruction", "ssp", "rr_variant", "borough_id", "deaths_averted"))
                writer.writeheader()
                for scenario in SCENARIOS:
                    for i in range(33):
                        writer.writerow(dict(scenario=scenario, cause="all_cause", reconstruction="worldclim_aligned", ssp="ssp370", rr_variant="central", borough_id=f"E{i:03d}", deaths_averted=-(i+1)/10))
            command = ["Rscript", str(SCRIPT), "--borough-summary", str(borough_path),
                       "--seasonal-health-root", str(health_root), "--borough-vector", str(vector_path),
                       "--output-dir", str(root / "candidate"), "--figure-version", "test_v4"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            with (root / "candidate/figure5_borough_cobenefits_test_v4_data.csv").open() as f:
                candidate = list(csv.DictReader(f))
            health = [r for r in candidate if r["metric"] == "Heat-related deaths averted"]
            self.assertEqual(len(health), 33 * 9)
            self.assertTrue(all(float(r["value"]) < 0 for r in health))
            self.assertTrue(all(r["unit"] == "Deaths/modelled summer" for r in health))
            capacity = [r for r in candidate if r["metric"] == "Heavy-work capacity gain"]
            self.assertTrue(all("summer daily mean" in r["unit"] for r in capacity))
            self.assertNotEqual(0, subprocess.run(command, capture_output=True).returncode)


if __name__ == "__main__":
    unittest.main()
