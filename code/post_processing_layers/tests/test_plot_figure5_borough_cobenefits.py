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

            with (output_dir / "figure5_borough_cobenefits_data.csv").open() as table:
                mapped = list(csv.DictReader(table))
            self.assertEqual(set(SCENARIOS), {row["scenario"] for row in mapped})
            self.assertEqual(33 * 9 * 3, len(mapped))


if __name__ == "__main__":
    unittest.main()
