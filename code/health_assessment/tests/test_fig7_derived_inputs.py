"""Integrity checks for the small, tracked Figure 7 production inputs."""

from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DERIVED = REPOSITORY_ROOT / "data" / "derived"
EXPECTED_LSOA_COUNT = 4_835
EXPECTED_DRAWS = 2_000


def _rows(filename: str) -> list[dict[str, str]]:
    with (DERIVED / filename).open(newline="", encoding="utf-8") as table:
        return list(csv.DictReader(table))


class Figure7DerivedInputTests(unittest.TestCase):
    def test_population_is_complete_and_officially_keyed(self) -> None:
        rows = _rows("lsoa_population_2021_by_lsoa11cd.csv")
        codes = [row["LSOA11CD"] for row in rows]
        populations = [float(row["population_2021"]) for row in rows]

        self.assertEqual(len(rows), EXPECTED_LSOA_COUNT)
        self.assertEqual(len(set(codes)), EXPECTED_LSOA_COUNT)
        self.assertTrue(all(code.startswith("E01") for code in codes))
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in populations))
        crosswalk_codes = {
            row["LSOA11CD"]
            for row in _rows("svi_lsoa11_crosswalk_nodata_harmonized.csv")
        }
        self.assertEqual(set(codes), crosswalk_codes)

    def test_health_rows_cover_every_code_once_per_case(self) -> None:
        population_codes = {
            row["LSOA11CD"]
            for row in _rows("lsoa_population_2021_by_lsoa11cd.csv")
        }
        health_rows = _rows(
            "health_lsoa_invest3202_population_weighted_2021_"
            "nodata_harmonized.csv"
        )
        cases: dict[tuple[str, str], set[str]] = {}
        seen: set[tuple[str, str, str]] = set()
        for row in health_rows:
            case = (row["scenario"], row["cause"])
            key = (*case, row["LSOA11CD"])
            self.assertNotIn(key, seen)
            seen.add(key)
            cases.setdefault(case, set()).add(row["LSOA11CD"])

        self.assertEqual(len(cases), 20)  # Four scenario-temperature cases x five causes.
        for codes in cases.values():
            self.assertEqual(codes, population_codes)

    def test_monte_carlo_draws_are_pairable(self) -> None:
        green = _rows("green30_nodata_harmonized_city_total_draws_by_cause.csv")
        target = _rows("target30_nodata_harmonized_city_total_draws_by_cause.csv")
        green_draws = [row["draw"] for row in green]
        target_draws = [row["draw"] for row in target]

        self.assertEqual(len(green_draws), EXPECTED_DRAWS)
        self.assertEqual(len(set(green_draws)), EXPECTED_DRAWS)
        self.assertEqual(green_draws, target_draws)


if __name__ == "__main__":
    unittest.main()
