"""Tests for population-weighted mortality allocation."""

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from prepare_mortality_population_weighted import allocate_deaths_by_population


class PopulationWeightedAllocationTests(unittest.TestCase):
    def test_preserves_zone_totals_and_uses_population_shares(self):
        population = np.array([[10.0, 30.0], [20.0, 40.0]])
        zones = np.array([[1, 2], [1, 2]], dtype=np.int32)
        deaths = np.array([0.0, 4.0, 12.0])

        allocated, population_totals, allocated_totals = (
            allocate_deaths_by_population(population, zones, deaths)
        )

        np.testing.assert_allclose(population_totals, [0.0, 30.0, 70.0])
        np.testing.assert_allclose(allocated_totals, deaths)
        np.testing.assert_allclose(allocated[:, 0], [4.0 / 3.0, 8.0 / 3.0])
        np.testing.assert_allclose(allocated[:, 1], [36.0 / 7.0, 48.0 / 7.0])

    def test_outside_zone_and_zero_population_are_not_allocated(self):
        population = np.array([[10.0, 0.0], [5.0, np.nan]])
        zones = np.array([[1, 1], [0, 0]], dtype=np.int32)
        deaths = np.array([0.0, 3.0])

        allocated, _, allocated_totals = allocate_deaths_by_population(
            population, zones, deaths
        )

        self.assertEqual(allocated[0, 0], 3.0)
        self.assertTrue(np.isnan(allocated[0, 1]))
        self.assertTrue(np.isnan(allocated[1, 0]))
        np.testing.assert_allclose(allocated_totals, deaths)

    def test_rejects_zone_without_positive_population(self):
        population = np.array([[10.0, 0.0]])
        zones = np.array([[1, 2]], dtype=np.int32)
        deaths = np.array([0.0, 3.0, 2.0])

        with self.assertRaisesRegex(ValueError, "No positive 2021 population"):
            allocate_deaths_by_population(population, zones, deaths)

    def test_rejects_negative_population(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            allocate_deaths_by_population(
                np.array([[10.0, -1.0]]),
                np.array([[1, 1]], dtype=np.int32),
                np.array([0.0, 3.0]),
            )


if __name__ == "__main__":
    unittest.main()
