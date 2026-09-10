"""Unit tests for the strict version 2 mortality model."""

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from health_modeling_v2 import (
    Cause,
    attributable_fraction,
    cause_beta_draws,
    monte_carlo_totals_from_moments,
)


class HealthModelV2Tests(unittest.TestCase):
    def test_zero_temperature_difference_has_zero_effect(self):
        result = attributable_fraction(np.log(1.02), np.zeros(3))
        np.testing.assert_allclose(result, 0.0)

    def test_one_degree_matches_relative_risk_definition(self):
        rr = 1.02
        result = attributable_fraction(np.log(rr), np.array([1.0]))
        np.testing.assert_allclose(result, [(rr - 1.0) / rr])

    def test_cooling_produces_negative_excess_fraction(self):
        result = attributable_fraction(np.log(1.02), np.array([-1.0]))
        self.assertLess(result[0], 0)

    def test_draws_are_reproducible_and_cause_specific(self):
        all_cause = Cause("all_cause", 1.02, 1.014, 1.026, Path("unused"))
        respiratory = Cause("respiratory", 1.036, 1.0318, 1.0402, Path("unused"))
        first = cause_beta_draws(all_cause, 20, 20260908)
        second = cause_beta_draws(all_cause, 20, 20260908)
        other = cause_beta_draws(respiratory, 20, 20260908)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_moment_method_matches_direct_monte_carlo_totals(self):
        delta_t = np.array([-2.1, -0.7, -0.05, 0.2])
        deaths = np.array([8.0, 12.0, 3.0, 1.0])
        beta_draws = np.array([0.003, 0.02, 0.045])
        moments = np.array([
            np.sum(deaths * delta_t**order) for order in range(1, 13)
        ])

        fast = monte_carlo_totals_from_moments(beta_draws, moments)
        direct = np.array([
            np.sum(deaths * attributable_fraction(beta, delta_t))
            for beta in beta_draws
        ])

        np.testing.assert_allclose(fast, direct, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
