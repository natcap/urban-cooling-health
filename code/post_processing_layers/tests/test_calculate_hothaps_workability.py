"""Unit checks for the project-specific Hothaps workability equation."""

import unittest

import numpy

from code.post_processing_layers.calculate_hothaps_workability import (
    DEFAULT_ALPHA1,
    DEFAULT_ALPHA2,
    _workability,
)


class HothapsEquationTests(unittest.TestCase):
    def test_reference_points_and_direction(self):
        wbgt = numpy.array([0.0, DEFAULT_ALPHA1, 40.0], dtype=numpy.float32)
        result = _workability(wbgt, DEFAULT_ALPHA1, DEFAULT_ALPHA2)

        numpy.testing.assert_allclose(result[:2], [1.0, 0.55], rtol=1e-6)
        self.assertTrue(numpy.all(numpy.diff(result) < 0))
        self.assertTrue(numpy.all((result >= 0.1) & (result <= 1.0)))


if __name__ == "__main__":
    unittest.main()
