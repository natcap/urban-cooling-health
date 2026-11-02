"""Small synthetic checks for the scenario raster grid gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_green_target_scenarios import _same_grid


class ScenarioGridTests(unittest.TestCase):
    def test_rejects_subpixel_shift_at_bng_coordinates(self) -> None:
        profile = {
            "driver": "GTiff",
            "height": 2,
            "width": 2,
            "count": 1,
            "dtype": np.uint8,
            "crs": "EPSG:27700",
        }
        with MemoryFile() as first, MemoryFile() as second, MemoryFile() as shifted:
            with first.open(**profile, transform=from_origin(500000, 200000, 10, 10)) as reference:
                with second.open(**profile, transform=from_origin(500000, 200000, 10, 10)) as same:
                    self.assertTrue(_same_grid(reference, same))
                with shifted.open(**profile, transform=from_origin(500000.1, 200000, 10, 10)) as offset:
                    self.assertFalse(_same_grid(reference, offset))


if __name__ == "__main__":
    unittest.main()
