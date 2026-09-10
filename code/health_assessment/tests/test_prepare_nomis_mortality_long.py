"""Tests for conversion of the Nomis mortality table."""

import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from prepare_nomis_mortality_long import CAUSE_PREFIXES, convert_nomis


LABELS = {
    "all_cause": "A00-R99, U00-Y89 All causes, all ages",
    "mental_disorder": "F00-F99 V Mental and behavioural disorders",
    "cardiovascular": "I00-I99 IX Diseases of the circulatory system",
    "respiratory": "J00-J99 X Diseases of the respiratory system",
    "self_harm": "X60-X84 Intentional self-harm",
}


def sample_frame() -> pd.DataFrame:
    rows = []
    for borough_number in range(1, 34):
        borough = f"E09{borough_number:06d}"
        for cause, label in LABELS.items():
            rows.append(
                {
                    "GEOGCODE": borough,
                    "DATE": "2021",
                    "CAUSE OF DEATH": label,
                    "VALUE": str(borough_number),
                }
            )
    return pd.DataFrame(rows)


class NomisConversionTests(unittest.TestCase):
    def test_creates_expected_long_table(self):
        result = convert_nomis(sample_frame())
        self.assertEqual(len(result), 33 * len(CAUSE_PREFIXES))
        self.assertEqual(set(result["cause"]), set(CAUSE_PREFIXES))
        self.assertTrue((result["year"] == 2021).all())

    def test_ignores_other_years_and_geographies(self):
        source = sample_frame()
        extra = source.iloc[[0]].copy()
        extra["DATE"] = "2020"
        region = source.iloc[[0]].copy()
        region["GEOGCODE"] = "E12000007"
        result = convert_nomis(pd.concat([source, extra, region], ignore_index=True))
        self.assertEqual(len(result), 165)

    def test_ignores_exact_duplicate_export_rows(self):
        source = sample_frame()
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        result = convert_nomis(source)
        self.assertEqual(len(result), 165)

    def test_rejects_conflicting_duplicate_rows(self):
        source = sample_frame()
        duplicate = source.iloc[[0]].copy()
        duplicate["VALUE"] = "999"
        source = pd.concat([source, duplicate], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "multiple rows"):
            convert_nomis(source)

    def test_rejects_missing_cause(self):
        source = sample_frame()
        source = source.loc[source["CAUSE OF DEATH"] != LABELS["self_harm"]]
        with self.assertRaisesRegex(ValueError, "five required cause"):
            convert_nomis(source)


if __name__ == "__main__":
    unittest.main()
