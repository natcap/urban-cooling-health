#!/usr/bin/env python3
"""Convert the project's Nomis mortality TSV into the revised long CSV.

The output is an intermediate project file, not a separately published data
source. It contains one 2021 registered-death count for every London borough
and each cause used by the health model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CAUSE_PREFIXES = {
    "all_cause": ("A00", "R99", "U00", "Y89"),
    "mental_disorder": ("F00", "F99"),
    "cardiovascular": ("I00", "I99"),
    "respiratory": ("J00", "J99"),
    "self_harm": ("X60", "X84"),
}


def _normalise_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _find_column(columns: list[str], *accepted: str) -> str:
    lookup = {_normalise_column(column): column for column in columns}
    for candidate in accepted:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        f"Nomis table has no column matching {accepted}; columns={columns}"
    )


def _cause_from_label(label: object) -> str | None:
    """Map the five selected Nomis ICD-10 groups to stable model names."""
    compact = re.sub(r"[^A-Z0-9]+", "", str(label).upper())
    for cause, prefixes in CAUSE_PREFIXES.items():
        if all(prefix in compact for prefix in prefixes):
            return cause
    return None


def convert_nomis(frame: pd.DataFrame, year: int = 2021) -> pd.DataFrame:
    """Return validated London-borough counts in the four-column schema."""
    # The archived Nomis exports contain each selected record twice. Remove
    # only byte-equivalent rows; conflicting duplicates are rejected below.
    frame = frame.drop_duplicates().copy()
    columns = list(frame.columns)
    geography = _find_column(columns, "geogcode", "geography code")
    date = _find_column(columns, "date", "year")
    cause_label = _find_column(columns, "cause of death", "cause of deat")
    value = _find_column(columns, "value")

    work = frame[[geography, date, cause_label, value]].copy()
    work[geography] = work[geography].astype(str).str.strip()
    work[date] = pd.to_numeric(work[date], errors="coerce")
    # E09000001-E09000033 are the 33 London local-authority codes.
    work = work.loc[
        (work[date] == year)
        & work[geography].str.fullmatch(r"E09\d{6}", na=False)
    ].copy()
    if work.empty:
        raise ValueError(f"No London-borough rows were found for {year}")

    work["cause"] = work[cause_label].map(_cause_from_label)
    work = work.loc[work["cause"].notna()].copy()
    work["deaths"] = pd.to_numeric(
        work[value].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    if work["deaths"].isna().any():
        labels = sorted(work.loc[work["deaths"].isna(), cause_label].astype(str).unique())
        raise ValueError(f"Selected Nomis counts contain non-numeric values: {labels}")

    result = work.rename(columns={geography: "borough_id"})[
        ["borough_id", "cause", "deaths"]
    ]
    result.insert(1, "year", year)
    duplicated = result.duplicated(["borough_id", "cause"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Nomis selection has multiple rows per borough/cause. Re-download "
            "the table with one sex total and one all-ages total, or inspect: "
            + ", ".join(
                sorted(result.loc[duplicated, ["borough_id", "cause"]]
                       .astype(str).agg("/".join, axis=1).unique())
            )
        )

    expected_causes = set(CAUSE_PREFIXES)
    found_causes = set(result["cause"])
    if found_causes != expected_causes:
        raise ValueError(
            "Nomis table does not contain the five required cause groups; "
            f"missing={sorted(expected_causes - found_causes)}"
        )
    counts = result.groupby("cause")["borough_id"].nunique()
    if len(set(counts)) != 1 or int(counts.iloc[0]) != 33:
        raise ValueError(f"Expected 33 boroughs for every cause; found {counts.to_dict()}")
    if (result["deaths"] < 0).any():
        raise ValueError("Nomis death counts must be non-negative")

    return result.sort_values(["borough_id", "cause"]).reset_index(drop=True)


def main(args: argparse.Namespace) -> None:
    source = Path(args.nomis_data_tsv).resolve()
    output = Path(args.output_csv).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Nomis TSV does not exist: {source}")
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; use --force after review")

    result = convert_nomis(pd.read_csv(source, sep="\t", dtype=str), args.year)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Created {output} with {len(result)} rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nomis-data-tsv", required=True, help="Nomis *_data.tsv export")
    parser.add_argument("--output-csv", required=True, help="Output mortality_2021_long.csv")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--force", action="store_true", help="Overwrite a reviewed output")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
