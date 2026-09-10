#!/usr/bin/env python3
"""Extend Target30 by rank until its realized canopy matches Green30 exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from rasterio.features import rasterize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_grid(left: rasterio.DatasetReader, right: rasterio.DatasetReader) -> bool:
    return (
        left.crs == right.crs
        and left.transform == right.transform
        and left.shape == right.shape
    )


def _validate_ranked_prefix(
    full_path: Path,
    full_layer: str,
    current_path: Path,
    current_layer: str,
    rank_field: str,
    expected_count: int,
) -> list[int]:
    """Confirm the current candidate set is a sampled prefix of the full set."""
    current_count = int(pyogrio.read_info(current_path, layer=current_layer)["features"])
    if current_count != expected_count:
        raise ValueError(
            f"Current ranked layer has {current_count:,} features; expected "
            f"skip-features={expected_count:,}"
        )

    sample_indices = sorted(
        {
            0,
            expected_count // 4,
            expected_count // 2,
            3 * expected_count // 4,
            expected_count - 1,
        }
    )
    for index in sample_indices:
        full_row = pyogrio.read_dataframe(
            full_path,
            layer=full_layer,
            columns=[rank_field],
            skip_features=index,
            max_features=1,
        )
        current_row = pyogrio.read_dataframe(
            current_path,
            layer=current_layer,
            columns=[rank_field],
            skip_features=index,
            max_features=1,
        )
        if full_row.empty or current_row.empty:
            raise ValueError(f"Could not read ranked-prefix sample at feature {index:,}")
        ranks_match = int(full_row.iloc[0][rank_field]) == int(
            current_row.iloc[0][rank_field]
        )
        geometries_match = full_row.geometry.iloc[0].equals_exact(
            current_row.geometry.iloc[0], tolerance=0
        )
        if not (ranks_match and geometries_match):
            raise ValueError(
                "Current ranked candidates are not the expected prefix of the full layer "
                f"(sample mismatch at feature {index:,})"
            )

    last_current = pyogrio.read_dataframe(
        current_path,
        layer=current_layer,
        columns=[rank_field],
        skip_features=expected_count - 1,
        max_features=1,
        read_geometry=False,
    )
    first_next = pyogrio.read_dataframe(
        full_path,
        layer=full_layer,
        columns=[rank_field],
        skip_features=expected_count,
        max_features=1,
        read_geometry=False,
    )
    if int(first_next.iloc[0][rank_field]) < int(last_current.iloc[0][rank_field]):
        raise ValueError("The full candidate layer is not ascending at the prefix boundary")
    return sample_indices


def calibrate(args: argparse.Namespace) -> None:
    paths = {
        "baseline": Path(args.baseline).resolve(),
        "current_target": Path(args.current_target).resolve(),
        "ranked_candidates": Path(args.ranked_candidates).resolve(),
        "current_ranked_candidates": Path(args.current_ranked_candidates).resolve(),
    }
    output_path = Path(args.output).resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if (output_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError("Output exists; use --force only after reviewing it")

    with rasterio.open(paths["baseline"]) as baseline_source, rasterio.open(
        paths["current_target"]
    ) as current_source:
        if not _same_grid(baseline_source, current_source):
            raise ValueError("Baseline and current Target30 grids differ")
        baseline = baseline_source.read(1, masked=True)
        current = current_source.read(1, masked=True)
        if not np.array_equal(np.ma.getmaskarray(baseline), np.ma.getmaskarray(current)):
            raise ValueError("Baseline and current Target30 NoData masks differ")
        valid = ~np.ma.getmaskarray(baseline)
        baseline_values = np.asarray(baseline.data)
        scenario = np.asarray(current.data).copy()
        removed = valid & (baseline_values == args.tree_class) & (scenario != args.tree_class)
        if np.any(removed):
            raise ValueError("Current Target30 removes baseline tree pixels")
        existing_added = int(
            np.count_nonzero(valid & (baseline_values != args.tree_class) & (scenario == args.tree_class))
        )
        if existing_added >= args.target_added_pixels:
            raise ValueError("Current Target30 already meets or exceeds the requested budget")
        profile = baseline_source.profile.copy()
        transform = baseline_source.transform

    info = pyogrio.read_info(paths["ranked_candidates"], layer=args.layer)
    feature_count = int(info["features"])
    if args.skip_features >= feature_count:
        raise ValueError("skip-features is outside the ranked candidate layer")
    prefix_samples = _validate_ranked_prefix(
        paths["ranked_candidates"],
        args.layer,
        paths["current_ranked_candidates"],
        args.current_layer,
        args.rank_field,
        args.skip_features,
    )

    offset = args.skip_features
    added = existing_added
    last_rank: int | None = None
    cutoff_rank: int | None = None
    features_read = 0

    while added < args.target_added_pixels and offset < feature_count:
        frame = pyogrio.read_dataframe(
            paths["ranked_candidates"],
            layer=args.layer,
            columns=[args.rank_field],
            skip_features=offset,
            max_features=args.batch_size,
        )
        if frame.empty:
            break
        ranks = frame[args.rank_field].to_numpy(dtype=np.int64)
        if np.any(np.diff(ranks) < 0):
            raise ValueError("Candidate features are not stored in ascending rank order")
        if last_rank is not None and ranks[0] < last_rank:
            raise ValueError("Candidate rank decreases between batches")
        last_rank = int(ranks[-1])

        # Reverse order so overlapping polygons retain their lowest rank.
        rank_surface = rasterize(
            ((geometry, int(rank)) for geometry, rank in zip(frame.geometry[::-1], ranks[::-1])),
            out_shape=scenario.shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="int32",
        )
        candidates = (
            valid
            & (baseline_values != args.tree_class)
            & (scenario != args.tree_class)
            & (rank_surface > 0)
        )
        rows, cols = np.nonzero(candidates)
        candidate_ranks = rank_surface[rows, cols]
        order = np.argsort(candidate_ranks, kind="stable")
        remaining = args.target_added_pixels - added
        take = min(remaining, len(order))
        chosen = order[:take]
        scenario[rows[chosen], cols[chosen]] = args.tree_class
        if take:
            cutoff_rank = int(candidate_ranks[chosen[-1]])
        added += take
        features_read += len(frame)
        offset += len(frame)
        print(f"Processed {offset:,} ranked features; realized additions: {added:,}")

    if added != args.target_added_pixels:
        raise ValueError(f"Only reached {added:,} of {args.target_added_pixels:,} pixels")

    profile.update(compress="lzw")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as target:
        target.write(scenario, 1)

    manifest = {
        "schema_version": 1,
        "method": "extend current Target30 with next-ranked candidate pixels",
        "tree_class": args.tree_class,
        "initial_added_tree_pixels": existing_added,
        "target_added_tree_pixels": args.target_added_pixels,
        "additional_tree_pixels": args.target_added_pixels - existing_added,
        "ranked_features_skipped": args.skip_features,
        "ranked_features_read": features_read,
        "rank_cutoff": cutoff_rank,
        "candidate_layer": args.layer,
        "current_candidate_layer": args.current_layer,
        "rank_field": args.rank_field,
        "ranked_prefix_sample_indices": prefix_samples,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyogrio": pyogrio.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created calibrated Target30: {output_path}")
    print(f"Manifest: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current-target", required=True)
    parser.add_argument("--ranked-candidates", required=True)
    parser.add_argument("--current-ranked-candidates", required=True)
    parser.add_argument("--layer", default="tree_equity_scenario730")
    parser.add_argument("--current-layer", default="tree_equity_scenario730v3")
    parser.add_argument("--rank-field", default="rank")
    parser.add_argument("--skip-features", type=int, default=1_819_000)
    parser.add_argument("--target-added-pixels", type=int, default=930_000)
    parser.add_argument("--tree-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    calibrate(build_parser().parse_args())
