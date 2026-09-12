#!/usr/bin/env python3
"""Run the production UCM and post-processing workflow from one config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("ucm", "hothaps", "summarize", "compare", "figure4", "figure5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _git_record(repo_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _run(stage: str, command: list[str], dry_run: bool) -> None:
    print(f"[{stage}] {shlex.join(command)}", flush=True)
    if not dry_run:
        environment = os.environ.copy()
        environment.setdefault(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "urban-cooling-mpl")
        )
        Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--stages", nargs="+", choices=STAGES)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        parser.error("The pipeline configuration must use schema_version 1")

    data_root_raw = args.data_root or os.environ.get("URBAN_COOLING_DATA_ROOT")
    if not data_root_raw:
        parser.error("Set --data-root or URBAN_COOLING_DATA_ROOT")
    data_root = Path(data_root_raw).expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"Data root not found: {data_root}")

    repo_root = Path(__file__).resolve().parents[2]
    input_root = data_root / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs"
    output_root = _resolve(data_root, config["output_root"])
    if "YYYY-MM-DD" in config["output_root"] and not (args.dry_run or args.validate_only):
        parser.error("Replace YYYY-MM-DD in output_root before running the pipeline")
    borough_path = _resolve(data_root, config["borough_vector"])
    scenarios = config.get("scenarios", {})
    if not scenarios:
        parser.error("At least one scenario is required")
    invalid_scenario_names = [
        name for name in scenarios if not re.fullmatch(r"[a-z][a-z0-9_]*", name)
    ]
    if invalid_scenario_names:
        parser.error(
            "Scenario names must use lowercase letters, numbers and underscores: "
            + ", ".join(invalid_scenario_names)
        )
    temperatures = [float(value) for value in config.get("temperatures_c", [])]
    valuation_temperatures = [
        float(value) for value in config.get("valuation_temperatures_c", [])
    ]
    if not temperatures or not set(valuation_temperatures) <= set(temperatures):
        parser.error("valuation_temperatures_c must be a subset of temperatures_c")

    selected_stages = args.stages or list(STAGES)
    compare_config = config.get("historical_comparison", {})
    if not compare_config.get("enabled", False):
        selected_stages = [stage for stage in selected_stages if stage != "compare"]
    figure4_config = config.get("figure4", {})
    if not figure4_config.get("enabled", False):
        selected_stages = [stage for stage in selected_stages if stage != "figure4"]
    figure5_config = config.get("figure5", {})
    if not figure5_config.get("enabled", False):
        selected_stages = [stage for stage in selected_stages if stage != "figure5"]
    if args.validate_only:
        selected_stages = ["ucm"]

    ucm_runner = repo_root / "code/Urban_Cooling_Modeling_Runs/run_ucm_scenarios.py"
    hothaps_runner = repo_root / "code/post_processing_layers/calculate_hothaps_workability.py"
    summary_runner = repo_root / "code/post_processing_layers/summarize_ucm_valuations.py"
    comparison_runner = repo_root / "code/post_processing_layers/compare_ucm_valuation_versions.py"
    figure4_runner = repo_root / "code/post_processing_layers/plot_figure4_citywide.R"
    figure5_runner = repo_root / "code/post_processing_layers/plot_figure5_borough_cobenefits.R"

    ucm_command = [
        sys.executable,
        str(ucm_runner),
        str(data_root),
        "--scenarios",
        *scenarios,
        "--temperatures",
        *(f"{value:g}" for value in temperatures),
        "--uhi-max",
        str(config.get("uhi_max_c", 5)),
        "--humidity",
        str(config.get("relative_humidity_pct", 45)),
        "--output-root",
        str(output_root),
        "--repo-root",
        str(repo_root),
    ]
    if valuation_temperatures:
        ucm_command.extend([
            "--valuation-temperatures",
            *(f"{value:g}" for value in valuation_temperatures),
        ])
    for name, relative_path in scenarios.items():
        ucm_command.extend([
            "--scenario-lulc",
            f"{name}={_resolve(input_root, relative_path)}",
        ])
    if args.validate_only:
        ucm_command.append("--validate-only")
    elif args.resume:
        ucm_command.append("--resume")

    commands = {
        "ucm": ucm_command,
        "hothaps": [
            sys.executable,
            str(hothaps_runner),
            str(output_root),
            "--scenarios",
            *scenarios,
            *(["--resume"] if args.resume else []),
        ],
        "summarize": [
            sys.executable,
            str(summary_runner),
            str(output_root),
            str(borough_path),
        ],
    }
    if compare_config.get("enabled", False):
        commands["compare"] = [
            sys.executable,
            str(comparison_runner),
            str(_resolve(data_root, compare_config["original_ucm_root"])),
            str(output_root / "summary/citywide_energy_productivity_summary.csv"),
        ]
    if figure4_config.get("enabled", False):
        if not figure4_config.get("health_output_root"):
            parser.error("figure4.health_output_root is required when Figure 4 is enabled")
        commands["figure4"] = [
            "Rscript",
            str(figure4_runner),
            "--citywide-summary",
            str(output_root / "summary/citywide_energy_productivity_summary.csv"),
            "--borough-summary",
            str(output_root / "summary/borough_energy_productivity_summary.csv"),
            "--health-root",
            str(_resolve(data_root, figure4_config["health_output_root"])),
            "--temperature",
            str(figure4_config.get("temperature_c", 25)),
            "--price-basis",
            str(figure4_config.get(
                "price_basis", "late-2025 input-price assumptions"
            )),
            "--output-dir",
            str(output_root / "summary/figure4"),
        ]
    if figure5_config.get("enabled", False):
        health_output_root = figure5_config.get(
            "health_output_root", figure4_config.get("health_output_root")
        )
        if not health_output_root:
            parser.error(
                "figure5.health_output_root is required when Figure 5 is enabled"
            )
        commands["figure5"] = [
            "Rscript",
            str(figure5_runner),
            "--borough-summary",
            str(output_root / "summary/borough_energy_productivity_summary.csv"),
            "--health-root",
            str(_resolve(data_root, health_output_root)),
            "--borough-vector",
            str(borough_path),
            "--temperature",
            str(figure5_config.get("temperature_c", 25)),
            "--output-dir",
            str(output_root / "summary/figure5"),
        ]

    if args.dry_run or args.validate_only:
        for stage in selected_stages:
            _run(stage, commands[stage], args.dry_run)
        return 0

    current_git = _git_record(repo_root)
    if current_git.get("dirty"):
        parser.error(
            "Commit or stash repository changes before a production pipeline run"
        )
    state_path = output_root / "pipeline_run_manifest.json"
    config_sha256 = _sha256(config_path)
    if state_path.exists():
        if not args.resume:
            parser.error(f"Pipeline manifest already exists; use --resume: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_sha256:
            parser.error("Cannot resume: pipeline configuration checksum changed")
        if state.get("git") != current_git:
            parser.error("Cannot resume: repository commit or working-tree state changed")
    else:
        state = {
            "schema_version": 1,
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "configuration": config,
            "data_root": str(data_root),
            "output_root": str(output_root),
            "python": sys.version,
            "git": current_git,
            "stages": {},
        }

    for stage in selected_stages:
        state["stages"][stage] = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "command": commands[stage],
        }
        _write_state(state_path, state)
        try:
            _run(stage, commands[stage], dry_run=False)
        except Exception:
            state["stages"][stage]["status"] = "failed"
            state["stages"][stage]["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_state(state_path, state)
            raise
        state["stages"][stage]["status"] = "complete"
        state["stages"][stage]["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_state(state_path, state)
    print(f"Pipeline complete: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
