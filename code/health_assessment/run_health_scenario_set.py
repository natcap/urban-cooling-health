#!/usr/bin/env python3
"""Validate or run every enabled scenario in a health-model configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--n-draws", type=int)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    enabled = [
        name
        for name, definition in config.get("scenarios", {}).items()
        if definition.get("enabled", False)
    ]
    scenarios = args.scenarios or enabled
    unknown = set(scenarios) - set(config.get("scenarios", {}))
    if unknown:
        parser.error(f"Unknown scenario(s): {sorted(unknown)}")

    model = Path(__file__).with_name("health_modeling_v2.py")
    for index, scenario in enumerate(scenarios, start=1):
        command = [
            sys.executable,
            str(model),
            "--config",
            str(config_path),
            "--scenario",
            scenario,
        ]
        if args.validate_only:
            command.append("--validate-only")
        if args.n_draws is not None:
            command.extend(["--n-draws", str(args.n_draws)])
        print(f"[{index}/{len(scenarios)}] {scenario}", flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
