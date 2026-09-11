# Code guide

Start new production analysis from [`workflows/`](workflows/README.md). The
other folders contain reusable pipeline components and historical analyses.

## Production entry points

| Task | Use this file |
|---|---|
| Complete UCM, energy and productivity workflow | `workflows/run_ucm_pipeline.py` |
| UCM scenarios only | `Urban_Cooling_Modeling_Runs/run_ucm_scenarios.py` |
| Health scenario batch | `health_assessment/run_health_scenario_set.py` |
| Figure 7 | `run-fig7.R` |

## Naming convention

New production scripts use `verb_subject.py`:

- `run_*` orchestrates a model or multi-step workflow;
- `prepare_*` creates an input;
- `generate_*` creates a scenario;
- `validate_*` checks an input or result without changing it;
- `calculate_*` derives a modeled layer;
- `summarize_*` creates aggregate tables; and
- `compare_*` compares versions or scenarios.

Use lowercase `snake_case` for Python and JSON filenames. Put dates, model
versions and scenario revisions in committed configuration files and output
directory names, not in script filenames. Name production configurations
`ucm_pipeline_YYYY-MM-DD.json` and keep them under `workflows/configs/`. This
allows the code to remain stable while each run remains identifiable.

Existing collaborator scripts and R Markdown notebooks retain their original
names to preserve provenance. A name such as `scenario510`, `scenario4`,
`*_NEW`, or a numbered notebook should be treated as historical unless a
folder README explicitly marks it as production.

## Folder roles

- `workflows/` — one-command orchestration, run configurations and lightweight
  workflow tests under `workflows/tests/`;
- `lc_scenarios/` — scenario generation and validation;
- `Urban_Cooling_Modeling_Runs/` — UCM execution components and historical
  launchers;
- `post_processing_layers/` — Hothaps, energy/productivity summaries and
  version comparison; unit tests are under `post_processing_layers/tests/`;
- `health_assessment/` — mortality, health impacts and LSOA aggregation;
- `preprocessing_layers/` — spatial input preparation; and
- top-level `*.Rmd` files — manuscript analysis, figures and historical
  exploration.

Do not copy a historical script to create another scenario-specific runner.
Add the scenario to a configuration file or extend the shared component.
