# Production analysis workflows

Use this folder for end-to-end entry points. Model-specific scripts remain in
their existing folders so historical commands and provenance links do not
break, but routine runs should start here.

## Unified UCM and valuation pipeline

`run_ucm_pipeline.py` replaces the manual sequence of separate temperature,
energy, WBGT, Hothaps and summary commands. One JSON file controls scenario
rasters, temperatures, valuation temperatures, output location and the
optional historical comparison. The configuration is the scenario source of
truth: add a new lowercase name and relative LULC path there rather than
creating another scenario-specific Python script.

The recommended design runs:

- 25°C once with temperature, energy and WBGT enabled;
- 28°C once as a temperature-only sensitivity;
- Hothaps workability only for temperatures with WBGT outputs;
- citywide and borough summaries; and
- manuscript Figure 4 plus aggregation-sensitivity outputs when
  `figure4.enabled` is `true`;
- Figure 5 borough co-benefit maps when `figure5.enabled` is `true`; and
- the manuscript-era comparison only when explicitly enabled.

This avoids the previous duplicate 25°C UCM run.

### 1. Create a run configuration

Copy `ucm_pipeline.example.json` to a dated file such as
`configs/ucm_pipeline_2026-09-11.json`, replace `YYYY-MM-DD` inside it with the
same run date, and commit the configuration before a publication run. The
production command requires a clean Git working tree so the recorded commit
identifies the exact code and configuration. Do not reuse an output directory
for a materially different configuration.

### 2. Set the external data root

```bash
export URBAN_COOLING_DATA_ROOT="/path/to/Wellcome Trust Project Data"
```

All configured paths are relative to that root. Local usernames and mounted
drive paths therefore stay out of the committed configuration.

### 3. Inspect and validate

```bash
conda run -n urban-cooling-invest-3.20.2 python \
  code/workflows/run_ucm_pipeline.py \
  --config code/workflows/ucm_pipeline.example.json \
  --dry-run

conda run -n urban-cooling-invest-3.20.2 python \
  code/workflows/run_ucm_pipeline.py \
  --config code/workflows/ucm_pipeline.example.json \
  --validate-only
```

The dry run prints every command without writing outputs. Validation checks
all UCM configurations and inputs without running the model.

Create or update the environment from
`code/Urban_Cooling_Modeling_Runs/environment-invest-3.20.2.yml`. Its
`proj-data` dependency supplies the British National Grid transformation
files locally, so validation does not depend on access to an online PROJ grid.

### 4. Run or resume

```bash
conda run -n urban-cooling-invest-3.20.2 python \
  code/workflows/run_ucm_pipeline.py \
  --config code/workflows/configs/ucm_pipeline_2026-09-11.json

# After an interruption, use the identical config:
conda run -n urban-cooling-invest-3.20.2 python \
  code/workflows/run_ucm_pipeline.py \
  --config code/workflows/configs/ucm_pipeline_2026-09-11.json \
  --resume
```

Resume is deliberately strict. UCM runs are skipped only when the output,
manifest, InVEST version, model arguments and input checksums match. Hothaps
outputs are skipped only when their input/output checksums and parameters
match. For shapefiles, every dataset component—including the attribute table,
projection and spatial index—is recorded. Pipeline resume also requires the
same Git commit and working-tree state. An incomplete or changed run stops for
review instead of silently mixing versions.

`pipeline_run_manifest.json` records each stage's command, timestamps and
completion or failure status. Model and Hothaps manifests retain the detailed
input/output checksums.

Existing standalone output folders predate this pipeline manifest and should
remain immutable audit records. Do not retrofit them as a pipeline run or
weaken a checksum mismatch to make resume succeed; start with a new dated
output directory instead.

### Run selected stages

```bash
conda run -n urban-cooling-invest-3.20.2 python \
  code/workflows/run_ucm_pipeline.py \
  --config code/workflows/configs/ucm_pipeline_2026-09-11.json \
  --stages summarize compare \
  --resume
```

Use selected stages only when their upstream outputs already exist. The
historical comparison stage runs only when `historical_comparison.enabled` is
`true`; keeping it off avoids reading several gigabytes of archived building
outputs during ordinary production runs.

The Figure 4 stage is similarly opt-in because health assessment is a separate
review gate. After the reviewed health run exists, set `figure4.enabled` to
`true` and point `figure4.health_output_root` to it. The stage produces the
citywide manuscript figure and aggregation-sensitivity artifacts under
`summary/figure4/` without rerunning UCM.

Enable `figure5` after the same health review gate to produce the configured-scenario
borough maps under `summary/figure5/`. If its health path is omitted, the
pipeline reuses `figure4.health_output_root`; otherwise provide the reviewed
health output explicitly.

## What remains separate

Health assessment and Figure 7 remain separate review gates because they use a
different Python environment and manuscript-facing derived inputs. Run them
only after reviewing the UCM summary and explicitly selecting the production
temperature workspace. They are documented under
`code/health_assessment/README.md`.
