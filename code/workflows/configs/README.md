# Production run configurations

Store the exact JSON used for each production pipeline run in this folder.
Start from `../ucm_pipeline.example.json` and name the copy
`ucm_pipeline_YYYY-MM-DD.json`, using the run date. If more than one materially
different run is started on the same date, add a short purpose suffix, for
example `ucm_pipeline_2026-09-11_energy_sensitivity.json`.

Commit the configuration before running. Do not edit it after outputs have
been created; make a new dated configuration and output directory instead.
The external data paths remain relative to `URBAN_COOLING_DATA_ROOT`, while
the generated pipeline manifest records the resolved paths and checksums.
