# Reproducibility guide

This project supports two different reproducibility goals. Do not conflate
them: one verifies the reviewed manuscript assets; the other rebuilds the full
restricted-data analysis.

## 1. Verify the current manuscript package

The curated copies in `figures/manuscript_current/` are generated from the
versioned source files listed in
`code/workflows/configs/manuscript_figure_package_2026-11-13.csv`. Their
status, original source path, generator and SHA-256 hash are written to a
generated manifest.

```bash
export URBAN_COOLING_DATA_ROOT="/path/to/Wellcome Trust Project Data"
python code/workflows/build_current_figure_package.py --verify
python code/workflows/verify_reproducibility.py --require-data-root
```

To rebuild the curated copies after reviewing an approved figure update, run:

```bash
python code/workflows/build_current_figure_package.py --refresh
```

`--refresh` replaces only the selected files named in the configuration. It
does not delete historical figures, run outputs or diagnostic folders. Review
the changed source output before using it, then rerun `--verify`.

## 2. Rebuild the full analysis

Full reproduction needs authorized access to the Wellcome Trust Project Data
folder and third-party inputs. It cannot be completed from the public Git
checkout alone because raw land-cover, climate, scenario and model rasters are
not distributed here.

1. Clone the repository and use a clean working tree.
2. Copy the relevant line from [`.Renviron.example`](../.Renviron.example) to
   a user-local environment file; do not commit a personal path or token.
3. Create the two documented Conda environments:

   ```bash
   conda env create -f code/Urban_Cooling_Modeling_Runs/environment-invest-3.20.2.yml
   conda env create -f code/health_assessment/environment-health.yml
   ```

4. Verify installed versions before a production run:

   ```bash
   conda run -n urban-cooling-invest-3.20.2 invest --version
   conda run -n urban-cooling-health python --version
   Rscript --version
   ```

5. Start from the dated, immutable run configuration in
   `code/workflows/configs/`; validate it before running the model. Follow the
   [workflow guide](WORKFLOW.md) and [workflow README](../code/workflows/README.md).
6. Run the lightweight checks before regenerating figures:

   ```bash
   python -m unittest code.workflows.tests.test_run_ucm_pipeline -v
   python -m unittest code.workflows.tests.test_current_figure_package -v
   ```

7. Preserve the generated pipeline, health and figure manifests with any
   revised output. Create a new dated configuration and output directory for a
   changed assumption; never overwrite an existing production run.

## Sources of truth

- [`code/workflows/configs/run_registry.csv`](../code/workflows/configs/run_registry.csv)
  identifies the current figure-producing run families.
- [`figures/manuscript_current/manifest.csv`](../figures/manuscript_current/manifest.csv)
  is generated and verifies the selected manuscript copies.
- [`manuscript/index.md`](manuscript/index.md) is the reader-facing guide to
  the current figures, legends, Results text and scientific qualifications.
- [`data/README.md`](../data/README.md) identifies local data roles and
  handling boundaries.

The figure-package script is intentionally dependency-free. The full UCM,
health and R figure workflows retain their dedicated environments because they
require incompatible geospatial and statistical dependencies.
