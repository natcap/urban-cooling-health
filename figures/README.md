# Figure folders

Start with [manuscript_current](manuscript_current/README.md): selected review
copies and comparison versions share one folder. Its manifest identifies each
source, generator and hash. Do not manually edit curated copies; use the
[package workflow](../docs/reproducibility.md).

| Location | Role |
|---|---|
| `figure2/`, `figure6/`, `figure7/` | Versioned source figures and supporting analyses; Figure 7 separates PNGs and CSVs |
| `climate_projections/`, `health_summaries/`, `building_statistics/`, `vulnerability_maps/` | Topic-specific source exports and historical figures |
| Other topic folders | Supporting diagnostics and comparisons, not automatically selected for submission |
| `manuscript_current/` | Curated copies, not an independent analysis |

Preserve prior versions. Version figure filenames, not duplicate production
scripts. Do not infer manuscript selection from modification time. Existing
GIS projects, aliases and spreadsheets are retained but are not portable
reproduction entry points. Captions belong in [manuscript documentation](../docs/manuscript/index.md).
