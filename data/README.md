# Data directory: what belongs here

This directory is a local working store, **not** a complete public data
release. Most contents are excluded from Git because they are large, may have
licence restrictions, or are machine-/collaborator-specific. Do not infer that
an ignored file is backed up elsewhere. The only explicitly tracked data
subfolder is the small, documented Figure 7 subset in
[`derived/README.md`](derived/README.md).

## Existing folder roles

| Location | Role | Handling |
|---|---|---|
| `01_raw/` | Downloaded archives and unmodified source snapshots, including boundary and road datasets | Preserve original archives; record source URL, version, acquisition date, licence and checksum before use. Do not rename or overwrite for a new analysis. |
| `02_preprocess/` | Spatial transformations and intermediate inputs | Keep reproducible scripts and input/output lineage; treat as generated, not as independent source data. |
| `derived/` | Small Figure 7 inputs, crosswalks, draw tables and manifests | See its README for the tracked subset and exact reproduction role. Other local files may be ignored. |
| `health-data/` | Nomis/ONS exports and health-model source/intermediate data | Preserve the original TSV exports; see [`DATA_SELECTION_AND_METHOD_DECISIONS.md`](../code/health_assessment/DATA_SELECTION_AND_METHOD_DECISIONS.md) for the selected mortality table and generated `mortality_2021_long.csv`. |
| Other named folders and top-level files | Historical analyses, collaborator deliveries, weather, tree and scenario inputs | Treat as legacy/local until an entry in a production manifest identifies the exact source and use. Do not bulk-move or deduplicate by filename alone. |

Production scenario, UCM and valuation rasters also live in the separate
Wellcome Trust Project Data store. Configure its root with
`URBAN_COOLING_DATA_ROOT` as described in the [project README](../README.md).
Do not copy that store into this repository merely to make a notebook run.

## Rules for new inputs and outputs

1. Put a new, unmodified download in `01_raw/` only after recording its source,
   version/date, licence/access class and checksum. Preserve the original
   filename when possible.
2. Put transformed intermediates in a named processing/output directory, not
   beside their source archive. Record the script/configuration, input
   checksums, CRS, grid/resolution, units, geography vintage and output
   checksum in a manifest.
3. Keep a public-ready, small derived file under `derived/` only after its
   source licence and disclosure status have been reviewed. Git ignores new
   files there by default; explicitly allow-list only approved outputs.
4. Keep credentials, personal cloud paths and restricted source rasters out
   of Git. See [the redistribution review](../DATA_LICENCE_AND_REDISTRIBUTION.md)
   before publishing machine-readable products derived from UKCEH LCM2023.
5. Do not delete a local or laptop copy until the authoritative destination,
   file integrity, licensing and backup have been checked. A Git push does not
   preserve ignored data.

The outstanding inventory and figure-to-input manifest are tracked in
[`TODO.md`](../TODO.md). This guide documents where data belong; it does not
claim that all historical files already have complete provenance.
