# Historical Target10/20/30 UCM launchers

The scripts in this folder reproduce the manuscript-era `scenario510`,
`scenario520` and `scenario530` runs. They remain version controlled only for
audit history and must not be used for the revised Green–Target comparison.

Historical aliases were:

```text
scenario510 / Target10 <- trial 710v2
scenario520 / Target20 <- trial 730v2
scenario530 / Target30 <- trial 730v3
```

These rasters did not have equal eligible-canopy budgets relative to Green and
included some ineligible land-cover transitions. Their launchers also contain
legacy output conventions.

For production, use:

```text
code/Urban_Cooling_Modeling_Runs/
└── run_ucm_scenarios.py
```

That runner maps the canonical scenario names `target10`, `target20` and
`target30` to the validated `revised_v2_rank_fid_equal_area_2026-09-10`
rasters, enforces InVEST 3.20.2 and refuses to overwrite documented outputs.
