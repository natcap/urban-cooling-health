# Why the Green–Target cooling ranking changed

11 November 2026 · [HTML](figure3_ranking_input_audit.html) · [Figure 3 review and figures](figure3_climate_uhi_audit.md)

## Finding in brief

**The Green30 reversal is not explained simply by moving canopy or upgrading
InVEST. A controlled, already-existing NoData-only test is the largest isolated
driver of the Green30 cooling reduction.** Revised canopy placement partly
offsets that reduction, while the revised Target design adds eligible canopy
and increases Target cooling. The precise decomposition is verified for the
30% tier; equivalent NoData-only controls for 10% and 20% were not found or run.

This is a read-only analysis of existing model outputs and current input files.
No model reruns, Word changes or pushes were performed. Numbers below use one
common London footprint, not the population-valid mask used in older diagnostic
reports. Confidence: **share with the stated attribution limits**.

## 1. Isolating Green30: a controlled sequence

All rows use the same 15,733,893 London cells and 33 boroughs; cooling is shown
as a positive reduction in temperature. The audit mask excludes 63 cells that
are present in the production Figure 3 mask so every historical/diagnostic raster
has identical support. No resampling was needed.

| Green30 input/run | Mean cooling across boroughs (°C) | London-area-weighted cooling (°C) | What this step changes |
|---|---:|---:|---|
| Historical Figure 3 raster | 0.373835 | 0.364478 | Reference |
| Same historical LULC rerun in 3.20.2 | 0.373835 | 0.364478 | Existing software/engine rerun; negligible numerical change |
| Same valid LULC, NoData recoded 255→0, in 3.20.2 | 0.324049 | 0.287505 | **Only missing-value encoding** |
| Revised Green30 LULC, NoData 0, in 3.20.2 | 0.336379 | 0.290959 | Revised canopy scenario at the same software/settings/encoding |

- **NoData-only effect:** reduces cooling by **0.049786°C** in borough means
  and **0.076973°C** in the London-area-weighted mean.
- **Subsequent canopy-scenario effect:** restores **0.012330°C** and
  **0.003454°C**, respectively. This includes changed planting positions,
  class composition and removal of woodland-to-100 relabelling; those individual
  contributions were not separately modelled.
- **Net Green30 reduction:** 0.037456°C across borough means and 0.073519°C
  area-weighted. A single percentage attribution would mislead because the
  canopy change offsets part of the NoData effect.

The NoData-only step leaves the valid land-cover classes **and their spatial
mask identical**, verified directly. It holds software versions, all non-LULC
arguments and ancillary input checksums constant. This is a numerical
missing-data/convolution sensitivity, not a physical reduction in tree canopy.
Its exact internal numerical pathway is not fully diagnosed here. It is **not**
the same as merely excluding 63 cells during summary calculation: the output
temperatures differ on the very same valid London cells.

For comparison, historical Target30 cooling is 0.366814°C across boroughs
(0.310221°C area-weighted). Even holding that Target result unchanged, the
NoData-harmonized historical Green30 falls below it. Thus the existing controlled
test is sufficient to reverse the displayed 30% ranking before the new canopy
scenario is introduced. This does not establish an exact version-only effect
for historical Target30, for which an unchanged-input control was not available.

Expressed directly as **Target30 cooling minus Green30 cooling**, the borough
ranking shifts from −0.007022°C (Green advantage) to +0.041586°C (Target
advantage). The 0.048608°C shift decomposes along this sequence into about
**+0.049786°C from Green NoData harmonization**, **−0.012330°C from the subsequent
Green canopy redesign**, and **+0.011152°C from the combined Target revision**,
with a negligible historical-versus-rerun Green residual. This is a sequential
decomposition, not a factorial estimate of independent effects or interactions.

## 2. Specific canopy changes, verified from rasters

An eligible addition is a baseline code **4, 20 or 21 → 100**. These figures
cover the full supplied scenario grid, not only the London summary mask.

| Tier | Old Green eligible cells | Revised Green and Target cells (each) | Old Target eligible cells | Extra eligible Target cells | Old Green locations replaced |
|---|---:|---:|---:|---:|---:|
| 10 | 307,768 | 307,768 | 273,242 | 34,526 | 61,870 |
| 20 | 595,084 | 595,084 | 535,737 | 59,347 | 92,396 |
| 30 | 894,249 | 894,249 | 858,121 | 36,128 | 156,345 |

**Green:** eligible canopy quantity is unchanged, but placement is not. The
revised generator uses focal codes `{1,2,100}` instead of `{1,2,4,20,21}`,
restricts conversions to `{4,20,21}`, and builds nested increments. Historical
Green also changed 12,232 / 24,916 / 35,751 woodland cells to code 100; those
changes are removed. They were not newly added canopy, but were not necessarily
thermally neutral: the biophysical table assigns woodland Kc=0.7 and code100
Kc=0.8. Revised Green30 shifts eligible additions from grassland toward suburban
and urban classes (full transition counts in the audit CSV).

**Target:** every historically eligible added cell is retained; additional
eligible cells fill the shortfall to each Green budget. Revised scenarios also
exclude 3,188 / 7,248 / 11,323 old conversions from ineligible classes. Selection
uses deterministic `(rank, FID)` ordering and a 5 m radius with pixel-centre
inclusion. The eligible area increments are 3.4526, 5.9347 and 3.6128 km².
The old Green–Target comparison therefore did **not** match eligible added area.

Revised Target additions are 71.3%, 69.1% and 67.3% Urban (code20), versus
23.6%, 25.0% and 27.1% for Green. Converting low-shade, low-Kc urban surfaces
to tree cover can yield more additional cooling than replacing already-green
land. This is a mechanistic explanation consistent with the input table,
**not a separately identified causal contribution**. Candidate-domain and
source-class differences mean the benefit cannot be attributed solely to the
social-vulnerability ranking. Full scenario rules remain in the
[scenario README](../../code/lc_scenarios/README.md), not duplicated here.

## 3. Model-version and grid effects: what is and is not established

- **Grid geometry:** all 17 temperature rasters checked share the same
  5,840×4,509 grid, 10 m cells, origin (503560, 200940), and projected CRS.
  The ranking reversal remains on an identical London mask. A change of output
  grid origin or aggregation footprint is not needed to produce it.
- **Historical Green30/baseline versus existing 3.20.2 reruns:** paired cooling
  differs by about 10⁻⁸°C in these summaries, effectively zero. A script header
  saying “generated by 3.14.1” does not identify the runtime that produced the
  archived Figure 3 raster; do not label this a verified 3.14.1→3.20.2 experiment.
- **The earlier +0.438°C diagnostic is a different comparison:** it tested an
  intermediate Target30-v4 input across model formulas/grids. It concerns
  absolute Target temperature, not the final old-versus-revised paired
  Green–Target cooling contrast. Its +0.501°C park-formula and −0.063°C
  alignment contributions must **not** be imported into this decomposition.
  See the [version investigation](../../code/Urban_Cooling_Modeling_Runs/INVEST_VERSION_COMPARISON.md).
- **Target:** the old-to-new changes combine revised area/eligibility/placement
  with any unverified historical runtime effects. Without a same-input Target
  rerun, a numerical percentage assigned to model version would be speculative.
  Historical Target filenames follow the confirmed manuscript aliases, but an
  immutable original-run input checksum/runtime record was not established here;
  current-file transition counts are not proof of every historical run snapshot.
- **10/20 tiers:** both old Green inputs use NoData255 and revised inputs use
  NoData0. Their temperature shifts are consistent with the 30% diagnosis, but
  its numerical attribution cannot be transferred to them without controls.

## 4. Meaning for the manuscript and next decision

Use the internally consistent revised outputs. Describe Target benefits as
results of the **revised matched-area allocation design**, not proof that
vulnerability ranking alone improves cooling. Keep the prescribed-climate
caveat from the Figure 3 review. The old comparison is useful provenance, not
an alternative set of numbers to combine with revised health outcomes.

For strict attribution at every tier, the smallest next experiment is to rerun
historical Green10/20 with only NoData changed, plus historical Target10/20/30
unchanged in 3.20.2. To separate individual canopy-rule effects further would
require additional counterfactual LULC runs. **None were launched in this audit.**
The NoData sensitivity also warrants a focused model-input handling test before
claiming absolute numerical robustness; it should not be hidden as a cosmetic
file-format change.

## Reproduction, files and checks

Generating analysis script: [audit_figure3_ranking_inputs.py](../../code/post_processing_layers/audit_figure3_ranking_inputs.py).
It reads existing legacy, diagnostic and production rasters, checks grid
equality and manifest/input checksums, recomputes same-mask temperature means,
and independently counts canopy transitions/overlap. It generates no figures.

```sh
"$URBAN_COOLING_PYTHON" code/post_processing_layers/audit_figure3_ranking_inputs.py "$URBAN_COOLING_DATA_ROOT" --output-dir "$URBAN_COOLING_DATA_ROOT/2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/summary"
```

Outputs: `figure3_ranking_input_audit_20261111_temperatures.csv`, `_canopy.csv`
and `_manifest.json`, pooled in the production valuation summary folder. The
manifest records exact source paths and checksums. Existing controlled runs
were generated using [execute_invest_urban_cooling_model_health_scenarios.py](../../code/Urban_Cooling_Modeling_Runs/execute_invest_urban_cooling_model_health_scenarios.py)
and the NoData input was prepared by [prepare_green30_nodata_harmonized.py](../../code/Urban_Cooling_Modeling_Runs/prepare_green30_nodata_harmonized.py).
For the figure-generating scripts and versioned PNGs, use the
[Figure 3 script table](figure3_climate_uhi_audit.md#original-workflow-reproduction-and-controlled-comparison-11-november).

HTML: `pandoc docs/manuscript/figure3_ranking_input_audit.md --standalone --embed-resources --toc --css docs/manuscript/figure_review.css -o docs/manuscript/figure3_ranking_input_audit.html`.
