# Figure 3 climate–UHI framing: audit and recommendations

The current parameter-preparation evidence and validation gate are maintained in
the [UCM climate-parameter pilot](ucm_climate_parameter_pilot.md); details are not
duplicated here. The author has retained 25/5°C for manuscript reporting as a
prescribed warm-condition stress-test, with 28/5°C retained for 2070; see the [decision and
qualification](ucm_climate_parameter_pilot.md#retained-manuscript-configuration).
Existing figures and production results are unchanged by this decision.

11 November 2026 · [Copy-friendly HTML](figure3_climate_uhi_audit.html)

**Current decision:** restore original-style year labels and simplified axes;
panel b reports equal-weight borough means for all nine scenarios.
The current output is `figure3_main_v03_20261111.png`, in the same
external production-summary folder as previous versions. No model run, Word
edit or push was performed. Earlier versions remain available.

## Current approved reporting version

### File naming and version comparison

Files use `figure3_<family>_v##_YYYYMMDD`, matching Figure 1's naming scheme.
They remain together in the external production-summary folder linked below.

| Family | Available versions | Meaning |
|---|---|---|
| `figure3_main` | v01, v02, **v03 current** | V03 uses self-contained y-axis labels and slightly greater height |
| `figure3_sm_borough` | v01, **v02 current** | Supplementary borough distributions |
| `figure3_legacy_replay` | v01, v02 | Original-input workflow reproduced for comparison, not current production |
| `figure3_alternative_area_weighted` | v01 | Earlier area-weighted alternative, not selected |
| `figure3_draft_panel_a/b` | v00_ms0806 | Exact images extracted from draft 0806; export date is not inferred |

Related CSVs and manifests share their figure's stem. Renaming preserved every
file's contents (SHA256 checked before/after each move); script/document paths
and manifest references were then updated. No figure was regenerated or deleted.
Sort by filename and use Finder Quick Look to compare adjacent versions.

[Open Figure 3 v03](</Users/yingjiel/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/summary/figure3_main_v03_20261111.png>) ·
[Borough-distribution supplement](</Users/yingjiel/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/summary/figure3_sm_borough_v02_20261111.png>)

V03 uses **Temperature change from baseline (°C)** in panel a and
**Temperature change due to land use (°C)** in panel b. Height increases from
3.7 to 4.2 inches; width remains 7.2 inches (450 dpi). Panel b bars use
alpha = 0.7, matching the original notebook; this styling adjustment was
saved to v03 at the author's request. Data, year labels and
the supplementary figure are unchanged. V01 and v02 remain available.

V02 removed both panel titles, restored `Year` and `Temperature change (°C)`
axis text, uses the original stem/point styling and internal legend, and places
equally sized panels side by side. Panel b restores angled scenario labels.
All numerical results are unchanged. Year labels are shorthand for the
original assumed settings, as defined below; they do not establish a direct
link to Figure 1's climate projections. Previous PNGs and manifests are retained.

### Proposed Methods: climate scenarios and temperature summaries

We compared land-use scenarios using InVEST 3.20.2 under assumed climate
conditions, defined by reference
temperature/maximum urban heat island (UHI) settings of 22/2, 22/5, 25/5 and
28/5°C, labelled 2021, 2030, 2050 and 2070, respectively. Future settings
represent illustrative warm-climate scenarios, not calibrated projections from Fig. 1.
Land-use effects were calculated relative to baseline land use under identical
climate conditions and other model inputs. Figure 3 reports equal-weight means
across London's 33 boroughs. Temperature-field reconstruction and UHI
sensitivity procedures are described in Supplementary Methods.

### Proposed main legend

**Fig. 3 | Modelled temperature changes across alternative climate and land-use
scenarios. a,** Temperature differences from baseline land use at 22/2°C.
Year labels denote assumed reference-temperature/maximum urban heat island
(UHI) settings: 2021, 22/2°C; 2030, 22/5°C; 2050, 25/5°C; and 2070, 28/5°C,
rather than calibrated projections. **b,** Land-use effects relative
to baseline land use at 25/5°C. Points and bars show equal-weight means across
London's 33 boroughs; stems connect points to zero. Negative values indicate
cooling. Scenario definitions are provided in Table 1.

### Proposed Supplementary Methods: temperature reconstruction and UHI sensitivity

The 2050 (25/5°C) and 2070 (28/5°C) configurations were retained as illustrative
warm-climate scenarios. The 5°C maximum UHI is not an estimate of
representative July daily-maximum urban–rural contrast. Daily maximum was the
primary statistic in the observational pilot to match the July Tmax climate
reference; daytime and night-time means provided separate diagnostic checks.

For fixed land cover and other inputs, InVEST calculates unmixed temperature as
`T_ref + (1 − HM) × UHI_max`, where `HM` is the heat-mitigation index, followed
by normalized spatial mixing. We reconstructed each 22/5°C field as
`T25_5 − 3°C` and the baseline 22/2°C field as
`22 + (2/5) × (Tbaseline25_5 − 25)`. The translation property was checked
against executed 25/5°C and 28/5°C runs for all ten land-use configurations;
the maximum cell discrepancy from a 3°C shift was 0.000018°C. Comparisons used
15,733,956 common valid London cells. Panel a subtracts the baseline 22/2°C
field; panel b subtracts baseline land use at 25/5°C. Cell temperatures were
averaged within each borough, then equally across the 33 boroughs. Geographic
variation was not interpreted as propagated model uncertainty.

To assess sensitivity to maximum UHI, we ran temperature-only configurations
at `T_ref = 25°C` and `UHI_max = 2 and 3°C` for baseline land use and all nine
alternative scenarios, and compared these with existing 5°C outputs. Land
cover, evapotranspiration, biophysical parameters and spatial settings were
held fixed. This diagnostic used area-weighted London means on the common
valid grid and borough-specific contrasts, distinct from Fig. 3's equal-weight
borough summary. Scenario rankings were compared across settings. No energy,
health or work-capacity sensitivity was calculated; temperature-ranking
stability therefore does not establish robustness of downstream magnitudes.

Implementation and detailed evidence: [generating scripts and checks](#generating-scripts-and-checks)
and the [climate-parameter pilot](ucm_climate_parameter_pilot.md#temperature-only-uhi-sensitivity).

### Proposed Results

Greening reduced modelled temperatures but did not offset the increase
relative to the 2021 reference configuration under the assumed 2050
(25/5°C) and 2070 (28/5°C) climate conditions (Fig. 3a). Under the assumed
2050 conditions,
AllBuilt and TreeRisk increased mean temperature across boroughs by 1.05°C
and 0.10°C, respectively, whereas TreeOpp reduced it by 0.06°C. Cooling
increased with canopy expansion and was greater under Target than Green
scenarios: reductions were 0.12, 0.23 and 0.34°C for Green10, Green20 and
Green30, compared with 0.14, 0.26 and 0.38°C for Target10, Target20 and
Target30 (Fig. 3b). All six greening scenarios cooled every borough, with
spatial variation in the magnitude of cooling. At a fixed reference temperature
of 25°C, varying maximum UHI from 2 to 5°C preserved the ranking of all nine
land-use scenarios, although absolute cooling scaled with UHI magnitude.

**Author check before insertion:** insert the concise Methods in the main text
and the reconstruction/sensitivity procedures in Supplementary Methods; assign a supplementary reference if including the
UHI sensitivity figure. Word has not been updated. These are model-based
scenario comparisons, not measured warming since 2021.

### Supplementary legend

**Borough variation in modelled land-use temperature effects.** Scenario
minus baseline temperature at common reference-temperature/UHI settings of
25°C/5°C. Each point is one of 33 boroughs. Boxes show medians and interquartile
ranges; whiskers extend to observations within 1.5 interquartile ranges.
Negative values indicate cooling. The spread represents geographic variation,
not propagated model uncertainty.

### Generating scripts and checks

- Entry notebook: [invest_result_zonal_viz_1_temp.Rmd](../../code/invest_result_zonal_viz_1_temp.Rmd).
- Summaries/validation: [summarize_figure3_prescribed.py](../../code/post_processing_layers/summarize_figure3_prescribed.py).
- Both PNGs: `build_figure3_borough_reporting()` in
  [plot_figure3_temperature.R](../../code/post_processing_layers/plot_figure3_temperature.R).
- Shared colours: [func_colors.R](../../code/func_colors.R).

Run the summary script with `--production-root`, `--borough-vector` and
`--output-dir` (the production `03_energy_work_capacity/summary` directory),
then call the R builder with that summary directory and repository `code/`.
The entry notebook does both using the shared configuration.

The same revised canopy/model inputs are used throughout. There are
15,733,956 common London cells and 33 boroughs. For each of ten land-use
configurations, the existing 28°C/5°C raster was compared with 25°C/5°C + 3°C:
maximum cell discrepancy **0.000017524°C**. Run metadata also agrees on canopy,
evapotranspiration, biophysical table and cooling settings. The model computes
`T_nomix = t_ref + (1-HM) * uhi_max` and normalizes its spatial convolution.
Thus, with these inputs/masks fixed, `T22_5 = T25_5 - 3` and
`Tbaseline22_2 = 22 + 0.4 * (Tbaseline25_5 - 25)`.
The latter UHI scaling is a structural derivation, not independently validated
against a new 22°C/2°C run. The checksum manifest records each source and
validation result; these derived settings must not be called new model runs.

**Why these choices:** preserve the original visual explanation, avoid implying
that UCM settings are Figure 1 projections, maintain the requested borough
estimand, and remove error bars that measured borough variation rather than
model uncertainty. This does not change Figure 4's separately defined weighting.
The old faithful replay and the earlier area-weighted redesign are retained
for provenance; neither is the current reporting version. Abstract revision
and broader manuscript reviews remain on hold.

Figure 1's author-selected CHELSA reference, additional historical WorldClim
reference and supplementary comparison are documented once in the
[historical-reference comparison](figure1_historical_reference_comparison.md).

## Earlier audit and alternatives (retained for provenance)

The earlier proposed legend/Results below describe archived alternatives,
not the current selection above.

## Author clarifications and holds 11 November

**Holds:** Abstract revision waits until the main sections are addressed.
Further manuscript-review work is paused until explicitly requested. The
clarifications here do not edit Word, rerun models or change figures.

### Borough variation is not model uncertainty

The historical temperature notebook calculates the mean across 33 boroughs,
then `mean ± 1.96 × SD/√33`. Its SD describes **differences between borough
outputs**, not uncertainty in temperature inputs, cooling parameters, model
structure or canopy assumptions. Under independent random sampling, this is
an approximate sampling interval for a population mean. Here all 33 London
boroughs are included, and adjacent boroughs are spatially related. It cannot
be described as propagated “95% confidence intervals for model estimates.”
The arithmetic is not inherently wrong; the interpretation is the problem.

For example, perfectly known modelled cooling of 0.1°C in one borough and
0.5°C in another has spatial variation without uncertainty about those two
model outputs. Conversely, all borough outputs can be similar while sharing
a large uncertainty in the same UHI parameter: the old error bars would be
small even though predictions are uncertain. The original panel b also
averages the period-specific means and interval bounds; that is not a new
propagated uncertainty interval.

**Recommended reporting:** deterministic London mean without uncertainty bars;
show borough points/boxplots or ranges separately. If retaining the original
bar figure, remove its uncertainty bars in a future reporting version, or
explicitly label the retained legacy bars as borough-based normal-approximation
intervals under an independence assumption, not model uncertainty. For genuine
model uncertainty, define plausible input/parameter distributions, propagate
joint draws through the relevant model chain, and summarize the resulting
London means. A few sensitivity settings or GCMs alone are not automatically
a probabilistic 95% interval. No such new analysis is implemented here.

### The exact GCM mismatch and citywide wording

The current Word **Methods → Climate scenarios, first paragraph** names
“EC-Earth3-Veg (EC-Earth consortium), MRI-ESM2-0 ... and IPSL-CM6A-LR.”
The actual 18 borough CSVs selected by the climate notebook contain six files
each for **ACCESS-CM2, IPSL-CM6A-LR and MRI-ESM2-0**, covering two SSPs and
three periods. No EC-Earth file enters that selection. Supplementary Fig. 12
also names ACCESS-CM2. Therefore the Methods does not match the current
notebook inputs. If these are the intended inputs, replace EC-Earth3-Veg with
ACCESS-CM2 and its correct model-centre attribution; if EC-Earth was intended,
resolve the input choice rather than just renaming files. This verifies the
present workflow, not every historical download.

The specific **“Citywide mean”** concern is in **Results → Heat exposure and
baseline inequalities across London, second paragraph**, beginning “Across
both SSP scenarios ...”, at the sentence “Citywide mean T_WM rises from
21.9°C ...”. The following sentence already says “Averaged across boroughs.”
The Figure 1 diamond represents an **equal-weight mean of borough means**, not
the area-weighted London raster mean. Suggested minimal wording: “The mean
across London's 33 boroughs rises ...”. Figure 1 legend should define its
diamond as the unweighted borough mean. This is not a blanket objection to
“citywide” elsewhere: Figure 4 totals/area-weighted summaries have their own
definitions, and Figure 6's z-score reference describes a different mean.

### What recalculated warming means here

These are rechecked **Figure 1 cross-product differences**, not newly
harmonized climate projections or the new Figure 3 land-use contrasts. The
same original process is retained: match borough names, subtract July 2021
CHELSA borough means from future WorldClim means, average three models, then
give the 33 boroughs equal weight. Baseline mean is 21.85052135°C. Only the
arithmetic/rounding has been rechecked; no climate input, weighting or model
was changed for this calculation.

| Period label | Recomputed SSP370–SSP585 difference from July 2021 (°C) | Correct one-decimal range | Current Results range |
|---|---|---|---|
| 2030 (2021–2040) | 2.97284–3.25226 | 3.0–3.3 | Not stated as a difference |
| 2050 (2041–2060) | 3.86794–4.21436 | 3.9–4.2 | 3.8–4.2 |
| 2070 (2061–2080) | 4.79076–5.46397 | 4.8–5.5 | 4.7–5.4 |

Thus the small revisions to these ranges are rounding/source-table
reconciliation, not evidence of a changed warming model. The distinct old
Figure 3 baseline-land-use differences (2.168/5.168/8.168°C) and their parameter
decomposition are documented in section 2 below; they are different quantities.

**Scripts:** Figure 1 is generated by
[gcm-data-clip-stats-viz.Rmd](../../code/gcm-data-clip-stats-viz.Rmd).
[audit_figure1_climate_summary.R](../../code/post_processing_layers/audit_figure1_climate_summary.R)
independently prints the model inventory and above means/ranges from the same
CSV inputs without generating figures. Run `Rscript code/post_processing_layers/audit_figure1_climate_summary.R`
from the repository root. It validates 33 matched boroughs per file and three
models per SSP/period. The Figure 3 script inventory remains below.

### Why CHELSA 2021 minus future WorldClim needs qualification

The subtraction is a valid **difference between the selected datasets**. It
is not automatically a clean estimate of long-term climate change: a single
July can be warmer or cooler than its contemporary climatology, and different
products can have different spatial calibration/downscaling biases. The result
mixes those offsets with the change in climate. WorldClim states that its
[future data](https://www.worldclim.org/data/cmip6/cmip6climate.html) are
20-year means calibrated to WorldClim v2.1, whose
[historical climate baseline](https://www.worldclim.org/data/worldclim21.html)
is 1970–2000. This calibration does not automatically harmonize a CHELSA
single-year baseline.

Illustration, not London results: future climatology 26°C minus a 2021 value
of 22°C gives 4°C. If the matching long-term baseline is 23°C, the anomaly
relative to that baseline is 3°C. The additional 1°C need not be future warming.

**Author decision, superseding the initial proposal:** retain CHELSA July 2021
for the main Figure 1 differences and Figure 2. Add historical WorldClim as a
separate reference and use matching-WorldClim differences in the supplement.
The download, checks, numerical comparison and proposed labels are in the
[reference comparison](figure1_historical_reference_comparison.md). Do not
interpret either historical reference as an interchangeable measurement.

### Prescribed conditions do not have to come from Figure 1

**Correction to earlier wording:** directly deriving Figure 3a from Figure 1
is not a requirement. A sensitivity experiment can legitimately prescribe
rural reference temperature and maximum UHI independently. What matters is
not describing that experiment as a calibrated projection of the same
SSP/year conditions without establishing the link.

For example, raising reference temperature from 25°C to 28°C asks how outcomes
respond under a 3°C warmer background; it does not establish that this is
London in 2070. Likewise, changing UHI from 2°C to 5°C imposes an additional
urban–rural contrast. It is not a future change inferred from Figure 1.
Figure 1's London borough temperature is also not interchangeable with UCM's
**rural** reference temperature.

Two defensible options remain: (1) retain the old panel-a style as a prescribed
climate/UHI sensitivity and label the combinations as such, avoiding unsupported
calendar forecasts; or (2) use the new matched-condition two-panel Figure 3
for land-use effects and keep Figure 1 as climate context. A directly coupled
forecast is a third, more demanding choice requiring rural-reference and UHI
justification. No further change to Figure 3a is made without agreement.

### Equal borough weights versus London area weights

Neither mean is inherently wrong. Equal borough weights answer “what is the
average borough's modelled change?”; area weights answer “what is the average
change over London's land area?”. For a 10 km² borough cooling by 1°C and a
90 km² borough cooling by 0.1°C, the equal-borough mean is 0.55°C, while the
area-weighted mean is 0.19°C. Neither is a resident-weighted exposure estimate.

For revised Green30/Target30, equal-borough cooling is 0.3364/0.3780°C and
area-weighted cooling is 0.2910/0.3192°C. This difference changes magnitudes,
not their revised ranking. Recommend area weighting for claims about London's
land-area mean, borough means/distributions for geographic comparison, and
explicit labels for both. The error-bar interpretation is a **separate** issue:
changing weights does not by itself propagate model uncertainty.

### Figure 3 changes and their justification

- **Reproduced original workflow:** retained old input table, parameter/date
  labels, equal-borough calculations and period averaging; restored the three
  Target scenarios excluded by the original presentation filter. V2 changes
  only presentation (internal panel-a legend, clearer panel-b labels/axes);
  its numerical comparison CSV is identical to v1.
- **Revised analytical figure:** uses revised matched-area canopy production,
  common 25°C reference/5°C UHI conditions, a London-area mean in a and borough
  distributions in b. It isolates land-use differences and does not imply
  propagated model intervals. This is a change of scientific presentation,
  not a faithful rerun of the original structure.
- **Input effects are separate:** the ranking reversal is not explained by
  plotting or weighting alone. The verified NoData-only Green30 effect and
  remaining canopy/model attribution limits are documented once in the
  [input audit](figure3_ranking_input_audit.md).

Both versions and all generating-script links are retained below for the
author's decision. No final manuscript selection is implied by the clarification.

## Archived area-weighted alternative and its text

**Output:** `figure3_alternative_area_weighted_v01_20261111.png`, in the external data
root under `2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/summary/`.
The same stem identifies the citywide/borough CSVs and checksum manifest.
Previous manuscript panels are retained there as
`figure3_draft_panel_a_v00_ms0806.png` and `_b.png` (exact embedded-image copies).

[Open the new Figure 3 PNG](</Users/yingjiel/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/summary/figure3_alternative_area_weighted_v01_20261111.png>).

### Proposed Figure 3 legend

**Fig. 3 | Modelled temperature effects of alternative land-use scenarios across
London.** Changes are relative to baseline land use under identical prescribed
conditions (rural reference air temperature, 25°C; maximum urban heat island
intensity, 5°C). **a,** London-area-weighted mean changes for nine scenarios.
**b,** Distributions of mean changes across 33 boroughs; each point represents
one borough. Boxes show the interquartile range and median; whiskers extend to
the most extreme values within 1.5 interquartile ranges of the box. All borough
values are plotted. Negative values indicate cooling. Panels use different
vertical scales. Borough spread represents geographic variation, not model
uncertainty. Scenario definitions are provided in Table 1; these comparisons
isolate land-use effects rather than changes between climate periods.

### Proposed Results

Under common prescribed warm conditions, AllBuilt increased London's
area-weighted mean temperature by 1.11°C, whereas TreeRisk increased it by
0.08°C and TreeOpp reduced it by 0.06°C (Fig. 3a). Cooling increased with canopy
expansion: Green10, Green20 and Green30 reduced temperature by 0.10, 0.20 and
0.29°C, respectively, compared with 0.11, 0.22 and 0.32°C for the corresponding
Target scenarios. All six greening scenarios cooled every borough, although
magnitudes varied geographically. Borough-mean cooling ranged from 0.09 to
0.60°C for Green30 and from 0.08 to 0.75°C for Target30 (Fig. 3b).

**Abstract implication:** use “up to 0.32°C in London's area-weighted mean” for
the strongest cooling among these nine scenarios, not the legacy 0.37°C
Green30 borough-average value. The 0.75°C maximum is a borough mean, not the
London mean or a pixel maximum. Any warming claim must likewise distinguish
AllBuilt's 1.11°C London mean from its 1.56°C maximum borough mean.

### Production and verification

- Entry point: [original temperature notebook](../../code/invest_result_zonal_viz_1_temp.Rmd).
  Its new production block replaces the default workflow; historical chunks
  remain visible but are disabled when knitting.
- [Temperature summarizer](../../code/post_processing_layers/summarize_figure3_temperature.py)
  pairs ten existing rasters (baseline plus nine scenarios), checks common grids
  and prescribed settings, and uses an exclusive London cell-centre mask.
- [Figure builder](../../code/post_processing_layers/plot_figure3_temperature.R)
  uses the shared scenario palette, signed regular-weight labels, lower-case
  panel tags and no redundant legend or in-figure caption. Sans-serif text and
  restrained line weights follow [Nature's artwork guidance](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
  The 450-dpi PNG is for review; editable final artwork remains a submission task.
- Validated 15,733,956 shared cells, 33 unique boroughs and 297 scenario–borough
  pairs. Cell-weighted borough contrasts reproduce London contrasts to less
  than 10⁻¹⁰°C. Duplicate temperature sources match byte-for-byte; source,
  boundary, code, palette and output checksums accompany the export. No statistical
  significance or propagated uncertainty is claimed.
- To reproduce, set `URBAN_COOLING_DATA_ROOT` and `URBAN_COOLING_PYTHON` (the
  GDAL-enabled InVEST environment's Python executable), then knit the notebook.
  Re-knitting refreshes the same version; change the shared output stem in both
  helpers before making a new review version. Existing older figures are not removed.

## Recommendation in brief

### Original-workflow reproduction and controlled comparison (11 November)

The original Figure 3 has now been **rerun**, not merely copied from Word.
`figure3_legacy_replay_v02_20261111.png` and separate `_a.png` and
`_b.png` panels are saved in the same summary folder as the revised version.
All nine scenarios were already present in the archived input table. The old
presentation call explicitly removed labels containing `Target`; that filter
is omitted in this reproduction. The original calculations, stem/point panel,
grey bars, coloured outlines and error-bar calculations are retained.
V2 returns the legend inside panel a, and refines panel b with regular dark
labels outside error bars, smaller sans-serif type and restrained axes. V1 is
retained as the closer original-style comparison. Calculations are unchanged;
these are presentation adjustments, not a new analytical version. This is faithful to the available script, not
a claim of pixel-identical recreation of the manually assembled Word figure.

**Scripts and inputs used:**

| Figure/export | Generating script | Source and calculation |
|---|---|---|
| Original-workflow all-nine PNG and separate panels | [reproduce_figure3_original.R](../../code/post_processing_layers/reproduce_figure3_original.R), replaying functions and the Fig. 3a plotting expression directly from [invest_result_zonal_viz_1_temp.Rmd](../../code/invest_result_zonal_viz_1_temp.Rmd) | `data/ucm_output_shp_stats_outcomes_by_borough.rds`; the historical preparation entry point is [invest_result_zonal_viz_0_data_prep.Rmd](../../code/invest_result_zonal_viz_0_data_prep.Rmd). No revised temperature data substituted. |
| Revised two-panel PNG | [plot_figure3_temperature.R](../../code/post_processing_layers/plot_figure3_temperature.R), called by the same original notebook's new production block | CSVs generated from revised production rasters by [summarize_figure3_temperature.py](../../code/post_processing_layers/summarize_figure3_temperature.py). |
| Numerical bridge/comparison CSV | [reproduce_figure3_original.R](../../code/post_processing_layers/reproduce_figure3_original.R) | Archived borough summaries versus the revised citywide CSV; no model rerun. |
| Preserved manuscript PNGs | No regeneration script: exact embedded-image copies | Word `image10.png` and `image11.png`, verified byte-for-byte; distinct from the rerun above. |

To rerun the original comparison after the revised summaries exist:

```sh
Rscript code/post_processing_layers/reproduce_figure3_original.R "$PWD" "$URBAN_COOLING_DATA_ROOT/2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/summary"
```

It exports `_by_year.csv`, `_relative_2021.csv`, `_comparison.csv` and a checksum
manifest alongside the PNGs. The replay does not execute the notebook's old
save commands or overwrite its saved R functions. Validation confirmed the
original row-order subtraction reproduces explicit borough/year-keyed means
**and standard deviations** within 10⁻¹⁰°C for these inputs (33 boroughs in each
scenario/year). That is a check of this table, not an endorsement of row-order
matching for future inputs.

**What differs:**

| Component | Original-workflow reproduction | Revised Figure 3 |
|---|---|---|
| Inputs | Archived borough RDS and original scenario configurations | Revised equal-area production temperature rasters, InVEST 3.20.2; shared London mask |
| Panel a comparator | Every configuration relative to 2021-labelled baseline land use (22°C reference, 2°C UHI) | Each scenario relative to baseline land use under the same 25°C reference and 5°C UHI |
| Panel a aggregation | Equal-weight mean of 33 boroughs; includes changes in reference temperature and UHI settings | Area-weighted London mean; isolates land-use contrast |
| Panel b | Land-use-only borough means calculated separately for 2030, 2050 and 2070, then averaged across those labels | Distribution of 33 borough contrasts at one prescribed setting |
| Error bars | Mean ± 1.96 SD/√33 for each period, then bounds averaged across periods | No model-uncertainty bars; boxplots and points show geographic variation |
| Climate interpretation | Calendar labels retained strictly for legacy comparison; not direct Figure 1 projections | Explicitly conditional on prescribed warm conditions |

**Numerical attribution:** the old future-period averaging versus selecting
only the old 2050-labelled run changes means by less than 0.0000001°C. It is
therefore immaterial here. Next compare old 2050 borough means with revised
borough means: this captures the **combined revised input/production and mask
effect**, not a clean test of any one model function. Finally compare revised
borough and London means: this isolates the reporting-weight change on the
same revised outputs.

For Green30, −0.3738°C becomes −0.3364°C with revised inputs/mask (+0.0375°C),
then −0.2910°C with area weighting (+0.0454°C). For Target30, −0.3668°C becomes
−0.3780°C (−0.0112°C), then −0.3192°C (+0.0587°C). Positive shifts mean less
cooling. AllBuilt, TreeRisk and TreeOpp borough means remain unchanged to
approximately 10⁻⁷°C; their London-mean differences are almost entirely weighting.

**Conclusion:** warming under AllBuilt/TreeRisk and cooling under TreeOpp and
all greening scenarios are consistent. The Green–Target ranking changes:
original Green scenarios cooled more on average at all three levels; revised
Target scenarios cool more under both borough and area-weighted summaries.
Thus the reversal cannot be attributed to area weighting alone. Do not attribute
it solely to an InVEST version change. The subsequent
[input-attribution audit](figure3_ranking_input_audit.md) isolates Green30's
NoData sensitivity using an existing controlled run; it is the largest isolated
driver, partially offset by the revised canopy scenario. Other tier-specific
contributions remain unresolved. Use the revised
production results for current manuscript claims; retain the original reproduction
for traceability rather than mixing legacy values with revised health/valuation outputs.

### Visual comparison

Original workflow, expanded to nine scenarios (legacy year framing retained):

![Original workflow Figure 3 with nine scenarios, presentation v2](</Users/yingjiel/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/summary/figure3_legacy_replay_v02_20261111.png>)

Revised matched-condition Figure 3 (different panel meanings; see table above):

![Revised Figure 3](</Users/yingjiel/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/summary/figure3_alternative_area_weighted_v01_20261111.png>)

### Recommended manuscript framing

Keep Figure 1 as the climate-projection context. Rebuild Figure 3 as a
matched-condition land-use comparison using the revised nine-scenario outputs,
not as a second calendar-year warming projection. Explain that the primary UCM
assessment uses a prescribed rural reference temperature of 25°C and maximum
UHI setting of 5°C; 28°C is a warmer reference-temperature sensitivity, not an
independently validated 2070 projection. This is honest alignment of the two
figures' roles, not numerical calibration between their temperatures.

Do not silently replace 25°C with the Figure 1 London mean: a city-average
temperature is not a rural reference temperature. Nor should extra UHI warming
be added to Figure 1 without a justified coupling procedure.

## 1. Evidence and scope

Reviewed the latest supplied `London urban cooling-ms-0806.docx` (modified
11 November) and `London urban cooling-SM.docx` (modified 10 November), their
accepted-change text and the embedded Figure 3 panels. Local evidence:

- [Climate notebook](../../code/gcm-data-clip-stats-viz.Rmd): July 2021 CHELSA
  baseline, future WorldClim borough CSVs, then model means per borough.
- [Legacy UCM preparation](../../code/invest_result_zonal_viz_0_data_prep.Rmd):
  temperature/UHI strings mapped to year labels, around lines 285–296.
- [Figure 3 notebook](../../code/invest_result_zonal_viz_1_temp.Rmd): reads
  `data/ucm_output_shp_stats_outcomes_by_borough.rds`, computes borough summaries
  and averages year summaries in the Figure 3b plotting function.
- [Current production runner](../../code/Urban_Cooling_Modeling_Runs/run_ucm_scenarios.py):
  defaults 25/28°C reference temperature, 5°C maximum UHI, 45% RH.
- [Revised-run QC](../../code/Urban_Cooling_Modeling_Runs/REVISED_EQUAL_AREA_RUN.md):
  paired 25/28°C fields differ by 3°C within float32 tolerance.
- Installed InVEST 3.20.2 source, `urban_cooling_model/urban_cooling_model.py`,
  `calc_t_air_nomix_op`: `t_ref + (1 - hm) * uhi_max`, followed by air mixing.

## 2. Why the two figures disagree

Figure 1 displays July mean daily maximum temperature, summarized as an
equal-weight mean across 33 boroughs after averaging the GCMs. Its source files
use ACCESS-CM2, IPSL-CM6A-LR and MRI-ESM2-0. Methods incorrectly names
EC-Earth3-Veg in place of ACCESS-CM2; Supplementary Fig. 12 has the correct list.

| Period label | Figure 1 SSP370 / SSP585 means (°C) | Legacy Figure 3 reference / UHI settings (°C) | Legacy baseline-land-use UCM mean (°C) |
|---|---:|---:|---:|
| 2021 | 21.851 / same baseline | 22 / 2 | 23.445 |
| 2030 | 24.823 / 25.103 | 22 / 5 | 25.613 |
| 2050 | 25.718 / 26.065 | 25 / 5 | 28.613 |
| 2070 | 26.641 / 27.314 | 28 / 5 | 31.613 |

The future Figure 1 labels represent 2021–2040, 2041–2060 and 2061–2080,
respectively. Figure 3's labels were assigned to prescribed parameter combinations;
they were not directly calculated from those Figure 1 ensemble means.

The old Figure 3 baseline-land-use warming is 2.168, 5.168 and 8.168°C.
The first step changes only the UHI setting, from 2 to 5°C, while reference
temperature remains 22°C. Its contribution is approximately
`(23.445318 − 22) / 2 × (5 − 2) = 2.167977°C`.
The later steps add 3°C and 6°C in prescribed reference temperature. Thus the
apparent trajectory contains an imposed UHI amplification, not only projected
background warming. The numerical decomposition reproduces the saved means.

In contrast, Figure 1's differences from its 2021 baseline are 2.973/3.252°C,
3.868/4.214°C and 4.791/5.464°C. The figures do not estimate the same quantity.

## 3. UHI source and interpretation

InVEST computes unmixed temperature as `T_ref + (1 − HM) × UHI_max`, then
spatially mixes the result. Its reference is rural air temperature, and its
UHI parameter is an urban–rural air-temperature contrast. The public
[InVEST guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest/3.20.0/userguide/en/urban_cooling_model.html#air-temperature-estimates)
also lists the Yale explorer as a possible source; that suggestion does not
make surface and air-temperature measures equivalent. The installed 3.20.2
equation was checked directly; the accessible web guide is version 3.20.0.

The cited [Yale explorer](https://geospatial.yale.edu/global-surface-uhi-explorer)
uses satellite land-surface temperature and reports cluster-averaged UHI
statistics. This audit does not establish which London extraction, season,
year or statistic yielded 2 and 5°C, or validate them against air measurements.
The manuscript's “average and maximum” values do not establish a temporal
increase from present to future. Treat the 5°C setting as a prescribed scenario
assumption until its air-temperature relevance is demonstrated.

This matters beyond labels: with heat-mitigation fields and mixing held fixed,
the land-use temperature contrast scales with `UHI_max`; changing reference
temperature alone cancels from the contrast. A 2°C versus 5°C UHI contrast
would therefore scale the temperature benefit by about 0.4, under those fixed
conditions. This is an algebraic sensitivity illustration, not a new calibrated
result. Health responses and work capacity need their own calculations if
parameters change; do not scale every downstream outcome by the same factor.

## 4. Approved Figure 3 design and alternative

**Implemented, limited-change option:** keep the number Figure 3 and a two-panel
structure, but replace the ambiguous calendar-year panel.

- **a:** London-boundary area-weighted mean temperature change for all nine
  scenarios, relative to matched baseline land use. Use the same common valid
  cell-centre mask as the current Figure 4 work-capacity summary. Calculate
  these values directly, rather than substituting unweighted borough means.
  No model-uncertainty bars for deterministic temperature outputs.
- **b:** borough distributions of those matched land-use temperature changes,
  with each borough represented equally. Label spread as geographic variation,
  not parameter/model uncertainty. This provides the spatial context for Fig. 5.

Use the canonical scenario palette, versioned PNG outputs, and the existing
temperature notebook as the user-facing entry point. Preserve the old figure.
This option requires aggregation/plotting, not a new UCM run. The scientific
framing must acknowledge that projected climate motivates the prescribed warm
conditions but is not an explicitly coupled SSP-by-year UCM ensemble.

**If a calendar-year projection panel is essential:** first define a rural
reference area or defensible station set; establish matching temperature
variables and baseline climatology; derive SSP/period anomalies using compatible
baseline/future data; add these to the rural reference; and separately justify
air-UHI settings for the same time of day. Hold UHI fixed in the primary temporal
comparison unless change is supported, and vary it in sensitivity analysis.
Then run/validate the resulting configurations and propagate changes to relevant
downstream outcomes. This is a model-design revision, not merely relabelling.
Do not present Figure 1's urban mean directly as `T_ref`, or the old 22/25/28°C
settings as exact SSP forecasts.

## 5. Verified revised land-use contrasts

The new summaries below are paired raster contrasts on a common London mask.
The unweighted borough mean gives each borough equal weight; the London mean
weights each borough by its valid cell area. Only the latter is used in panel a.
The mixed production layout stores some 25°C outputs in `03_energy_work_capacity`
and others in `01_temperature_invest3202`; both are stages of the same revised run.

| Scenario | Original panel b: borough/period mean (°C) | Revised borough mean (°C) | Revised London-area-weighted mean (°C) |
|---|---:|---:|---:|
| AllBuilt | +1.0521 | +1.0521 | +1.1088 |
| TreeRisk | +0.0970 | +0.0970 | +0.0764 |
| TreeOpp | −0.0588 | −0.0588 | −0.0624 |
| Green10 | −0.1618 | −0.1202 | −0.1021 |
| Target10 | −0.1274 | −0.1410 | −0.1134 |
| Green20 | −0.2671 | −0.2264 | −0.1954 |
| Target20 | −0.2360 | −0.2583 | −0.2153 |
| Green30 | −0.3738 | −0.3364 | −0.2910 |
| Target30 | −0.3668 | −0.3780 | −0.3192 |

The old +1.05°C remains close to the revised AllBuilt borough mean, but is not
a maximum borough change. The old −0.37°C Green30 claim is outdated. Do not
replace it with −0.38°C in a citywide-area-weighted Abstract; the correct revised
Target30 area-weighted contrast is −0.32°C. The old notebook's `mean ± 1.96 SD/√33` describes borough
variation under independence assumptions, not propagated model uncertainty;
averaging its year-specific bounds does not repair that interpretation.

## 6. Figure 1 and manuscript alignment still needed

1. Correct the GCM list to match the actual files. Use “mean across boroughs,”
   not an area-weighted citywide mean. Consistently define July mean daily Tmax.
2. Describe Figure 1 as future WorldClim climatologies compared with July 2021
   CHELSA estimates. A single year and a different product are not a common
   long-term baseline; the differences are not automatically harmonized climate
   anomalies. A climatological anomaly interpretation needs additional checks.
3. Replace Figure 3's claims of warming “across future periods” with conditional
   land-use effects. Replace unqualified “mid-century predictions” with an
   explicit statement that warm-condition parameters were prescribed to explore
   adaptation benefits. Do not imply this fixes parameter-validation gaps.
4. Keep the [Figures 4–5 audit](figure4_5_audit.md) as the source for their numbers
   and legends. Its author-completed status does not resolve the UHI justification.
5. Before submission, retrieve the exact UHI extraction and assess compatible
   air-temperature evidence. A small UHI sensitivity is advisable if calibration
   evidence is unavailable, but no new runs are authorized by this audit.

### Proposed framing (for review, not inserted in Word)

> Figure 1 characterizes projected July temperature exposure. Separately, we
> evaluated land-use effects under prescribed warm conditions using a rural
> reference air temperature of 25°C and a maximum UHI setting of 5°C. These
> parameters were held constant across land-use scenarios, isolating their
> modelled temperature effects. The assessment is conditional on these settings
> and does not represent a directly coupled SSP-specific projection of urban
> temperatures. A 28°C reference-temperature run provides a warmer-condition
> sensitivity rather than a calibrated projection for a particular year.

## Remaining author review

Review the new PNG and copy-ready legend/Results above, then replace the old
panels and temperature claims in Word. The separation of roles is now implemented
in Figure 3, but Figure 1 and Word remain unchanged. Resolve the UHI source and
air-temperature applicability described above before making calibrated climate
projection claims. No additional sensitivity model runs are implied by this update.

## Recreate the HTML

From the repository root:

```sh
pandoc docs/manuscript/figure3_climate_uhi_audit.md --standalone --embed-resources --toc --css docs/manuscript/figure_review.css --metadata title="Figure 3 climate and UHI audit" -o docs/manuscript/figure3_climate_uhi_audit.html
```
