# Data licence and redistribution review

Reviewed: 10 September 2026

This review distinguishes public source data, derived project outputs and
large external inputs that are intentionally excluded from Git. It is a
reproducibility record, not legal advice.

## Figure 7 and health workflow

| Source | Files or outputs affected | Published terms | Action |
|---|---|---|---|
| WorldPop 2021 population | `data/derived/lsoa_population_2021*.csv`; population-weighted health and Figure 7 outputs | CC BY 4.0 permits sharing and adaptation with attribution | Add the dataset-specific citation/DOI from WorldPop item 75899 to the manuscript and derived-data README |
| ONS mortality through Nomis | population-weighted health and Figure 7 outputs | Open Government Licence; credit ONS | Retain `Source: Office for National Statistics` in data documentation |
| ONS LSOA boundaries | `data/derived/fig7_vulnerability_lsoa11*.gpkg`, crosswalks and maps | Open Government Licence v3.0; ONS specifies an ONS source statement and OS copyright/database-right statement | Add the required boundary attribution with the source-data year |
| GLA/Bloomberg Climate Risk Mapping vulnerability variables | `data/derived/fig7_vulnerability_lsoa11*.gpkg`, `svi_lsoa11_crosswalk*.csv` and Figure 7 | London Datastore identifies the dataset as Open Government Licence v3 | Credit Greater London Authority and Bloomberg Associates and link the 2024 dataset/methodology |
| UKCEH Land Cover Map 2023 | scenario rasters; UCM temperatures; downstream health tables and figures | Product-specific licence terms apply. UKCEH states that publication rights depend on whether an output is considered Derived Data and recommends asking its Licensing Team when uncertain | **User confirmation required:** locate the licence accepted when LCM2023 was obtained and confirm whether the intended Git/manuscript outputs may be redistributed. Until resolved, do not add raw/clipped LCM, scenario or UCM rasters to Git |

Official references:

- [WorldPop redistribution and attribution FAQ](https://www.worldpop.org/faq/)
- [Nomis copyright and OGL guidance](https://www.nomisweb.co.uk/home/copyright.asp)
- [ONS digital-boundary terms and required attribution](https://www.ons.gov.uk/methodology/geography/geographicalproducts/digitalboundaries)
- [GLA/Bloomberg Climate Risk Mapping dataset and licence](https://data.london.gov.uk/dataset/climate-risk-mapping-2oxg6)
- [UKCEH LCM2023 catalogue and licence notice](https://catalogue.ceh.ac.uk/documents/68712ac3-d740-41df-bcb3-4d341f859909)

## Exact UKCEH confirmation requested

Please provide either the LCM2023 licence file, the order/download email, or
the licence name shown in the UKCEH portal for the copy used to create
`LCM2023_London_10m_clip2aoi_tcc24.tif`. The specific question for UKCEH is:

> May we publicly redistribute aggregated tables, modeled temperature and
> mortality summaries, and publication figures derived from LCM2023, while
> withholding all source, clipped and scenario LCM rasters? If so, what exact
> attribution and licence notice must accompany them?

The repository currently does **not** track the source/clipped LCM2023 raster,
the Target/Green scenario rasters or the UCM temperature rasters. Keep that
policy until the licence response is recorded.

## Other tracked files needing provenance review

These are not needed for the revised Figure 7 run, but a repository-wide
redistribution audit cannot close until their sources and terms are recorded:

- `data/London_Ward_aoi.{shp,shx,dbf,prj,...}` — the embedded metadata only
  traces a local `London-wards-2018_ESRI` file; it does not identify its
  publisher, download URL or licence. Add the original source record. If it is
  an ONS boundary, apply the ONS/OS attribution above.
- `data/weather_station_metadata.rds` and
  `data/london_weather_obs_stat.RDS` — `code/weather_obs.Rmd` traces these to
  the MIDAS Open UK hourly weather observations archive, but the exact archive
  version, download URL and licence are not recorded. Add the source README or
  catalogue record before treating these binary extracts as redistributable.

## Minimum attribution checklist before release

1. Add source, dataset title, version/year, stable URL/DOI and licence for every
   public input to `data/derived/README.md`.
2. Include modification wording for transformed WorldPop, GLA and ONS data.
3. Add the current ONS and Ordnance Survey copyright year required for the
   boundary product actually used.
4. Record the UKCEH determination and exact acknowledgement text.
5. Recheck the Git-tracked data inventory before a public release; do not infer
   that an output is unrestricted merely because it is highly processed.
