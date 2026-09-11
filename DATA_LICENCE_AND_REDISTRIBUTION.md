# Data licence and redistribution review

Reviewed: 11 September 2026

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
| UKCEH Land Cover Map 2023 | scenario rasters; UCM temperatures; downstream health tables and figures | The 10 m GB raster uses the UKCEH Land Cover Map Raster licence. It permits non-commercial publications/reports and non-substitute derived products under a non-commercial licence, but prohibits placing the source Data on an internet site or redistributing it | Cite the dataset DOI and add the required UKCEH acknowledgement. Keep raw/clipped LCM, scenario and UCM rasters out of Git. Before placing machine-readable derived products under the repository's general licence, obtain written confirmation from UKCEH or give those files an explicit compatible non-commercial notice |

Official references:

- [WorldPop redistribution and attribution FAQ](https://www.worldpop.org/faq/)
- [Nomis copyright and OGL guidance](https://www.nomisweb.co.uk/home/copyright.asp)
- [ONS digital-boundary terms and required attribution](https://www.ons.gov.uk/methodology/geography/geographicalproducts/digitalboundaries)
- [GLA/Bloomberg Climate Risk Mapping dataset and licence](https://data.london.gov.uk/dataset/climate-risk-mapping-2oxg6)
- [UKCEH LCM2023 collection](https://catalogue.ceh.ac.uk/documents/73ecb85e-c55a-4505-9c39-526b464e1efd)
- [LCM2023 10 m classified pixels, GB (DOI 10.5285/7727ce7d-531e-4d77-b756-5cc59ff016bd)](https://doi.org/10.5285/7727ce7d-531e-4d77-b756-5cc59ff016bd)
- [UKCEH Land Cover Map Raster licence](https://eidc.ac.uk/licences/lcm-raster/plain)

## UKCEH licence decision and remaining confirmation

The catalogue link supplied on 11 September 2026 identifies the collection,
and its GB 10 m resource identifies the applicable **Land Cover Map Raster**
licence. The dataset citation is:

> Morton, R.D.; Marston, C.G.; O'Neil, A.W.; Rowland, C.S. (2024). Land Cover
> Map 2023 (10m classified pixels, GB). NERC EDS Environmental Information
> Data Centre. https://doi.org/10.5285/7727ce7d-531e-4d77-b756-5cc59ff016bd

The licence requires the acknowledgement `Data owned by UK Centre for Ecology
& Hydrology` and `© Database Right/Copyright UKCEH` on images of the Data. It
allows non-commercial publications and reports based on the Data, while the
Data itself may not be posted or redistributed. Derived products must not act
as a substitute and access must be restricted to non-commercial use.

This resolves the dataset and standard-licence identification. One narrower
question remains because this repository's general licence may permit uses
beyond the UKCEH derived-product conditions:

> May the project's small machine-readable aggregate tables and rendered
> publication figures be distributed through a public code repository if they
> carry a separate non-commercial UKCEH-derived-output notice and all source,
> clipped, scenario and modeled raster layers are withheld?

The repository currently does **not** track the source/clipped LCM2023 raster,
the Target/Green scenario rasters or the UCM temperature rasters. Keep that
policy until the licence response is recorded.

## Other tracked files needing provenance review

These are not needed for the revised Figure 7 run, but a repository-wide
redistribution audit cannot close until their sources and terms are recorded:

- `data/London_Ward_aoi.{shp,shx,dbf,prj,...}` — the embedded metadata traces
  `London-wards-2018_ESRI`, and the one-feature dissolved mask has the same
  EPSG:27700 extent as the GLA London Wards layer. This is strong evidence that
  it derives from the GLA's `London-wards-2018.zip`, but no source checksum is
  available for an exact identity test. Cite the
  [GLA Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-for-london-20od9),
  record the transformation as a dissolve to one London mask, and use the
  source page's required statements: `Contains National Statistics data ©
  Crown copyright and database right [2015]` and `Contains Ordnance Survey
  data © Crown copyright and database right [2015]`. The dataset page lists
  Open Government Licence v2.
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
