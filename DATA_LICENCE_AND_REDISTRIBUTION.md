# Data licence and redistribution review

Reviewed: 4 November 2026 (rendered-report scope clarified; other provenance review dated 1 November)

This review distinguishes public source data, derived project outputs and
large external inputs that are intentionally excluded from Git. It is a
reproducibility record, not legal advice.

## Release decision at this review

- **Code and documentation:** the repository licence does not sublicense
  third-party data. Preserve source-specific notices rather than describing
  every repository file as GPL-covered.
- **UKCEH LCM2023:** the current [Land Cover Map Raster
  licence](https://eidc.ac.uk/licences/lcm-raster/plain) permits non-commercial
  publications/reports based on the Data, with required citation and image
  acknowledgement. It prohibits putting the source Data on an internet site.
  Access to non-substitute Derived Products and Services must be restricted to
  non-commercial use. Keep raw/clipped LCM, scenario and UCM rasters out of
  Git. Do **not** infer that the current public CSV/GPKG release is
  cleared by the repository's code licence alone; obtain the specific
  written determination below before adding more machine-readable outputs or
  declaring the public-release gate closed.
- **ONS/Nomis and GLA climate-risk variables:** the official source pages
  identify Open Government Licence terms, subject to the source and boundary
  attribution below. WorldPop remains CC BY 4.0 with dataset-specific credit.
- **MIDAS Open extracts:** the source notebook points to the v202407 archive,
  whose [CEDA record](https://catalogue.ceda.ac.uk/uuid/c50776e4903942cdb329589da70b83fe/)
  says OGL v3 and requires its dataset citation. The exact raw files used to
  create the two tracked RDS extracts have not been independently matched;
  this is a provenance task, not evidence that the extracts came from the
  restricted full MIDAS collection.

No licensing contact or public-release approval was made during this review.
Existing Git history may already contain derived outputs; this review does
not retrospectively clear those files or erase them.

## Rendered report figures: review 4 November 2026

The prior blanket hold on the SVI report's illustrations was too broad.
The [applicable licence](https://eidc.ac.uk/licences/lcm-raster/plain), linked
by the GB 10 m dataset record, explicitly permits non-commercial/personal-use
publications and reports on third-party internet sites (clause 3.3).
Clauses 4.1 and 4.3 prohibit distribution of the Data itself; clause 4.4
expressly preserves the exceptions in 3.3–3.5. Clauses 5.2 and 6 require
acknowledgement and citation.

Our assessment is that `docs/manuscript/SVI_SPATIAL_RESULTS.md` and its
12 PNG illustrations constitute such a research report: a vulnerability
quintile-count chart, a model-contrast chart, five fitted-curve plots and five
LSOA model-residual maps. They do not reproduce the source land-cover raster
or allow recovery of its pixel classifications. The SVI-only count chart
does not use UKCEH land-cover values at all. Release these rendered report
illustrations with the attribution and reuse notice in the report; do not
apply the repository's GPL code licence to third-party rights. This is a
documented reading of the standard terms, not written UKCEH approval or legal
advice. No separate written approval prerequisite is stated for clause 3.3.

This narrow release does not clear CSV/GPKG derivatives, source/clipped LCM,
scenario or UCM rasters, commercial services, or a general open-data licence.
Keep the machine-readable derivative question below open. Standalone reuse
must retain the report's source credits and respect the applicable terms.

## Figure 7 and health workflow

| Source | Files or outputs affected | Published terms | Action |
|---|---|---|---|
| WorldPop 2021 population | `data/derived/lsoa_population_2021*.csv`; population-weighted health and Figure 7 outputs | CC BY 4.0 permits sharing and adaptation with attribution | Add the dataset-specific citation/DOI from WorldPop item 75899 to the manuscript and derived-data README |
| ONS mortality through Nomis | population-weighted health and Figure 7 outputs | Open Government Licence; credit ONS | Retain `Source: Office for National Statistics` in data documentation |
| GLA-hosted ONS LSOA11 boundaries | `data/derived/fig7_vulnerability_lsoa11*.gpkg`, crosswalks and maps | The exact archived GLA shapefile is listed under OGL v2; its page prescribes 2015 National Statistics and OS map credits | Use the exact credits documented in `data/derived/README.md` |
| GLA/Bloomberg Climate Risk Mapping vulnerability variables | `data/derived/fig7_vulnerability_lsoa11*.gpkg`, `svi_lsoa11_crosswalk*.csv` and Figure 7 | London Datastore identifies the dataset as Open Government Licence v3 | Credit Greater London Authority and Bloomberg Associates and link the 2024 dataset/methodology |
| UKCEH Land Cover Map 2023 | scenario rasters; UCM temperatures; downstream health tables and figures | The 10 m GB raster uses the UKCEH Land Cover Map Raster licence. It permits non-commercial publications/reports and non-substitute derived products subject to non-commercial-use restrictions, but prohibits posting or redistributing source Data | Cite the dataset DOI and add the required UKCEH acknowledgement. Keep raw/clipped LCM, scenario and UCM rasters out of Git. Obtain written guidance before assigning a separate notice to public machine-readable derivatives; do not assume a notice alone cures an incompatible release |

Official references:

- [WorldPop redistribution and attribution FAQ](https://www.worldpop.org/faq/)
- [Nomis copyright and OGL guidance](https://www.nomisweb.co.uk/home/copyright.asp)
- [ONS digital-boundary terms and required attribution](https://www.ons.gov.uk/methodology/geography/geographicalproducts/digitalboundaries)
- [GLA Statistical GIS Boundary Files for London and required 2015 map credits](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-for-london-20od9)
- [GLA/Bloomberg Climate Risk Mapping dataset and licence](https://data.london.gov.uk/dataset/climate-risk-mapping-2oxg6)
- [CEDA MIDAS Open v202407 record and OGL v3](https://catalogue.ceda.ac.uk/uuid/c50776e4903942cdb329589da70b83fe/)
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
question remains because the repository's public code licence and open GitHub
distribution may permit uses beyond the UKCEH derived-product conditions:

> May the project's small machine-readable LSOA/borough/city aggregate tables,
> geospatial derivative files be distributed
> through a public GitHub code repository under a separate non-commercial
> UKCEH-derived-output notice, while all source/clipped LCM, scenario and UCM
> rasters are withheld? Which specific output types require a separate licence
> or written approval, and what exact attribution/notice should accompany them?

The author or data steward can send this question, the LCM2023 GB 10 m DOI,
the repository URL and representative output schemas to the UKCEH Data
Licensing team at `spatialdata@ceh.ac.uk` (listed in the licence). Record the
reply and any conditions here before changing the machine-readable release classification.

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
- `data/weather_station_metadata.rds` (SHA-256
  `0dfe04699189766dd3be6806b9cada9aac625becfd8807fa5b6a78428194b873`)
  and `data/london_weather_obs_stat.RDS` (SHA-256
  `cb1023f6b4e9691171cfe096feb2b46f50125c90c688364ec4bfa4e603d582d7`)
  — `code/weather_obs.Rmd` names v202407 station metadata and
  `dataset-version-202407/` observations. The CEDA v202407 record is DOI
  `10.5285/c50776e4903942cdb329589da70b83fe`, OGL v3, superseded but
  citable. The notebook also reads an `E:/London/uk_weather/` working copy;
  its exact source-file list and checksums are not saved. Match that copy to
  the v202407 archive before claiming byte-level recreation of either RDS.

## Minimum attribution checklist before release

1. Add source, dataset title, version/year, stable URL/DOI and licence for every
   public input to `data/derived/README.md`.
2. Include modification wording for transformed WorldPop, GLA and ONS data.
3. Use the GLA source page's 2015 National Statistics and OS map credits for
   the exact LSOA11 archive matched to the Figure 7 manifest.
4. For rendered report figures, apply the review above and the required
   acknowledgement; for machine-readable derivatives, record UKCEH's determination.
5. Recheck the Git-tracked data inventory before a public release; do not infer
   that an output is unrestricted merely because it is highly processed.
