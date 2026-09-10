#!/usr/bin/env Rscript

# Aggregate the count-preserved 2021 WorldPop raster to the exact LSOA
# geometries used by Figure 7. The output remains keyed to the legacy numeric
# `id` because the serialized analysis layer does not retain LSOA11CD.

required_packages <- c("digest", "jsonlite", "sf", "terra")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop("Install required R packages: ", paste(missing_packages, collapse = ", "))
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(
    "Usage: prepare_lsoa_population_2021.R HEALTH_RDS POPULATION_TIF ",
    "OUTPUT_CSV OUTPUT_MANIFEST_JSON"
  )
}

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(file_argument) != 1L) {
  stop("Could not resolve the preparation script path")
}
script_path <- normalizePath(sub("^--file=", "", file_argument))

health_path <- normalizePath(args[[1]], mustWork = TRUE)
population_path <- normalizePath(args[[2]], mustWork = TRUE)
output_path <- normalizePath(args[[3]], mustWork = FALSE)
manifest_path <- normalizePath(args[[4]], mustWork = FALSE)

if (file.exists(output_path) || file.exists(manifest_path)) {
  stop("Output already exists; remove it only after reviewing the prior result")
}

health <- readRDS(health_path)
if (!inherits(health, "sf") || !all(c("id", "geom") %in% names(health))) {
  stop("Health RDS must be an sf object containing `id` and `geom`")
}
if (anyNA(health$id) || anyDuplicated(sort(unique(health$id))) > 0) {
  stop("Health RDS contains invalid IDs")
}

# Each ID is repeated across model/scenario rows. Retain one exact geometry per
# ID without subsetting the legacy sf object, whose active geometry metadata is
# not compatible with current sf subsetting methods.
first_row <- !duplicated(health$id)
lsoa <- sf::st_sf(
  id = health$id[first_row],
  geometry = health$geom[first_row],
  crs = sf::st_crs(health$geom)
)
lsoa <- lsoa[order(lsoa$id), ]
if (nrow(lsoa) != 4835L || anyDuplicated(lsoa$id)) {
  stop("Expected exactly 4,835 unique Figure 7 LSOA IDs")
}
if (any(sf::st_is_empty(lsoa))) {
  stop("LSOA geometry contains empty features")
}
invalid_before <- which(!sf::st_is_valid(lsoa))
if (length(invalid_before) > 0L) {
  # Eight legacy polygons have ring self-intersections with current GEOS.
  # st_make_valid repairs topology while retaining the same IDs and footprint.
  lsoa <- sf::st_make_valid(lsoa)
}
if (any(!sf::st_is_valid(lsoa))) {
  stop("LSOA geometry remains invalid after deterministic topology repair")
}

population <- terra::rast(population_path)
if (terra::nlyr(population) != 1L) {
  stop("Population raster must contain exactly one band")
}
if (!terra::same.crs(population, terra::vect(lsoa))) {
  stop("Population raster and LSOA geometry must use the same CRS")
}

# Pixel-centre allocation is consistent with the borough mask used to create
# the count-preserved 10 m population raster. LSOAs partition London, so every
# non-zero London pixel should be assigned once apart from boundary precision.
extracted <- terra::extract(
  population,
  terra::vect(lsoa),
  fun = sum,
  na.rm = TRUE,
  touches = FALSE
)
lookup <- data.frame(
  id = lsoa$id,
  population_2021 = as.numeric(extracted[[2]])
)
if (anyNA(lookup$population_2021) || any(lookup$population_2021 <= 0)) {
  stop("Every LSOA must have a positive, non-missing population")
}

raster_total <- terra::global(population, "sum", na.rm = TRUE)[1, 1]
lsoa_total <- sum(lookup$population_2021)
allocation_difference <- lsoa_total - raster_total
allocation_relative_difference <- allocation_difference / raster_total
if (abs(allocation_relative_difference) > 0.001) {
  stop(
    sprintf(
      "LSOA allocation differs from raster total by %.3f%%",
      100 * allocation_relative_difference
    )
  )
}

ons_ts001_london_total <- 8799776
benchmark_relative_difference <-
  (lsoa_total - ons_ts001_london_total) / ons_ts001_london_total

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(lookup, output_path, row.names = FALSE, quote = FALSE)

git_value <- function(arguments) {
  result <- suppressWarnings(system2("git", arguments, stdout = TRUE, stderr = FALSE))
  if (!identical(attr(result, "status"), NULL) && attr(result, "status") != 0) {
    return(NULL)
  }
  paste(result, collapse = "\n")
}
repository_root <- normalizePath(file.path(dirname(script_path), "..", ".."))
old_working_directory <- setwd(repository_root)
on.exit(setwd(old_working_directory), add = TRUE)
git_status <- git_value(c("status", "--porcelain"))

manifest <- list(
  schema_version = 1,
  method = "sum count-preserved WorldPop 2021 pixels by Figure 7 LSOA polygon",
  geography = list(
    count = nrow(lsoa),
    vintage = "LSOA 2011",
    identifier = "legacy numeric id from health_sf.rds",
    crs = sf::st_crs(lsoa)$input,
    allocation_rule = "raster pixel centre falls inside polygon",
    invalid_geometry_ids_repaired = lsoa$id[invalid_before]
  ),
  validation = list(
    population_raster_total = unname(raster_total),
    lsoa_population_total = unname(lsoa_total),
    allocation_difference = unname(allocation_difference),
    allocation_relative_difference = unname(allocation_relative_difference),
    ons_census_2021_ts001_london_total = ons_ts001_london_total,
    benchmark_relative_difference = unname(benchmark_relative_difference),
    benchmark_role = "external London-total reasonableness check; not directly joined"
  ),
  inputs = list(
    health_rds = list(path = health_path, sha256 = digest::digest(health_path, "sha256", file = TRUE)),
    population_raster = list(
      path = population_path,
      sha256 = digest::digest(population_path, "sha256", file = TRUE)
    )
  ),
  output = list(
    path = output_path,
    sha256 = digest::digest(output_path, "sha256", file = TRUE)
  ),
  software = list(
    r = R.version.string,
    sf = as.character(utils::packageVersion("sf")),
    terra = as.character(utils::packageVersion("terra")),
    git_commit = git_value(c("rev-parse", "HEAD")),
    git_worktree_dirty = !is.null(git_status) && nzchar(git_status),
    preparation_script_sha256 = digest::digest(
      script_path, "sha256", file = TRUE
    )
  )
)
jsonlite::write_json(manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("Created %s with %s LSOAs\n", output_path, format(nrow(lookup), big.mark = ",")))
cat(sprintf("LSOA total: %s\n", formatC(lsoa_total, format = "f", digits = 2, big.mark = ",")))
cat(sprintf("Difference from TS001 London total: %+.3f%%\n", 100 * benchmark_relative_difference))
cat(sprintf("Manifest: %s\n", manifest_path))
