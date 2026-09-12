#!/usr/bin/env Rscript

# Build borough-level co-benefit maps from the same revised scenario run used
# for Figure 4. These maps describe spatial heterogeneity; they are not a
# substitute for the London-wide totals reported in Figure 4.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(sf)
  library(terra)
  library(tidyr)
})

parse_args <- function(values) {
  result <- list(temperature = 25)
  index <- 1
  while (index <= length(values)) {
    key <- values[[index]]
    if (!startsWith(key, "--") || index == length(values)) {
      stop("Arguments must be supplied as --name value pairs", call. = FALSE)
    }
    result[[gsub("-", "_", substring(key, 3))]] <- values[[index + 1]]
    index <- index + 2
  }
  result
}

required_path <- function(value, label, directory = FALSE) {
  if (is.null(value)) stop(sprintf("Missing --%s", gsub("_", "-", label)), call. = FALSE)
  path <- normalizePath(value, mustWork = FALSE)
  exists <- if (directory) dir.exists(path) else file.exists(path)
  if (!exists) stop(sprintf("%s not found: %s", label, path), call. = FALSE)
  path
}

scenario_order <- c("green10", "target10", "green20", "target20", "green30", "target30")
scenario_labels <- c(
  green10 = "Green 10%", target10 = "Target 10%",
  green20 = "Green 20%", target20 = "Target 20%",
  green30 = "Green 30%", target30 = "Target 30%"
)

args <- parse_args(commandArgs(trailingOnly = TRUE))
borough_summary_path <- required_path(args$borough_summary, "borough_summary")
health_root <- required_path(args$health_root, "health_root", directory = TRUE)
borough_vector_path <- required_path(args$borough_vector, "borough_vector")
output_dir <- normalizePath(args$output_dir, mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
temperature <- as.numeric(args$temperature)
if (!is.finite(temperature)) stop("--temperature must be numeric", call. = FALSE)
temperature_label <- format(temperature, scientific = FALSE, trim = TRUE)

boroughs <- st_read(borough_vector_path, quiet = TRUE) %>%
  select(borough = NAME, borough_code = GSS_CODE)
if (nrow(boroughs) != 33 || anyDuplicated(boroughs$borough_code)) {
  stop("Expected 33 uniquely coded London borough features", call. = FALSE)
}

summary_data <- read.csv(
  borough_summary_path, stringsAsFactors = FALSE, check.names = FALSE
) %>% filter(temperature_c == temperature)
baseline <- summary_data %>%
  filter(scenario == "baseline") %>%
  select(
    borough_code,
    baseline_energy = energy_savings,
    baseline_workability = workability_fraction
  )

# Match the manuscript-era borough estimand: subtract the baseline within each
# borough, then retain every borough separately for mapping.
borough_change <- summary_data %>%
  filter(scenario %in% scenario_order) %>%
  inner_join(baseline, by = "borough_code") %>%
  transmute(
    borough, borough_code, scenario,
    energy_change_million_gbp = (energy_savings - baseline_energy) / 1e6,
    productivity_change_pp =
      round(workability_fraction * 100, 2) -
      round(baseline_workability * 100, 2)
  )
if (nrow(borough_change) != 33 * length(scenario_order)) {
  stop("Energy/productivity input must contain 33 boroughs for all six scenarios", call. = FALSE)
}

# Health rasters already encode scenario-minus-baseline excess mortality.
# Assign each 10 m cell to its borough by cell centre and sum the values; the
# sign is reversed so positive values mean deaths averted.
health_change <- lapply(scenario_order, function(scenario) {
  raster_path <- file.path(
    health_root,
    sprintf("%s_%sc", scenario, temperature_label),
    "Excess_all_cause.tif"
  )
  required_path(raster_path, sprintf("%s health raster", scenario))
  health_raster <- rast(raster_path)
  borough_for_raster <- st_transform(boroughs, crs(health_raster))
  totals <- terra::extract(
    health_raster, vect(borough_for_raster), fun = sum, na.rm = TRUE
  )[[2]]
  if (length(totals) != nrow(boroughs) || any(!is.finite(totals))) {
    stop(sprintf("Invalid borough health totals for %s", scenario), call. = FALSE)
  }
  data.frame(
    borough_code = boroughs$borough_code,
    scenario = scenario,
    deaths_averted = -totals,
    health_raster = raster_path
  )
}) %>% bind_rows()

map_data <- borough_change %>%
  left_join(health_change, by = c("borough_code", "scenario")) %>%
  pivot_longer(
    c(energy_change_million_gbp, productivity_change_pp, deaths_averted),
    names_to = "metric", values_to = "value"
  ) %>%
  mutate(
    scenario = factor(scenario, levels = scenario_order, labels = scenario_labels),
    metric = recode(
      metric,
      energy_change_million_gbp = "Avoided energy cost",
      productivity_change_pp = "Heavy-work capacity gain",
      deaths_averted = "Heat-related deaths averted"
    ),
    unit = recode(
      metric,
      `Avoided energy cost` = "£ million/month per borough",
      `Heavy-work capacity gain` = "Percentage points",
      `Heat-related deaths averted` = "Annual deaths"
    )
  )

write.csv(
  map_data %>%
    arrange(metric, scenario, borough_code) %>%
    transmute(
      borough, borough_code,
      scenario = as.character(scenario), metric, value, unit
    ),
  file.path(output_dir, "figure5_borough_cobenefits_data.csv"),
  row.names = FALSE
)

map_sf <- boroughs %>%
  left_join(map_data, by = c("borough", "borough_code"))

make_map_row <- function(metric_name, palette, legend_title) {
  ggplot(filter(map_sf, metric == metric_name)) +
    geom_sf(aes(fill = value), colour = "white", linewidth = 0.12) +
    facet_wrap(~scenario, nrow = 1) +
    scale_fill_distiller(palette = palette, direction = 1, name = legend_title) +
    coord_sf(datum = NA) +
    labs(title = metric_name) +
    theme_void(base_size = 9) +
    theme(
      plot.title = element_text(face = "bold", size = 10, margin = margin(b = 3)),
      strip.text = element_text(face = "bold", size = 8),
      strip.background = element_rect(fill = "#F1F1F1", colour = NA),
      legend.position = "right",
      legend.key.height = grid::unit(0.45, "in"),
      plot.margin = margin(2, 2, 2, 2)
    )
}

combined_plot <-
  make_map_row("Avoided energy cost", "YlOrBr", "£m/month") /
  make_map_row("Heavy-work capacity gain", "PuBu", "Percentage\npoints") /
  make_map_row("Heat-related deaths averted", "YlGnBu", "Annual\ndeaths") +
  plot_annotation(
    title = "Borough-level co-benefits of alternative land-use scenarios",
    subtitle = paste0(
      "Scenario-minus-baseline changes under ", temperature_label,
      "°C mid-century climate conditions"
    ),
    caption = paste(
      "Each map shows a borough-specific change; colours are comparable across",
      "the six scenarios within a row. Health maps are deterministic spatial",
      "allocations and do not show borough-level uncertainty."
    ),
    theme = theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10),
      plot.caption = element_text(hjust = 0, size = 8, colour = "#4A4A4A")
    )
  )

for (extension in c("png", "pdf", "svg")) {
  ggsave(
    file.path(output_dir, paste0("figure5_borough_cobenefits.", extension)),
    combined_plot, width = 13.2, height = 8.6, units = "in", dpi = 400,
    bg = "white", limitsize = FALSE
  )
}

artifact_names <- c(
  "figure5_borough_cobenefits_data.csv",
  "figure5_borough_cobenefits.png",
  "figure5_borough_cobenefits.pdf",
  "figure5_borough_cobenefits.svg"
)
manifest <- list(
  schema_version = 1,
  created_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  temperature_c = temperature,
  estimand = "Scenario-minus-baseline change calculated separately within each borough",
  health_zonal_rule = "10 m raster-cell centre assigned to borough; values summed and sign reversed",
  uncertainty = "No borough confidence intervals; mapped variation is spatial heterogeneity",
  inputs = list(
    borough_summary = list(
      path = borough_summary_path,
      sha256 = digest::digest(file = borough_summary_path, algo = "sha256")
    ),
    borough_vector = list(
      path = borough_vector_path,
      sha256 = digest::digest(file = borough_vector_path, algo = "sha256")
    ),
    health_root = health_root
  ),
  outputs = setNames(lapply(artifact_names, function(name) {
    path <- file.path(output_dir, name)
    list(path = name, sha256 = digest::digest(file = path, algo = "sha256"))
  }), artifact_names)
)
write_json(
  manifest,
  file.path(output_dir, "figure5_borough_cobenefits_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE
)
message("Created borough co-benefit maps in ", output_dir)
