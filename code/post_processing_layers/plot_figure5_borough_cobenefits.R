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

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot resolve the Figure 5 script path", call. = FALSE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_argument[[1]])))
source(file.path(script_dir, "figure4_panel_helpers.R"))

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

scenario_order <- figure4_scenarios
scenario_labels <- figure4_labels
counterfactual_scenarios <- c("allbuilt", "treerisk", "treeopp")
canopy_scenarios <- setdiff(scenario_order, counterfactual_scenarios)

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
  stop("Energy/productivity input must contain 33 boroughs for all nine scenarios", call. = FALSE)
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
    scenario_id = scenario,
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
      scenario = scenario_id,
      scenario_label = gsub("\\n", " ", as.character(scenario)),
      metric, value, unit
    ),
  file.path(output_dir, "figure5_borough_cobenefits_data.csv"),
  row.names = FALSE
)

map_sf <- boroughs %>%
  left_join(map_data, by = c("borough", "borough_code"))

make_map_row <- function(metric_name, scenario_ids, legend_title, show_legend) {
  # Use the same symmetric range for both scenario groups. This preserves
  # direct colour comparability and makes zero the neutral midpoint.
  metric_limit <- max(abs(filter(map_sf, metric == metric_name)$value), na.rm = TRUE)
  plot_data <- filter(
    map_sf, metric == metric_name, scenario_id %in% scenario_ids
  ) %>%
    mutate(scenario = droplevels(scenario))

  ggplot(plot_data) +
    geom_sf(aes(fill = value), colour = "white", linewidth = 0.12) +
    facet_wrap(~scenario, nrow = 1) +
    scale_fill_gradient2(
      low = "#B2182B", mid = "#F7F7F7", high = "#2166AC",
      midpoint = 0, limits = c(-metric_limit, metric_limit),
      name = legend_title,
      guide = guide_colourbar(
        title.position = "top", title.hjust = 0.5,
        barheight = grid::unit(0.68, "in"),
        barwidth = grid::unit(0.16, "in")
      )
    ) +
    coord_sf(datum = NA) +
    labs(title = metric_name) +
    theme_void(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 10, margin = margin(b = 3)),
      strip.text = element_text(face = "bold", size = 10),
      strip.background = element_rect(fill = "#F1F1F1", colour = NA),
      legend.position = if (show_legend) "right" else "none",
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 9),
      plot.margin = margin(2, 2, 2, 2)
    )
}

metric_specs <- list(
  c("Avoided energy cost", "£m/month"),
  c("Heavy-work capacity gain", "Percentage\npoints"),
  c("Heat-related deaths averted", "Annual\ndeaths")
)

make_group_heading <- function(label) {
  ggplot() +
    annotate("text", x = 0, y = 0.5, label = label, hjust = 0,
             fontface = "bold", size = 3.9) +
    xlim(0, 1) + ylim(0, 1) +
    theme_void() +
    theme(plot.margin = margin(0, 2, 0, 2))
}

counterfactual_rows <- wrap_plots(lapply(metric_specs, function(spec) {
  make_map_row(spec[[1]], counterfactual_scenarios, spec[[2]], FALSE)
}), ncol = 1)
counterfactual_block <-
  make_group_heading("a  Original counterfactuals") /
  counterfactual_rows +
  plot_layout(heights = c(0.12, 3))

canopy_rows <- wrap_plots(lapply(metric_specs, function(spec) {
  make_map_row(spec[[1]], canopy_scenarios, spec[[2]], TRUE)
}), ncol = 1)
canopy_block <-
  make_group_heading("b  Canopy-addition scenarios") /
  canopy_rows +
  plot_layout(heights = c(0.12, 3))

caption_text <- paste(strwrap(paste(
  "Each map shows a borough-specific change; colours are comparable across",
  "all nine scenarios within an outcome row. Red indicates a loss relative",
  "to baseline, white indicates zero change and blue indicates a gain.",
  "Health maps are deterministic spatial allocations and do not show",
  "borough-level uncertainty."
), width = 180), collapse = "\n")

combined_plot <- (counterfactual_block | canopy_block) +
  plot_layout(widths = c(3.2, 6.8)) +
  plot_annotation(
    title = "Borough-level co-benefits of alternative land-use scenarios",
    subtitle = paste0(
      "Scenario-minus-baseline changes under ", temperature_label,
      "°C mid-century climate conditions"
    ),
    caption = caption_text,
    theme = theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10),
      plot.caption = element_text(hjust = 0, size = 9.5, colour = "#4A4A4A")
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
  schema_version = 2,
  created_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  temperature_c = temperature,
  estimand = "Scenario-minus-baseline change calculated separately within each borough",
  health_zonal_rule = "10 m raster-cell centre assigned to borough; values summed and sign reversed",
  uncertainty = "No borough confidence intervals; mapped variation is spatial heterogeneity",
  scenarios = list(
    original_counterfactuals = counterfactual_scenarios,
    canopy_addition = canopy_scenarios
  ),
  colour_scale = "Shared symmetric diverging scale within each outcome; midpoint is zero",
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
