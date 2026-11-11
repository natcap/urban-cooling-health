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
palette_env <- new.env()
sys.source(file.path(script_dir, "..", "func_colors.R"), envir = palette_env)

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
      (workability_fraction - baseline_workability) * 100
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
  file.path(output_dir, "figure5_borough_cobenefits_20261110_v3_data.csv"),
  row.names = FALSE
)

map_sf <- boroughs %>%
  left_join(map_data, by = c("borough", "borough_code"))

# One shared scale per outcome across all nine scenarios. A symmetric
# pseudo-log (asinh) colour transform preserves zero/sign and reveals modest
# greening differences without clipping the extreme AllBuilt counterfactual.
# Legends remain in original units; explicitly describe the transform in legend.
make_map_row <- function(metric_name, legend_title, letter) {
  plot_data <- filter(map_sf, metric == metric_name) %>%
    mutate(scenario = factor(scenario_id, levels = scenario_order,
      labels = c("AllBuilt", "TreeRisk", "TreeOpp", "Green10", "Target10",
                 "Green20", "Target20", "Green30", "Target30")))
  limit <- max(abs(plot_data$value), na.rm = TRUE)
  sigma <- limit / 20
  breaks <- c(-limit, -limit / 5, 0, limit / 5, limit)
  p <- ggplot(plot_data) +
    geom_sf(aes(fill = value), colour = "white", linewidth = 0.10) +
    facet_wrap(~scenario, nrow = 1) +
    scale_fill_gradient2(
      low = "#B35806", mid = "#FAFAFA", high = "#2166AC",
      midpoint = 0, limits = c(-limit, limit),
      trans = scales::pseudo_log_trans(sigma = sigma, base = 10),
      breaks = breaks, labels = scales::label_number(accuracy = ifelse(limit < 10, 0.1, 1)),
      name = legend_title,
      guide = guide_colourbar(title.position = "top", title.hjust = 0,
        barheight = grid::unit(0.65, "in"), barwidth = grid::unit(0.14, "in"))
    ) +
    coord_sf(datum = NA, expand = FALSE) +
    labs(title = paste(letter, " ", metric_name)) +
    theme_void(base_size = 11, base_family = "sans") +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0,
                                margin = margin(b = 5)),
      strip.text = element_text(face = "bold", colour = "black", size = 10,
                                margin = margin(t = 5, b = 5)),
      strip.background = element_blank(),
      legend.position = "right", legend.title = element_text(size = 10),
      legend.text = element_text(size = 10),
      panel.spacing.x = grid::unit(0.08, "in"),
      plot.margin = margin(3, 3, 3, 3)
    )
  # Colour only scenario text; map fill retains its quantitative meaning.
  g <- ggplotGrob(p)
  strips <- which(grepl("^strip-t", g$layout$name))
  strips <- strips[order(g$layout$l[strips])]
  for (i in seq_along(strips)) {
    colour <- palette_env$scenario_colors[[scenario_order[[i]]]]
    # Preserve hue, darkening only as needed for legibility on white (4.5:1).
    # This text-only adjustment does not change the shared scenario fill palette.
    rgb <- grDevices::col2rgb(colour) / 255
    contrast <- function(x) {
      linear <- ifelse(x <= .04045, x / 12.92, ((x + .055) / 1.055)^2.4)
      1.05 / (sum(linear * c(.2126, .7152, .0722)) + .05)
    }
    while (contrast(rgb) < 4.5) rgb <- rgb * .97
    ink <- grDevices::rgb(rgb[1], rgb[2], rgb[3])
    recolour <- function(grob) {
      if (inherits(grob, "rect")) { grob$gp$fill <- NA; grob$gp$col <- NA }
      if (inherits(grob, "text")) grob$gp$col <- ink
      if (!is.null(grob$children)) grob$children <- do.call(grid::gList, lapply(grob$children, recolour))
      if (!is.null(grob$grobs)) grob$grobs <- lapply(grob$grobs, recolour)
      grob
    }
    g$grobs[[strips[[i]]]] <- recolour(g$grobs[[strips[[i]]]])
  }
  g
}

# Grob layout needs a device; a null device avoids creating stray Rplots.pdf.
grDevices::pdf(file = NULL)
map_rows <- list(
  make_map_row("Avoided energy cost", "£m/month", "a"),
  make_map_row("Heavy-work capacity gain", "Percentage\npoints", "b"),
  make_map_row("Heat-related deaths averted", "Annual\ndeaths", "c")
)
# Equal legend columns keep scenario maps vertically aligned across outcomes.
common_widths <- do.call(grid::unit.pmax, lapply(map_rows, function(g) g$widths))
combined_plot <- wrap_plots(lapply(map_rows, function(g) {
  g$widths <- common_widths
  wrap_elements(full = g)
}), ncol = 1)

for (extension in c("png")) {
  ggsave(
    file.path(output_dir, paste0("figure5_borough_cobenefits_20261110_v3.", extension)),
    combined_plot, width = 13.2, height = 5.8, units = "in", dpi = 400,
    bg = "white", limitsize = FALSE
  )
}

grDevices::dev.off()
artifact_names <- c(
  "figure5_borough_cobenefits_20261110_v3_data.csv",
  "figure5_borough_cobenefits_20261110_v3.png"
)
manifest <- list(
  generator = normalizePath(file.path(script_dir, "plot_figure5_borough_cobenefits.R")),
  generator_sha256 = digest::digest(file = file.path(script_dir, "plot_figure5_borough_cobenefits.R"), algo = "sha256"),
  palette_source = normalizePath(file.path(script_dir, "..", "func_colors.R")),
  figure_version = "20261110_v3",
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
  colour_scale = "Shared symmetric pseudo-log (asinh) diverging scale per outcome; sigma=max(abs(value))/20, midpoint zero; colourbar labels in original units",
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
  file.path(output_dir, "figure5_borough_cobenefits_20261110_v3_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE
)
message("Created borough co-benefit maps in ", output_dir)
