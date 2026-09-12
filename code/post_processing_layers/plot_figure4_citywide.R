#!/usr/bin/env Rscript

# Build manuscript Figure 4 from production citywide summaries. The main
# figure reports London-wide estimands; a separate sensitivity figure preserves
# the equal-weight borough view used by the earlier manuscript workflow.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(tidyr)
})

parse_args <- function(values) {
  result <- list(temperature = 25, price_basis = "late-2025 input-price assumptions")
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

scenario_order <- c(
  "allbuilt", "treerisk", "treeopp",
  "green10", "target10", "green20", "target20", "green30", "target30"
)
scenario_labels <- c(
  allbuilt = "All\nBuilt", treerisk = "Tree\nRisk", treeopp = "Tree\nOpp",
  green10 = "Green\n10%", target10 = "Target\n10%",
  green20 = "Green\n20%", target20 = "Target\n20%",
  green30 = "Green\n30%", target30 = "Target\n30%"
)

args <- parse_args(commandArgs(trailingOnly = TRUE))
city_path <- required_path(args$citywide_summary, "citywide_summary")
health_root <- required_path(args$health_root, "health_root", directory = TRUE)
output_dir_value <- if (is.null(args$output_dir)) {
  file.path(dirname(city_path), "figure4")
} else {
  args$output_dir
}
output_dir <- normalizePath(output_dir_value, mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
temperature <- as.numeric(args$temperature)
if (!is.finite(temperature)) stop("--temperature must be numeric", call. = FALSE)

city <- read.csv(city_path, stringsAsFactors = FALSE, check.names = FALSE) %>%
  filter(temperature_c == temperature, scenario %in% scenario_order)
if (!identical(sort(unique(city$scenario)), sort(scenario_order))) {
  stop("Citywide summary must contain all nine manuscript scenarios", call. = FALSE)
}

health_rows <- lapply(scenario_order, function(scenario) {
  folder <- file.path(health_root, sprintf("%s_%gc", scenario, temperature))
  deterministic_path <- file.path(folder, "city_totals_deterministic.csv")
  draws_path <- file.path(folder, "city_total_draws_by_cause.csv")
  required_path(deterministic_path, "health deterministic table")
  required_path(draws_path, "health draw table")
  deterministic <- read.csv(deterministic_path, stringsAsFactors = FALSE)
  draws <- read.csv(draws_path, stringsAsFactors = FALSE)
  point <- deterministic %>% filter(cause == "all_cause")
  if (nrow(point) != 1 || !"all_cause" %in% names(draws)) {
    stop(sprintf("Expected one all_cause result in %s", folder), call. = FALSE)
  }
  averted_draws <- -as.numeric(draws$all_cause)
  data.frame(
    scenario = scenario,
    estimate = as.numeric(point$deaths_averted),
    lower95 = unname(quantile(averted_draws, 0.025, na.rm = TRUE)),
    upper95 = unname(quantile(averted_draws, 0.975, na.rm = TRUE)),
    draws = sum(is.finite(averted_draws))
  )
}) %>% bind_rows()

health_input_manifest <- setNames(lapply(scenario_order, function(scenario) {
  folder <- file.path(health_root, sprintf("%s_%gc", scenario, temperature))
  deterministic_path <- file.path(folder, "city_totals_deterministic.csv")
  draws_path <- file.path(folder, "city_total_draws_by_cause.csv")
  list(
    deterministic = list(
      path = deterministic_path,
      sha256 = digest::digest(file = deterministic_path, algo = "sha256")
    ),
    draws = list(
      path = draws_path,
      sha256 = digest::digest(file = draws_path, algo = "sha256")
    )
  )
}), scenario_order)

figure_data <- bind_rows(
  city %>% transmute(
    scenario,
    metric = "a  Avoided energy cost",
    estimate = energy_change_vs_baseline / 1e6,
    lower95 = NA_real_, upper95 = NA_real_,
    unit = "£ million/month",
    aggregation = "Citywide sum across unique buildings"
  ),
  city %>% transmute(
    scenario,
    metric = "b  Heavy-work capacity gain",
    estimate = productivity_change_vs_baseline_pp,
    lower95 = NA_real_, upper95 = NA_real_,
    unit = "Percentage points",
    aggregation = "Citywide area-weighted mean over valid 10 m pixels"
  ),
  health_rows %>% transmute(
    scenario,
    metric = "c  Heat-related deaths averted",
    estimate, lower95, upper95,
    unit = "Annual deaths",
    aggregation = "Citywide total using 2021 population and registered mortality"
  )
) %>%
  mutate(
    scenario_id = scenario,
    scenario = factor(scenario, levels = scenario_order, labels = scenario_labels),
    strategy = case_when(
      scenario_id == "allbuilt" ~ "AllBuilt",
      scenario_id == "treerisk" ~ "TreeRisk",
      scenario_id == "treeopp" ~ "TreeOpp",
      grepl("^green", scenario_id) ~ "Green",
      TRUE ~ "Target"
    ) %>% factor(levels = c("AllBuilt", "TreeRisk", "TreeOpp", "Green", "Target")),
    metric = factor(metric, levels = c(
      "a  Avoided energy cost",
      "b  Heavy-work capacity gain",
      "c  Heat-related deaths averted"
    ))
  )

write.csv(
  figure_data %>%
    arrange(metric, scenario) %>%
    transmute(
      scenario_label = gsub("\n", " ", as.character(scenario)),
      scenario = scenario_id,
      metric, estimate, lower95, upper95, unit, aggregation, strategy
    ),
  file.path(output_dir, "figure4_citywide_cobenefits_data.csv"),
  row.names = FALSE
)

panel_units <- c(
  "a  Avoided energy cost" = "£ million/month",
  "b  Heavy-work capacity gain" = "Percentage points",
  "c  Heat-related deaths averted" = "Annual deaths"
)

scenario_colours <- c(
  AllBuilt = "#D95F0E", TreeRisk = "#FEC44F", TreeOpp = "#D9F0A3",
  Green = "#3B7D4A", Target = "#6B5B95"
)

main_plot <- ggplot(figure_data, aes(x = scenario, y = estimate, fill = strategy)) +
  geom_hline(yintercept = 0, colour = "#777777", linewidth = 0.3) +
  geom_vline(xintercept = 3.5, colour = "#B8B8B8", linewidth = 0.35) +
  geom_col(width = 0.72, colour = "#2B2B2B", linewidth = 0.25) +
  geom_errorbar(
    data = figure_data %>% filter(!is.na(lower95)),
    aes(ymin = lower95, ymax = upper95),
    width = 0.18, linewidth = 0.5
  ) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = scenario_colours) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.10))) +
  labs(
    x = NULL,
    y = NULL,
    fill = "Scenario type",
    caption = paste(strwrap(paste0(
      "Bars are relative to the matching baseline under identical climate conditions. ",
      "Energy uses ", args$price_basis, ". Error bars are shown only for health and ",
      "represent the 95% interval from paired exposure-response draws."
    ), width = 175), collapse = "\n")
  ) +
  theme_classic(base_size = 10) +
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", hjust = 0),
    axis.text.x = element_text(size = 8),
    plot.caption = element_text(hjust = 0, size = 8, colour = "#4A4A4A"),
    panel.spacing.x = grid::unit(1.2, "lines")
  )

# Add truthful panel-specific units without implying a shared y-axis.
main_plot <- main_plot + geom_text(
  data = data.frame(
    metric = factor(names(panel_units), levels = levels(figure_data$metric)),
    scenario = factor(rep(scenario_labels[[1]], length(panel_units)), levels = scenario_labels),
    estimate = 0,
    label = unname(panel_units),
    strategy = "Green"
  ),
  aes(label = label), x = -Inf, y = Inf, hjust = -0.02, vjust = 1.4,
  inherit.aes = FALSE, size = 2.7, colour = "#4A4A4A"
)

for (extension in c("png", "pdf", "svg")) {
  ggsave(
    file.path(output_dir, paste0("figure4_citywide_cobenefits.", extension)),
    main_plot, width = 12, height = 4.4, units = "in", dpi = 400,
    bg = "white"
  )
}

borough_outputs <- NULL
if (!is.null(args$borough_summary)) {
  borough_path <- required_path(args$borough_summary, "borough_summary")
  borough <- read.csv(borough_path, stringsAsFactors = FALSE, check.names = FALSE) %>%
    filter(temperature_c == temperature)
  baseline <- borough %>%
    filter(scenario == "baseline") %>%
    select(borough_code, baseline_energy = energy_savings,
           baseline_workability = workability_fraction)
  borough_change <- borough %>%
    filter(scenario %in% scenario_order) %>%
    inner_join(baseline, by = "borough_code") %>%
    transmute(
      borough, borough_code, scenario,
      energy_change_million_gbp = (energy_savings - baseline_energy) / 1e6,
      productivity_change_pp =
        round(workability_fraction * 100, 2) -
        round(baseline_workability * 100, 2)
    )
  if (any(table(borough_change$scenario) != 33)) {
    stop("Borough sensitivity requires exactly 33 boroughs per scenario", call. = FALSE)
  }
  borough_long <- borough_change %>%
    pivot_longer(
      c(energy_change_million_gbp, productivity_change_pp),
      names_to = "metric", values_to = "value"
    ) %>%
    mutate(
      scenario_id = scenario,
      scenario = factor(scenario, levels = scenario_order, labels = scenario_labels),
      strategy = case_when(
        scenario_id == "allbuilt" ~ "AllBuilt",
        scenario_id == "treerisk" ~ "TreeRisk",
        scenario_id == "treeopp" ~ "TreeOpp",
        grepl("^green", scenario_id) ~ "Green",
        TRUE ~ "Target"
      ) %>% factor(levels = c("AllBuilt", "TreeRisk", "TreeOpp", "Green", "Target")),
      metric = recode(
        metric,
        energy_change_million_gbp = "Avoided energy cost (£ million/month per borough)",
        productivity_change_pp = "Heavy-work capacity gain (percentage points)"
      )
    )
  write.csv(
    borough_long %>%
      arrange(metric, scenario, borough_code) %>%
      transmute(
        borough, borough_code,
        scenario_label = gsub("\n", " ", as.character(scenario)),
        scenario = scenario_id,
        metric, value, strategy
      ),
    file.path(output_dir, "figure4_borough_sensitivity_data.csv"),
    row.names = FALSE
  )
  sensitivity_plot <- ggplot(
    borough_long,
    aes(x = scenario, y = value, fill = strategy)
  ) +
    geom_hline(yintercept = 0, colour = "#777777", linewidth = 0.3) +
    geom_vline(xintercept = 3.5, colour = "#B8B8B8", linewidth = 0.35) +
    geom_boxplot(width = 0.66, outlier.size = 0.8, linewidth = 0.35) +
    facet_wrap(~metric, scales = "free_y", nrow = 1) +
    scale_fill_manual(values = scenario_colours) +
    labs(
      x = NULL, y = NULL, fill = "Scenario type",
      caption = paste0(
        "Boxes show the spatial distribution across 33 boroughs, not model uncertainty. ",
        "Each borough has equal weight."
      )
    ) +
    theme_classic(base_size = 10) +
    theme(
      legend.position = "top", strip.background = element_blank(),
      strip.text = element_text(face = "bold"), axis.text.x = element_text(size = 8),
      plot.caption = element_text(hjust = 0, size = 8, colour = "#4A4A4A")
    )
  for (extension in c("png", "pdf", "svg")) {
    ggsave(
      file.path(output_dir, paste0("figure4_borough_sensitivity.", extension)),
      sensitivity_plot, width = 8.2, height = 4.2, units = "in", dpi = 400,
      bg = "white"
    )
  }

  # Put the two estimands beside one another without forcing unlike energy
  # quantities (a London total and a mean borough) onto a shared axis.
  city_comparison <- figure_data %>%
    filter(metric %in% c(
      "a  Avoided energy cost", "b  Heavy-work capacity gain"
    )) %>%
    transmute(
      scenario_id, strategy,
      scenario = factor(scenario, levels = scenario_labels),
      metric = recode(
        as.character(metric),
        `a  Avoided energy cost` = "Avoided energy cost",
        `b  Heavy-work capacity gain` = "Heavy-work capacity gain"
      ),
      estimand = ifelse(
        metric == "Avoided energy cost",
        "Citywide total (£ million/month)",
        "Citywide area-weighted mean (percentage points)"
      ),
      estimate
    )
  borough_comparison <- borough_long %>%
    group_by(scenario_id, scenario, strategy, metric) %>%
    summarise(estimate = mean(value), .groups = "drop") %>%
    mutate(
      metric = ifelse(
        grepl("energy", metric, ignore.case = TRUE),
        "Avoided energy cost", "Heavy-work capacity gain"
      ),
      estimand = ifelse(
        metric == "Avoided energy cost",
        "Unweighted mean borough (£ million/month per borough)",
        "Unweighted mean borough (percentage points)"
      )
    )
  aggregation_comparison <- bind_rows(city_comparison, borough_comparison) %>%
    mutate(
      metric = factor(
        metric,
        levels = c("Avoided energy cost", "Heavy-work capacity gain")
      ),
      estimand = factor(estimand, levels = c(
        "Citywide total (£ million/month)",
        "Unweighted mean borough (£ million/month per borough)",
        "Citywide area-weighted mean (percentage points)",
        "Unweighted mean borough (percentage points)"
      ))
    )
  write.csv(
    aggregation_comparison %>%
      arrange(metric, estimand, scenario) %>%
      transmute(
        scenario = scenario_id,
        scenario_label = gsub("\n", " ", as.character(scenario)),
        metric, estimand, estimate, strategy
      ),
    file.path(output_dir, "extended_data_citywide_vs_borough_data.csv"),
    row.names = FALSE
  )

  make_comparison_panel <- function(estimand_name, panel_label) {
    ggplot(
      filter(aggregation_comparison, estimand == estimand_name),
      aes(x = scenario, y = estimate, fill = strategy)
    ) +
      geom_col(width = 0.72, colour = "#2B2B2B", linewidth = 0.25) +
      scale_fill_manual(values = scenario_colours) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.10))) +
      labs(title = panel_label, subtitle = estimand_name, x = NULL, y = NULL) +
      theme_classic(base_size = 9) +
      theme(
        plot.title = element_text(face = "bold", size = 10),
        plot.subtitle = element_text(size = 8.5),
        axis.text.x = element_text(size = 7.5),
        legend.position = "none"
      )
  }

  aggregation_plot <-
    make_comparison_panel(
      "Citywide total (£ million/month)", "a  Energy: citywide"
    ) +
    make_comparison_panel(
      "Unweighted mean borough (£ million/month per borough)",
      "b  Energy: equal-weight borough"
    ) +
    make_comparison_panel(
      "Citywide area-weighted mean (percentage points)",
      "c  Productivity: citywide"
    ) +
    make_comparison_panel(
      "Unweighted mean borough (percentage points)",
      "d  Productivity: equal-weight borough"
    ) +
    plot_layout(ncol = 2, guides = "collect") +
    plot_annotation(
      title = "Sensitivity of co-benefit summaries to geographic aggregation",
      caption = paste(
        "Citywide panels answer the London-wide benefit question. Equal-weight",
        "borough panels describe the average local borough response. Different",
        "scales are intentional; compare scenario ordering, not bar heights",
        "between panels. Health is excluded because no reviewed borough-level",
        "uncertainty workflow is currently available."
      ),
      theme = theme(
        plot.title = element_text(face = "bold", size = 12),
        plot.caption = element_text(hjust = 0, size = 8, colour = "#4A4A4A"),
        legend.position = "top"
      )
    ) & theme(legend.position = "top")

  for (extension in c("png", "pdf", "svg")) {
    ggsave(
      file.path(
        output_dir,
        paste0("extended_data_citywide_vs_borough.", extension)
      ),
      aggregation_plot, width = 9.2, height = 7.0, units = "in", dpi = 400,
      bg = "white"
    )
  }
  borough_outputs <- list(
    source = list(
      path = borough_path,
      sha256 = digest::digest(file = borough_path, algo = "sha256")
    ),
    data = file.path(output_dir, "figure4_borough_sensitivity_data.csv"),
    aggregation_comparison = file.path(
      output_dir, "extended_data_citywide_vs_borough_data.csv"
    ),
    interpretation = "Spatial borough distribution; not a confidence interval"
  )
}

artifact_names <- c(
  "figure4_citywide_cobenefits_data.csv",
  "figure4_citywide_cobenefits.png",
  "figure4_citywide_cobenefits.pdf",
  "figure4_citywide_cobenefits.svg"
)
if (!is.null(borough_outputs)) {
  artifact_names <- c(
    artifact_names,
    "figure4_borough_sensitivity_data.csv",
    "figure4_borough_sensitivity.png",
    "figure4_borough_sensitivity.pdf",
    "figure4_borough_sensitivity.svg",
    "extended_data_citywide_vs_borough_data.csv",
    "extended_data_citywide_vs_borough.png",
    "extended_data_citywide_vs_borough.pdf",
    "extended_data_citywide_vs_borough.svg"
  )
}
artifact_manifest <- setNames(lapply(artifact_names, function(name) {
  path <- file.path(output_dir, name)
  list(path = name, sha256 = digest::digest(file = path, algo = "sha256"))
}), artifact_names)

manifest <- list(
  schema_version = 1,
  created_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  temperature_c = temperature,
  estimands = list(
    energy = "Citywide sum across unique building records",
    productivity = "Citywide pixel-area-weighted mean continuous Hothaps workability change",
    health = "Citywide deaths averted using 2021 population and registered mortality"
  ),
  uncertainty = list(
    health = "2.5th and 97.5th percentiles of exposure-response draws",
    energy = "Not propagated; no error bar",
    productivity = "Not propagated; no error bar"
  ),
  price_basis = args$price_basis,
  inputs = list(
    citywide_summary = list(path = city_path, sha256 = digest::digest(file = city_path, algo = "sha256")),
    health = health_input_manifest,
    borough_sensitivity = borough_outputs
  ),
  outputs = artifact_manifest
)
write_json(
  manifest,
  file.path(output_dir, "figure4_citywide_cobenefits_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE
)
message("Created Figure 4 outputs in ", output_dir)
