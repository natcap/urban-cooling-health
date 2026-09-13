# Shared plotting conventions for the three Figure 4 source notebooks.
# Keeping these values here prevents scenario order, labels and colours from
# drifting between the energy, productivity and health panels.

figure4_scenarios <- c(
  "allbuilt", "treerisk", "treeopp",
  "green10", "target10", "green20", "target20", "green30", "target30"
)

figure4_labels <- c(
  allbuilt = "All\nBuilt", treerisk = "Tree\nRisk", treeopp = "Tree\nOpp",
  green10 = "Green\n10%", target10 = "Target\n10%",
  green20 = "Green\n20%", target20 = "Target\n20%",
  green30 = "Green\n30%", target30 = "Target\n30%"
)

figure4_colours <- c(
  AllBuilt = "#D95F0E", TreeRisk = "#FEC44F", TreeOpp = "#D9F0A3",
  Green = "#3B7D4A", Target = "#6B5B95"
)

figure4_data_root <- function() {
  value <- Sys.getenv("URBAN_COOLING_DATA_ROOT", unset = "")
  if (!nzchar(value) || !dir.exists(value)) {
    stop(
      paste(
        "Set URBAN_COOLING_DATA_ROOT to the Wellcome Trust Project Data folder",
        "before knitting this notebook."
      ),
      call. = FALSE
    )
  }
  normalizePath(value)
}

figure4_output_dir <- function(default) {
  value <- Sys.getenv("URBAN_COOLING_FIGURE_OUTPUT_DIR", unset = "")
  normalizePath(if (nzchar(value)) value else default, mustWork = FALSE)
}

figure4_validate_scenarios <- function(data, column = "scenario") {
  observed <- unique(as.character(data[[column]]))
  missing <- setdiff(figure4_scenarios, observed)
  if (length(missing)) {
    stop("Missing Figure 4 scenarios: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  invisible(data)
}

figure4_prepare_scenarios <- function(data, column = "scenario") {
  figure4_validate_scenarios(data, column)
  data %>%
    mutate(
      scenario_id = .data[[column]],
      scenario = factor(
        .data[[column]], levels = figure4_scenarios, labels = figure4_labels
      ),
      strategy = case_when(
        scenario_id == "allbuilt" ~ "AllBuilt",
        scenario_id == "treerisk" ~ "TreeRisk",
        scenario_id == "treeopp" ~ "TreeOpp",
        grepl("^green", scenario_id) ~ "Green",
        TRUE ~ "Target"
      ) %>% factor(levels = c("AllBuilt", "TreeRisk", "TreeOpp", "Green", "Target")),
      scenario_family = ifelse(
        scenario_id %in% c("allbuilt", "treerisk", "treeopp"),
        "Original counterfactuals", "Canopy-addition scenarios"
      )
    )
}

figure4_panel_theme <- function() {
  theme_classic(base_size = 11) +
    theme(
      legend.position = "top",
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9),
      axis.text.x = element_text(size = 8.5),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 9),
      plot.caption = element_text(hjust = 0, size = 8, colour = "#4A4A4A")
    )
}

# Add signed values in the style of func_plot_change_point(). For health,
# labels are placed beyond the confidence interval so they remain legible.
figure4_add_value_labels <- function(plot, data, digits, size = 3) {
  if (!"lower95" %in% names(data)) data$lower95 <- NA_real_
  if (!"upper95" %in% names(data)) data$upper95 <- NA_real_

  label_data <- data %>%
    mutate(
      label = sprintf(paste0("%+.", digits, "f"), estimate),
      label_anchor = ifelse(
        estimate >= 0,
        ifelse(is.na(upper95), estimate, upper95),
        ifelse(is.na(lower95), estimate, lower95)
      ),
      label_vjust = ifelse(estimate >= 0, -0.45, 1.35)
    )

  plot +
    geom_text(
      data = label_data,
      aes(y = label_anchor, label = label, vjust = label_vjust),
      size = size, fontface = "bold", show.legend = FALSE
    ) +
    coord_cartesian(clip = "off")
}

figure4_save_panel <- function(plot, output_dir, stem) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (extension in c("png", "pdf", "svg")) {
    ggsave(
      file.path(output_dir, paste0(stem, ".", extension)),
      plot, width = 5.2, height = 4.2, units = "in", dpi = 400, bg = "white"
    )
  }
}
