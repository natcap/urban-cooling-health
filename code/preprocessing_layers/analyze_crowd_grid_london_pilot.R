# Crowd-Grid sensitivity for the InVEST UHI parameter pilot.
#
# The analysis uses Crowd-Grid-derived MSOA daily temperature averages, not
# direct station observations. It compares the 90th percentile across London
# MSOAs with the median of the three MSOAs containing the screened rural MIDAS
# sites (Wisley, Charlwood and Woburn). The contrast is a diagnostic spatial
# proxy; it is not a calibrated replacement for InVEST's uhi_max parameter.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(sf)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript analyze_crowd_grid_london_pilot.R <input_dir> <station_screen_csv> <figure_png>")
}
input_dir <- args[[1]]
station_screen_csv <- args[[2]]
figure_png <- args[[3]]

years <- 2018:2023
month_keep <- 7L

read_heat_pack <- function(path, geography_group, variable) {
  # Five metadata lines precede the header in the official product.
  x <- read_csv(path, skip = 5, show_col_types = FALSE)
  names(x)[1] <- "date"
  x |>
    mutate(date = as.Date(date)) |>
    filter(lubridate::year(date) %in% years, lubridate::month(date) == month_keep) |>
    pivot_longer(-date, names_to = "area_name", values_to = "temperature_c") |>
    mutate(geography_group = geography_group, variable = variable)
}

daily_files <- list.files(file.path(input_dir, "daily_csvs"), full.names = TRUE)

parse_file <- function(path) {
  slug <- sub("_recent-heat-pack.*$", "", basename(path))
  variable <- if (grepl("maximum", basename(path))) "Daily maximum" else "Daily minimum"
  group <- if (slug %in% c("guildford", "crawley", "central-bedfordshire")) {
    "rural_reference_la"
  } else {
    "london_la"
  }
  read_heat_pack(path, group, variable) |> mutate(source_slug = slug)
}
all_daily <- bind_rows(lapply(daily_files, parse_file)) |>
  filter(geography_group %in% c("london_la", "rural_reference_la"))

# Identify the exact 2021 MSOAs containing the three screened rural stations.
boundaries <- st_read(
  file.path(input_dir, "boundaries", "uk-msoa-iz-dez-boundaries.shp"),
  quiet = TRUE
) |>
  st_transform(4326)
stations <- read_csv(station_screen_csv, show_col_types = FALSE) |>
  filter(include_screened_contrast, grepl("Rural-background", screened_role)) |>
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)
station_msoa <- st_join(stations, boundaries[, c("NAME", "CODE")], left = TRUE) |>
  st_drop_geometry() |>
  select(station_id, station_name, msoa_name = NAME, msoa_code = CODE)
if (nrow(station_msoa) != 3 || any(is.na(station_msoa$msoa_name))) {
  stop("Could not resolve all three rural stations to supplied MSOA boundaries")
}
write_csv(station_msoa, file.path(input_dir, "rural_station_msoa_lookup.csv"))

# Remove each pack's local-authority aggregate (its second column) and retain
# unique MSOA series. Duplicate rows are possible only if a geography appears
# in more than one downloaded pack, so collapse them deterministically.
london_msoa <- all_daily |>
  filter(geography_group == "london_la", grepl(" [0-9]{3}$", area_name)) |>
  group_by(date, variable, area_name) |>
  summarise(temperature_c = mean(temperature_c, na.rm = TRUE), .groups = "drop")
rural_station_series <- all_daily |>
  filter(geography_group == "rural_reference_la", area_name %in% station_msoa$msoa_name) |>
  group_by(date, variable, area_name) |>
  summarise(temperature_c = mean(temperature_c, na.rm = TRUE), .groups = "drop")

if (n_distinct(rural_station_series$area_name) != 3) {
  stop("Expected temperature series for three rural station MSOAs")
}

daily_proxy <- london_msoa |>
  group_by(date, variable) |>
  summarise(
    london_msoa_median_c = median(temperature_c, na.rm = TRUE),
    london_msoa_p90_c = quantile(temperature_c, 0.90, na.rm = TRUE),
    london_msoa_max_c = max(temperature_c, na.rm = TRUE),
    london_msoa_n = sum(!is.na(temperature_c)),
    .groups = "drop"
  ) |>
  left_join(
    rural_station_series |>
      group_by(date, variable) |>
      summarise(
        rural_station_msoa_median_c = median(temperature_c, na.rm = TRUE),
        rural_station_msoa_n = sum(!is.na(temperature_c)),
        .groups = "drop"
      ),
    by = c("date", "variable")
  ) |>
  mutate(
    year = lubridate::year(date),
    london_p90_minus_rural_median_c = london_msoa_p90_c - rural_station_msoa_median_c,
    london_max_minus_rural_median_c = london_msoa_max_c - rural_station_msoa_median_c
  )
hot_dates <- daily_proxy |>
  filter(variable == "Daily maximum", london_msoa_p90_c >= 28) |>
  pull(date)
daily_proxy <- daily_proxy |>
  mutate(hot_day = date %in% hot_dates)
write_csv(daily_proxy, file.path(input_dir, "crowd_grid_daily_uhi_proxy.csv"))

summarise_contrast <- function(data, column, label) {
  data |>
    group_by(variable) |>
    summarise(
      contrast = label,
      n_days = n(),
      median_c = median({{ column }}, na.rm = TRUE),
      p90_c = quantile({{ column }}, 0.90, na.rm = TRUE),
      maximum_c = max({{ column }}, na.rm = TRUE),
      proportion_ge_5c = mean({{ column }} >= 5, na.rm = TRUE),
      .groups = "drop"
    )
}
summary_all <- bind_rows(
  summarise_contrast(daily_proxy, london_p90_minus_rural_median_c, "London MSOA P90 - rural-station MSOA median"),
  summarise_contrast(daily_proxy, london_max_minus_rural_median_c, "London MSOA maximum - rural-station MSOA median"),
  summarise_contrast(filter(daily_proxy, hot_day), london_p90_minus_rural_median_c, "Hot-day London MSOA P90 - rural-station MSOA median"),
  summarise_contrast(filter(daily_proxy, hot_day), london_max_minus_rural_median_c, "Hot-day London MSOA maximum - rural-station MSOA median")
)
write_csv(summary_all, file.path(input_dir, "crowd_grid_uhi_proxy_summary.csv"))

plot_data <- daily_proxy |>
  mutate(
    variable = factor(variable, levels = c("Daily maximum", "Daily minimum")),
    year = factor(year)
  )
p <- ggplot(plot_data, aes(year, london_p90_minus_rural_median_c)) +
  geom_hline(yintercept = 5, linewidth = 0.45, colour = "#B2182B", linetype = 2) +
  geom_boxplot(width = 0.58, outlier.shape = NA, fill = "#D9E6F2", colour = "#3F4E5E", linewidth = 0.45) +
  geom_jitter(aes(colour = hot_day), width = 0.13, height = 0, alpha = 0.55, size = 1.05) +
  facet_wrap(~variable, nrow = 1) +
  scale_colour_manual(values = c(`FALSE` = "#7A8793", `TRUE` = "#D55E00"),
                      labels = c(`FALSE` = "Other July day", `TRUE` = "London P90 >= 28 C"),
                      name = NULL) +
  labs(
    x = NULL,
    y = "London MSOA P90 - rural MSOA median (C)"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.margin = margin(8, 10, 8, 14)
  )
ggsave(figure_png, p, width = 8.0, height = 4.0, dpi = 300, bg = "white")

message("Wrote Crowd-Grid pilot outputs to: ", input_dir)
