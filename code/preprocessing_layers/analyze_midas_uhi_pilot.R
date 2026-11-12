#!/usr/bin/env Rscript

# Analyse the bounded 2018-2023 MIDAS London UHI pilot.
#
# Primary estimates retain only observations whose MIDAS status digit is 0
# (observed and not suspect). Status 1 observations are retained only in the
# explicit QC sensitivity table. Station roles remain provisional until the
# exposure, history and surrounding land-cover checks are complete.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
  library(sf)
})

# Surface data/plot warnings at the operation that generated them.
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: analyze_midas_uhi_pilot.R INPUT_DIR LONDON_BOUNDARY_SHP FIGURE_DIR")
}

input_dir <- normalizePath(args[[1]], mustWork = TRUE)
boundary_path <- normalizePath(args[[2]], mustWork = TRUE)
figure_dir <- normalizePath(args[[3]], mustWork = TRUE)

stations <- data.table::data.table(
  station_id = c("00697", "62084", "00723", "62073", "00709",
                 "00719", "30620", "00487", "00744", "00458"),
  station_name = c("St James's Park", "Battersea Heliport", "Kew Gardens",
                   "Teddington, Bushy Park", "Northolt", "Wisley",
                   "Charlwood", "Writtle", "East Malling", "Woburn"),
  role = rep(c("Urban candidate", "Rural candidate"), each = 5),
  latitude = c(51.505, 51.470, 51.482, 51.422, 51.549,
               51.311, 51.144, 51.734, 51.288, 52.014),
  longitude = c(-0.131, -0.179, -0.294, -0.341, -0.417,
                -0.476, -0.229, 0.429, 0.448, -0.596),
  elevation_m = c(5, 5, 6, 9, 33, 38, 67, 32, 33, 89)
)

read_badcsv <- function(path) {
  first_lines <- readLines(path, warn = FALSE, n = 400)
  header_row <- which(startsWith(first_lines, "ob_time,"))
  if (length(header_row) != 1L) stop("Could not identify BADC-CSV header: ", path)
  x <- withCallingHandlers(
    data.table::fread(
      path,
      skip = header_row - 1L,
      select = c("ob_time", "version_num", "src_id", "air_temperature",
                 "air_temperature_q", "meto_stmp_time"),
      na.strings = c("NA", "")
    ),
    warning = function(w) {
      if (grepl("Discarded single-line footer", conditionMessage(w), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
  x[, station_id := sprintf("%05d", as.integer(src_id))]
  x[, datetime := as.POSIXct(ob_time, tz = "UTC")]
  x[, record_timestamp := as.POSIXct(meto_stmp_time, tz = "UTC")]
  x[, temperature_c := as.numeric(air_temperature)]
  x[, qc_code := as.integer(air_temperature_q)]
  # MESQL is right-aligned; the hundreds digit is qc_status.
  x[, qc_status := (qc_code %/% 100L) %% 10L]
  x <- x[version_num == 1L & !is.na(datetime) & month(datetime) == 7L]
  data.table::setorder(x, station_id, datetime, record_timestamp, na.last = TRUE)
  x <- x[, .SD[.N], by = .(station_id, datetime)]
  x[, source_file := basename(path)]
  x[, .(station_id, datetime, temperature_c, qc_code, qc_status, source_file)]
}

month <- function(x) as.integer(format(x, "%m", tz = "UTC"))

files <- list.files(file.path(input_dir, "hourly"), pattern = "\\.csv$",
                    recursive = TRUE, full.names = TRUE)
if (length(files) != 60L) warning("Expected 60 hourly files; found ", length(files))
hourly <- data.table::rbindlist(lapply(files, read_badcsv), use.names = TRUE)
hourly <- merge(hourly, stations, by = "station_id", all.x = TRUE)
if (anyNA(hourly$role)) stop("One or more station IDs lack metadata")
hourly[, year := as.integer(format(datetime, "%Y", tz = "UTC"))]
hourly[, date := as.Date(datetime, tz = "UTC")]
hourly[, hour_utc := as.integer(format(datetime, "%H", tz = "UTC"))]
hourly[, suspect := qc_status != 0L]

completeness <- hourly[, .(
  expected_hours = 31L * 24L,
  records_after_deduplication = .N,
  numeric_temperature_hours = sum(!is.na(temperature_c)),
  valid_primary_hours = sum(!is.na(temperature_c) & qc_status == 0L),
  suspect_temperature_hours = sum(!is.na(temperature_c) & qc_status == 1L),
  other_status_hours = sum(!is.na(temperature_c) & !qc_status %in% c(0L, 1L)),
  primary_completeness_pct = round(100 * sum(!is.na(temperature_c) & qc_status == 0L) /
                                     (31L * 24L), 1)
), by = .(station_id, station_name, role, year)]

make_daily <- function(include_suspect = FALSE) {
  h <- if (include_suspect) hourly[!is.na(temperature_c)] else
    hourly[!is.na(temperature_c) & qc_status == 0L]
  h[, temperature_adjusted_5m_c := temperature_c + 0.0065 * (elevation_m - 5)]
  long <- melt(
    h,
    id.vars = c("station_id", "station_name", "role", "elevation_m", "year",
                "date", "hour_utc"),
    measure.vars = c("temperature_c", "temperature_adjusted_5m_c"),
    variable.name = "elevation_treatment", value.name = "temperature"
  )
  long[, elevation_treatment := fifelse(
    elevation_treatment == "temperature_c", "Raw", "Adjusted to 5 m")]
  long[, .(
    valid_hours = uniqueN(hour_utc),
    daily_tmax_c = if (uniqueN(hour_utc) >= 18L) max(temperature) else NA_real_,
    daily_tmin_c = if (uniqueN(hour_utc) >= 18L) min(temperature) else NA_real_,
    # Supplemental daytime mean. July observations are in UTC, so 09:00-17:00
    # UTC approximates 10:00-18:00 British Summer Time. Daily maximum remains
    # primary because it matches the July Tmax basis used for UCM T_ref.
    daytime_mean_c = {
      z <- temperature[hour_utc %in% 9L:17L]
      if (length(z) >= 7L) mean(z) else NA_real_
    },
    nocturnal_mean_c = {
      z <- temperature[hour_utc %in% c(22L, 23L, 0L:5L)]
      if (length(z) >= 6L) mean(z) else NA_real_
    }
  ), by = .(station_id, station_name, role, elevation_m, year, date,
            elevation_treatment)]
}

make_group_uhi <- function(daily) {
  metric_long <- melt(
    daily,
    id.vars = c("station_id", "station_name", "role", "year", "date",
                "elevation_treatment"),
    measure.vars = c("daily_tmax_c", "daytime_mean_c", "nocturnal_mean_c"),
    variable.name = "metric", value.name = "temperature_c",
    na.rm = TRUE
  )
  grouped <- metric_long[, .(
    group_temperature_c = median(temperature_c),
    station_count = uniqueN(station_id)
  ), by = .(year, date, elevation_treatment, metric, role)]
  wide <- dcast(grouped, year + date + elevation_treatment + metric ~ role,
                value.var = c("group_temperature_c", "station_count"))
  setnames(wide,
           c("group_temperature_c_Urban candidate", "group_temperature_c_Rural candidate",
             "station_count_Urban candidate", "station_count_Rural candidate"),
           c("urban_temperature_c", "rural_temperature_c", "urban_stations", "rural_stations"))
  wide <- wide[urban_stations >= 3L & rural_stations >= 3L]
  wide[, uhi_c := urban_temperature_c - rural_temperature_c]
  wide[, metric := fcase(
    metric == "daily_tmax_c", "Daily maximum",
    metric == "daytime_mean_c", "Daytime mean",
    metric == "nocturnal_mean_c", "Night-time mean"
  )]
  hot_thresholds <- wide[metric == "Daily maximum",
    .(hot_day_threshold_c = quantile(rural_temperature_c, 0.90, na.rm = TRUE)),
    by = elevation_treatment]
  hot_dates <- merge(
    wide[metric == "Daily maximum", .(elevation_treatment, date, rural_temperature_c)],
    hot_thresholds, by = "elevation_treatment"
  )
  hot_dates[, hot_day := rural_temperature_c >= hot_day_threshold_c]
  wide <- merge(wide, hot_dates[, .(elevation_treatment, date, hot_day)],
                by = c("elevation_treatment", "date"), all.x = TRUE)
  wide[]
}

summarise_uhi <- function(group_uhi, qc_rule) {
  all_days <- group_uhi[, .(
    n_days = .N, mean_c = mean(uhi_c), median_c = median(uhi_c),
    p10_c = quantile(uhi_c, 0.10), p90_c = quantile(uhi_c, 0.90),
    maximum_c = max(uhi_c)
  ), by = .(elevation_treatment, metric)][, day_subset := "All July days"]
  hot <- group_uhi[hot_day == TRUE, .(
    n_days = .N, mean_c = mean(uhi_c), median_c = median(uhi_c),
    p10_c = quantile(uhi_c, 0.10), p90_c = quantile(uhi_c, 0.90),
    maximum_c = max(uhi_c)
  ), by = .(elevation_treatment, metric)][, day_subset := "Hot days"]
  out <- rbind(all_days, hot, use.names = TRUE)
  out[, qc_rule := qc_rule]
  setcolorder(out, c("qc_rule", "day_subset", "metric", "elevation_treatment",
                    "n_days", "mean_c", "median_c", "p10_c", "p90_c", "maximum_c"))
  out[]
}

daily_primary <- make_daily(FALSE)
uhi_primary <- make_group_uhi(daily_primary)
daily_all_numeric <- make_daily(TRUE)
uhi_all_numeric <- make_group_uhi(daily_all_numeric)

summary_primary <- summarise_uhi(uhi_primary, "Exclude MIDAS suspect status")
qc_sensitivity <- rbind(
  summary_primary,
  summarise_uhi(uhi_all_numeric, "Include MIDAS suspect status")
)

yearly <- uhi_primary[, .(
  n_days = .N, mean_c = mean(uhi_c), median_c = median(uhi_c),
  p90_c = quantile(uhi_c, 0.90), maximum_c = max(uhi_c)
), by = .(year, metric, elevation_treatment)]

leave_one_out <- data.table::rbindlist(lapply(stations$station_id, function(drop_id) {
  candidate <- daily_primary[station_id != drop_id]
  out <- make_group_uhi(candidate)
  if (!nrow(out)) return(NULL)
  out[, .(
    n_days = .N, mean_c = mean(uhi_c), median_c = median(uhi_c),
    p90_c = quantile(uhi_c, 0.90), maximum_c = max(uhi_c)
  ), by = .(metric, elevation_treatment)][,
    `:=`(excluded_station_id = drop_id,
         excluded_station = stations[station_id == drop_id, station_name],
         excluded_role = stations[station_id == drop_id, role])]
}))

data.table::fwrite(completeness, file.path(input_dir, "station_completeness.csv"))
data.table::fwrite(daily_primary, file.path(input_dir, "station_daily_metrics_primary.csv"))
data.table::fwrite(uhi_primary, file.path(input_dir, "group_uhi_daily_primary.csv"))
data.table::fwrite(summary_primary, file.path(input_dir, "uhi_summary_primary.csv"))
data.table::fwrite(qc_sensitivity, file.path(input_dir, "uhi_qc_sensitivity.csv"))
data.table::fwrite(yearly, file.path(input_dir, "uhi_summary_by_year.csv"))
data.table::fwrite(leave_one_out, file.path(input_dir, "uhi_leave_one_station_out.csv"))

theme_paper <- theme_minimal(base_family = "Helvetica", base_size = 9) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        plot.title.position = "plot", plot.title = element_text(face = "bold", size = 10),
        plot.tag = element_text(face = "bold", size = 12),
        legend.position = "top", legend.justification = "left",
        axis.title = element_text(size = 9), axis.text = element_text(colour = "#333333"))

plot_data <- uhi_primary[, day_class := fifelse(hot_day, "Hot days", "All July days")]
p1 <- ggplot(plot_data, aes(factor(year), uhi_c, fill = elevation_treatment)) +
  geom_hline(yintercept = 0, colour = "#999999", linewidth = 0.35) +
  geom_boxplot(width = 0.72, position = position_dodge(width = 0.78), outlier.size = 0.45,
               linewidth = 0.35, alpha = 0.72) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c("Raw" = "#377EB8", "Adjusted to 5 m" = "#E69F00")) +
  labs(tag = "a", title = "Urban-rural temperature contrasts by July",
       x = "Year", y = "Urban minus rural temperature (°C)", fill = NULL) +
  theme_paper + theme(legend.position = "none")

p2 <- ggplot(plot_data, aes(uhi_c, colour = elevation_treatment, fill = elevation_treatment)) +
  geom_vline(xintercept = 0, colour = "#999999", linewidth = 0.35) +
  geom_density(alpha = 0.18, linewidth = 0.65) +
  facet_grid(metric ~ day_class, scales = "free") +
  scale_colour_manual(values = c("Raw" = "#377EB8", "Adjusted to 5 m" = "#E69F00")) +
  scale_fill_manual(values = c("Raw" = "#377EB8", "Adjusted to 5 m" = "#E69F00")) +
  guides(fill = "none", colour = guide_legend(override.aes = list(fill = NA))) +
  labs(tag = "b", title = "Ordinary-July and hot-day distributions",
       x = "Urban minus rural temperature (°C)", y = "Density", colour = NULL, fill = NULL) +
  theme_paper

comparison_figure <- p1 / p2 + plot_layout(heights = c(1, 1.05))
comparison_path <- file.path(figure_dir, "midas_uhi_multiyear_2018_2023_20261111_v2.png")
ggsave(comparison_path, comparison_figure, width = 180, height = 220, units = "mm", dpi = 320,
       bg = "white")

london <- st_read(boundary_path, quiet = TRUE) |>
  st_transform(4326) |>
  st_union()
station_sf <- st_as_sf(stations, coords = c("longitude", "latitude"), crs = 4326)
map <- ggplot() +
  geom_sf(data = london, fill = "#F2F2F2", colour = "#666666", linewidth = 0.45) +
  geom_sf(data = station_sf, aes(colour = role, shape = role), size = 2.7, stroke = 0.6) +
  geom_text_repel(data = stations,
                  aes(longitude, latitude, label = station_name, colour = role),
                  size = 2.35, family = "Helvetica", seed = 20261111,
                  box.padding = 0.28, point.padding = 0.3,
                  min.segment.length = 0, max.overlaps = Inf,
                  segment.colour = "#8A8A8A", segment.size = 0.25,
                  show.legend = FALSE) +
  scale_colour_manual(values = c("Urban candidate" = "#0072B2", "Rural candidate" = "#D55E00")) +
  scale_shape_manual(values = c("Urban candidate" = 16, "Rural candidate" = 17)) +
  coord_sf(xlim = range(stations$longitude) + c(-0.08, 0.08),
           ylim = range(stations$latitude) + c(-0.06, 0.08), expand = FALSE) +
  labs(title = "Provisional MIDAS station network", subtitle = "July 2018-2023",
       colour = NULL, shape = NULL, x = NULL, y = NULL) +
  theme_paper +
  theme(panel.grid.major = element_line(colour = "#E5E5E5", linewidth = 0.25),
        axis.text = element_blank(), legend.position = "bottom")
map_path <- file.path(figure_dir, "midas_uhi_station_map_2018_2023_20261111_v1.png")
ggsave(map_path, map, width = 165, height = 145, units = "mm", dpi = 320, bg = "white")

manifest <- list(
  generated_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  script = "code/preprocessing_layers/analyze_midas_uhi_pilot.R",
  source_dataset = "MIDAS Open UK hourly weather observations v202607",
  years = 2018:2023,
  station_roles = "Provisional; not a final urban/rural classification",
  primary_qc = "version_num=1; retain air_temperature qc_status=0; deduplicate by station/time using latest record timestamp",
  sensitivity_qc = "include qc_status=1 suspect observations",
  daily_completeness = "daily extrema require at least 18 distinct hours; daytime mean uses 09:00-17:00 UTC and requires at least 7 of 9 hours; night-time mean uses 22:00-05:00 UTC and requires at least 6 of 8 hours",
  group_estimator = "median across available stations; require at least 3 urban and 3 rural stations per date",
  hot_day_definition = "rural group daily maximum at or above pooled July 90th percentile, separately by elevation treatment",
  elevation_sensitivity = "fixed 0.0065 C/m adjustment to 5 m; sensitivity only, not primary correction",
  output_figures = c(basename(comparison_path), basename(map_path))
)
jsonlite::write_json(manifest, file.path(input_dir, "analysis_manifest.json"),
                     pretty = TRUE, auto_unbox = TRUE)

cat("Analysed", length(files), "MIDAS files and", nrow(hourly), "deduplicated July records.\n")
print(summary_primary)
cat("Figures:\n", comparison_path, "\n", map_path, "\n")
