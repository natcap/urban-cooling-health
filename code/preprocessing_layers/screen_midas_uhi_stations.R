#!/usr/bin/env Rscript

# Screen the provisional MIDAS UHI network using official station metadata,
# MIDAS temperature QC and ESA WorldCover exposure. The resulting one-urban-
# versus-three-rural contrast is a sensitivity diagnostic, not calibration.

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: screen_midas_uhi_stations.R PILOT_DIR WORLDCOVER_TIF FIGURE_DIR")
}
pilot_dir <- normalizePath(args[[1]], mustWork = TRUE)
worldcover_path <- normalizePath(args[[2]], mustWork = TRUE)
figure_dir <- normalizePath(args[[3]], mustWork = TRUE)

stations <- data.table(
  station_id = c("00697", "62084", "00723", "62073", "00709",
                 "00719", "30620", "00487", "00744", "00458"),
  station_name = c("St James's Park", "Battersea Heliport", "Kew Gardens",
                   "Teddington, Bushy Park", "Northolt", "Wisley",
                   "Charlwood", "Writtle", "East Malling", "Woburn"),
  original_role = rep(c("Urban candidate", "Rural candidate"), each = 5),
  latitude = c(51.505, 51.470, 51.482, 51.422, 51.549,
               51.311, 51.144, 51.734, 51.288, 52.014),
  longitude = c(-0.131, -0.179, -0.294, -0.341, -0.417,
                -0.476, -0.229, 0.429, 0.448, -0.596),
  elevation_m = c(5, 5, 6, 9, 33, 38, 67, 32, 33, 89)
)

target_numeric_ids <- as.integer(stations$station_id)

extract_source_dates <- function(path) {
  lines <- readLines(path, warn = FALSE)
  output <- rbindlist(lapply(target_numeric_ids, function(id) {
    hit <- grep(sprintf("^\\s*0*%d,", id), lines, value = TRUE)
    if (length(hit) != 1L) stop("Expected one SRCE record for source ", id)
    dates <- regmatches(hit, gregexpr("[0-9]{4}-[0-9]{2}-[0-9]{2}", hit))[[1]]
    data.table(
      station_id = sprintf("%05d", id),
      source_begin_date = dates[1],
      source_end_date = dates[2],
      source_record_update_date = dates[3]
    )
  }))
  output
}

extract_relevant_remarks <- function(path) {
  lines <- readLines(path, warn = FALSE)
  records <- rbindlist(lapply(target_numeric_ids, function(id) {
    hit <- grep(sprintf("^\\s*0*%d,", id), lines, value = TRUE)
    if (!length(hit)) return(NULL)
    rbindlist(lapply(hit, function(line) {
      m <- regexec(
        "^\\s*([0-9]+),\\s*([0-9]+),([^,]+),([0-9]{4}-[0-9]{2}-[0-9]{2}),(.*),\\s*1001,([0-9]{4}-[0-9]{2}-[0-9]{2})$",
        line
      )
      z <- regmatches(line, m)[[1]]
      if (!length(z)) return(NULL)
      data.table(
        station_id = sprintf("%05d", as.integer(z[2])),
        remark_number = as.integer(z[3]),
        remark_type = trimws(z[4]),
        remark_begin_date = z[5],
        remark_text = trimws(z[6]),
        remark_end_date = z[7]
      )
    }), fill = TRUE)
  }), fill = TRUE)
  records[
    remark_type %chin% c("SITE INFORMATION", "OBSERVING PRACTICE", "IDENTIFIERS") |
      grepl("TEMPERATURE|DRY BULB|ALL ELEMENTS|HEAT SOURCE|SUSPECT", remark_text,
            ignore.case = TRUE)
  ]
}

worldcover <- rast(worldcover_path)
if (!grepl("longlat", crs(worldcover, proj = TRUE))) {
  stop("WorldCover input must be in geographic coordinates for this bounded extractor")
}

exposure_for_station <- function(row, radius_m) {
  margin_x <- radius_m / (111320 * cos(row$latitude * pi / 180)) * 1.2
  margin_y <- radius_m / 110540 * 1.2
  local <- crop(worldcover, ext(row$longitude - margin_x, row$longitude + margin_x,
                                row$latitude - margin_y, row$latitude + margin_y))
  point <- vect(data.frame(x = row$longitude, y = row$latitude),
                geom = c("x", "y"), crs = "EPSG:4326")
  distances <- distance(local, point)
  values_local <- values(mask(local, distances <= radius_m, maskvalues = 0), mat = FALSE)
  values_local <- values_local[!is.na(values_local) & values_local != 0]
  data.table(
    station_id = row$station_id,
    radius_m = radius_m,
    valid_cells = length(values_local),
    built_fraction = mean(values_local == 50),
    tree_fraction = mean(values_local == 10),
    grass_fraction = mean(values_local == 30),
    cropland_fraction = mean(values_local == 40),
    water_fraction = mean(values_local == 80),
    natural_green_fraction = mean(values_local %in% c(10, 20, 30, 90))
  )
}

exposure <- rbindlist(lapply(seq_len(nrow(stations)), function(i) {
  rbindlist(lapply(c(500, 1000, 2000), function(radius) {
    exposure_for_station(stations[i], radius)
  }))
}))

completeness <- fread(file.path(pilot_dir, "station_completeness.csv"))
completeness[, station_id := sprintf("%05d", as.integer(station_id))]
qc_station <- completeness[, .(
  numeric_temperature_hours = sum(numeric_temperature_hours),
  valid_primary_hours = sum(valid_primary_hours),
  suspect_temperature_hours = sum(suspect_temperature_hours),
  minimum_year_completeness_pct = min(primary_completeness_pct)
), by = station_id]
qc_station[, suspect_fraction := fifelse(
  numeric_temperature_hours > 0,
  suspect_temperature_hours / numeric_temperature_hours,
  NA_real_
)]

source_dates <- extract_source_dates(file.path(pilot_dir, "metadata", "SRCE.DATA"))
remarks <- extract_relevant_remarks(file.path(pilot_dir, "metadata", "SRCR.DATA"))

screen <- merge(stations, exposure[radius_m == 1000], by = "station_id")
screen <- merge(screen, qc_station, by = "station_id")
screen <- merge(screen, source_dates, by = "station_id")
screen[, official_all_elements_suspect := station_id %chin%
         remarks[grepl("ALL ELEMENTS ARE SUSPECT", remark_text, ignore.case = TRUE), station_id]]

# Predeclared diagnostic rules. These are screening thresholds, not universal
# LCZ definitions: >=50% built within 1 km identifies urban background;
# <=10% built identifies rural background. Intermediate sites are transition,
# park or suburban settings and are excluded from the screened contrast.
screen[, screened_role := fcase(
  official_all_elements_suspect | suspect_fraction >= 0.5, "Reject: official/QC concern",
  built_fraction >= 0.50, "Urban-background candidate",
  built_fraction <= 0.10 & elevation_m <= 100, "Rural-background candidate",
  default = "Transition/green-space site"
)]
screen[, include_screened_contrast := screened_role %chin%
         c("Urban-background candidate", "Rural-background candidate")]
screen[, screening_reason := fcase(
  official_all_elements_suspect, "Official remark: all elements suspect from site heat/exhaust/rotor/building effects",
  suspect_fraction >= 0.5, "At least half of numeric temperatures carry MIDAS suspect status",
  built_fraction >= 0.50, "At least 50% built cover within 1 km; no overriding QC exclusion",
  built_fraction <= 0.10 & elevation_m <= 100, "At most 10% built cover within 1 km and elevation at most 100 m",
  default = "Built cover lies between rural and urban thresholds"
)]

fwrite(exposure, file.path(pilot_dir, "station_worldcover_exposure.csv"))
fwrite(remarks, file.path(pilot_dir, "station_relevant_official_remarks.csv"))
fwrite(screen, file.path(pilot_dir, "station_screening_decision.csv"))

daily <- fread(file.path(pilot_dir, "station_daily_metrics_primary.csv"))
daily[, station_id := sprintf("%05d", as.integer(station_id))]
urban_ids <- screen[screened_role == "Urban-background candidate", station_id]
rural_ids <- screen[screened_role == "Rural-background candidate", station_id]
if (length(urban_ids) != 1L || length(rural_ids) < 3L) {
  stop("Expected one urban-background and at least three rural-background candidates")
}

metric_long <- melt(
  daily[station_id %chin% c(urban_ids, rural_ids)],
  id.vars = c("station_id", "station_name", "year", "date", "elevation_treatment"),
  measure.vars = c("daily_tmax_c", "daytime_mean_c", "nocturnal_mean_c"),
  variable.name = "metric", value.name = "temperature_c", na.rm = TRUE
)
metric_long[, screened_group := fifelse(station_id %chin% urban_ids, "Urban", "Rural")]
grouped <- metric_long[, .(
  group_temperature_c = median(temperature_c),
  station_count = uniqueN(station_id)
), by = .(date, year, elevation_treatment, metric, screened_group)]
screened_uhi <- dcast(grouped, date + year + elevation_treatment + metric ~ screened_group,
                      value.var = c("group_temperature_c", "station_count"))
setnames(screened_uhi,
         c("group_temperature_c_Urban", "group_temperature_c_Rural",
           "station_count_Urban", "station_count_Rural"),
         c("urban_temperature_c", "rural_temperature_c", "urban_stations", "rural_stations"))
screened_uhi <- screened_uhi[urban_stations == 1L & rural_stations >= 3L]
screened_uhi[, uhi_c := urban_temperature_c - rural_temperature_c]
screened_uhi[, metric := fcase(
  metric == "daily_tmax_c", "Daily maximum",
  metric == "daytime_mean_c", "Daytime mean",
  metric == "nocturnal_mean_c", "Night-time mean"
)]
hot_threshold <- screened_uhi[metric == "Daily maximum",
  .(hot_threshold_c = quantile(rural_temperature_c, 0.90)), by = elevation_treatment]
hot_dates <- merge(screened_uhi[metric == "Daily maximum",
                                .(date, elevation_treatment, rural_temperature_c)],
                   hot_threshold, by = "elevation_treatment")
hot_dates[, hot_day := rural_temperature_c >= hot_threshold_c]
screened_uhi <- merge(screened_uhi, hot_dates[, .(date, elevation_treatment, hot_day)],
                      by = c("date", "elevation_treatment"), all.x = TRUE)

summarise_subset <- function(x, subset_name) {
  x[, .(
    n_days = .N, mean_c = mean(uhi_c), median_c = median(uhi_c),
    p10_c = quantile(uhi_c, 0.10), p90_c = quantile(uhi_c, 0.90),
    maximum_c = max(uhi_c)
  ), by = .(metric, elevation_treatment)][, day_subset := subset_name]
}
screened_summary <- rbind(
  summarise_subset(screened_uhi, "All July days"),
  summarise_subset(screened_uhi[hot_day == TRUE], "Hot days")
)
setcolorder(screened_summary, c("day_subset", "metric", "elevation_treatment",
                               "n_days", "mean_c", "median_c", "p10_c", "p90_c", "maximum_c"))
fwrite(screened_uhi, file.path(pilot_dir, "uhi_screened_daily_sensitivity.csv"))
fwrite(screened_summary, file.path(pilot_dir, "uhi_screened_summary_sensitivity.csv"))

role_levels <- c("Urban-background candidate", "Rural-background candidate",
                 "Transition/green-space site", "Reject: official/QC concern")
screen[, screened_role := factor(screened_role, levels = role_levels)]
screen_plot <- screen[order(built_fraction)]
screen_plot[, station_label := factor(station_name, levels = station_name)]
colors <- c(
  "Urban-background candidate" = "#0072B2",
  "Rural-background candidate" = "#009E73",
  "Transition/green-space site" = "#999999",
  "Reject: official/QC concern" = "#D55E00"
)
theme_paper <- theme_minimal(base_family = "Helvetica", base_size = 9) +
  theme(panel.grid.minor = element_blank(), plot.title.position = "plot",
        plot.title = element_text(face = "bold", size = 10),
        plot.tag = element_text(face = "bold", size = 12),
        axis.text = element_text(colour = "#333333"), legend.position = "bottom")

p1 <- ggplot(screen_plot, aes(built_fraction * 100, station_label, colour = screened_role)) +
  geom_vline(xintercept = c(10, 50), linetype = "dashed", colour = "#777777", linewidth = 0.4) +
  geom_point(size = 2.5) +
  scale_colour_manual(values = colors, drop = FALSE) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
  scale_x_continuous(limits = c(0, 75), breaks = c(0, 10, 25, 50, 75)) +
  labs(tag = "a", title = "Station exposure screen", subtitle = "ESA WorldCover built cover within 1 km",
       x = "Built cover (%)", y = NULL, colour = NULL) + theme_paper +
  theme(legend.text = element_text(size = 8))

plot_uhi <- rbind(
  copy(screened_uhi)[, day_subset := "All July days"],
  copy(screened_uhi)[hot_day == TRUE][, day_subset := "Hot days"]
)
p2 <- ggplot(plot_uhi, aes(uhi_c, colour = elevation_treatment, fill = elevation_treatment)) +
  geom_vline(xintercept = 0, colour = "#777777", linewidth = 0.4) +
  geom_density(alpha = 0.16, linewidth = 0.65) +
  facet_grid(metric ~ day_subset, scales = "free") +
  scale_colour_manual(values = c("Raw" = "#377EB8", "Adjusted to 5 m" = "#E69F00")) +
  scale_fill_manual(values = c("Raw" = "#377EB8", "Adjusted to 5 m" = "#E69F00")) +
  guides(fill = "none") +
  labs(tag = "b", title = "Screened-network sensitivity",
       subtitle = "One urban-background station versus median of three rural-background stations",
       x = "Urban minus rural temperature (°C)", y = "Density", colour = NULL) + theme_paper

figure <- p1 / p2 + plot_layout(heights = c(0.9, 1.25))
figure_path <- file.path(figure_dir, "midas_uhi_station_screen_2018_2023_20261111_v2.png")
ggsave(figure_path, figure, width = 180, height = 245, units = "mm", dpi = 320, bg = "white")

metadata_files <- file.path(pilot_dir, "metadata",
                            c("SRCE.DATA", "SRCR.DATA", "MIDAS_METADATA_README.txt", "QC_J_flags.html"))
manifest <- list(
  generated_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  script = "code/preprocessing_layers/screen_midas_uhi_stations.R",
  exposure_source = worldcover_path,
  exposure_source_md5 = as.character(tools::md5sum(worldcover_path)),
  exposure_classes = list(built = 50, tree = 10, grass = 30, cropland = 40,
                          water = 80, natural_green = c(10, 20, 30, 90)),
  screening_rules = list(
    reject = "official all-elements-suspect remark or >=50% suspect temperatures",
    urban_background = ">=50% built cover within 1 km without overriding QC exclusion",
    rural_background = "<=10% built cover within 1 km and elevation <=100 m",
    transition = "all other stations"
  ),
  screened_urban_station_ids = urban_ids,
  screened_rural_station_ids = rural_ids,
  warning = "Only one urban-background station passed; screened contrast is sensitivity only",
  metadata_snapshots = setNames(as.list(as.character(tools::md5sum(metadata_files))),
                                basename(metadata_files)),
  output_figure = basename(figure_path)
)
write_json(manifest, file.path(pilot_dir, "station_screening_manifest.json"),
           pretty = TRUE, auto_unbox = TRUE)

cat("Screening decisions:\n")
print(screen[, .(station_id, station_name, built_fraction, suspect_fraction,
                 screened_role, screening_reason)])
cat("\nScreened sensitivity summary:\n")
print(screened_summary)
cat("\nFigure:", figure_path, "\n")
