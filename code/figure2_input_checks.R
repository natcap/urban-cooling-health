# Input contracts only: these checks do not change the climate calculation.
figure2_missing_ids <- c("1674", "4696", "4710", "4828")

validate_figure2_zonal <- function(data, expected_ids, key = "borough",
                                   value = "mean") {
  if (!all(c(key, value) %in% names(data))) stop("Missing climate key/value columns")
  ids <- as.character(data[[key]])
  if (anyNA(ids) || anyDuplicated(ids) ||
      !setequal(ids, as.character(expected_ids))) {
    stop("Climate area IDs are missing, duplicated or do not match the LSOA layer")
  }
  if (!is.numeric(data[[value]])) stop("Climate values must be numeric")
  missing_ids <- ids[!is.finite(data[[value]])]
  if (any(!missing_ids %in% figure2_missing_ids)) {
    stop("Unexpected missing climate values outside the four documented source gaps")
  }
  invisible(data)
}

validate_figure2_models <- function(meta) {
  required <- c("ACCESS-CM2", "IPSL-CM6A-LR", "MRI-ESM2-0")
  if (!nrow(meta) || anyNA(meta[c("model", "scenario", "period")])) {
    stop("Missing or unrecognized climate model metadata")
  }
  groups <- split(meta$model, interaction(meta$scenario, meta$period, drop = TRUE))
  if (any(vapply(groups, function(x) length(x) != length(required) ||
                 !setequal(x, required), logical(1)))) {
    stop("Each scenario/period requires exactly one file for each of the three models")
  }
  invisible(meta)
}

select_figure2_temperature <- function(data, scenario = "ssp370", year = 2050) {
  if (!all(c("scenario", "year", "geoid") %in% names(data))) {
    stop("Temperature cache lacks scenario, year or geoid")
  }
  selected <- data[!is.na(data$scenario) & !is.na(data$year) &
                     tolower(as.character(data$scenario)) == tolower(scenario) &
                     as.character(data$year) == as.character(year), ]
  if (!nrow(selected) || anyDuplicated(as.character(selected$geoid))) {
    stop("Requested temperature scenario/year is absent or has duplicate LSOAs")
  }
  selected
}
