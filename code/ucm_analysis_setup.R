# Shared setup for the invest_result_zonal_viz_* analysis notebooks.
#
# Keep machine-specific paths in environment variables, not tracked scripts.
# Existing notebooks can request the historical variable names during gradual
# migration; new code should read paths and constants from the returned object.

ucm_analysis_setup <- function(
    data_root = Sys.getenv("URBAN_COOLING_DATA_ROOT", unset = ""),
    baseline_year = 2021L,
    run_date = Sys.getenv(
      "URBAN_COOLING_RUN_DATE", unset = format(Sys.Date(), "%Y%m%d")
    ),
    export_legacy_names = FALSE,
    envir = parent.frame()) {
  required_packages <- c(
    "here", "readr", "dplyr", "tidyr", "stringr", "lubridate",
    "ggplot2", "sf", "purrr"
  )
  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_packages)) {
    stop(
      "Install missing R package(s): ",
      paste(missing_packages, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(lapply(
    required_packages,
    function(package) library(package, character.only = TRUE, warn.conflicts = FALSE)
  ))

  if (!nzchar(data_root)) {
    stop(
      paste(
        "Set URBAN_COOLING_DATA_ROOT to the Wellcome Trust Project Data",
        "folder before running or knitting this notebook. See .Renviron.example."
      ),
      call. = FALSE
    )
  }
  data_root <- normalizePath(path.expand(data_root), mustWork = FALSE)
  if (!dir.exists(data_root)) {
    stop("URBAN_COOLING_DATA_ROOT does not exist: ", data_root, call. = FALSE)
  }

  paths <- list(
    data_root = data_root,
    input_root = file.path(
      data_root, "1_preprocess", "UrbanCoolingModel", "OfficialWorkingInputs"
    ),
    aoi_dir = file.path(
      data_root, "1_preprocess", "UrbanCoolingModel", "OfficialWorkingInputs", "AOIs"
    ),
    figure_dir = file.path(data_root, "3_final", "UCM_figures"),
    ucm_output_dir = file.path(
      data_root, "2_postprocess_intermediate", "UCM_official_runs"
    ),
    work_productivity_dir = file.path(
      data_root, "2_postprocess_intermediate", "UCM_official_runs",
      "new_work_intensity"
    ),
    repository_data_dir = here::here("data")
  )
  required_directories <- paths[c("input_root", "aoi_dir", "ucm_output_dir")]
  missing_directories <- names(required_directories)[
    !vapply(required_directories, dir.exists, logical(1))
  ]
  if (length(missing_directories)) {
    details <- paste(
      paste0(missing_directories, "=", unlist(required_directories[missing_directories])),
      collapse = "; "
    )
    stop("Required project-data folder(s) missing: ", details, call. = FALSE)
  }

  helper_paths <- here::here(
    "code", c("func_ggsave.R", "func_get_color_scale.R", "func_colors.R")
  )
  missing_helpers <- helper_paths[!file.exists(helper_paths)]
  if (length(missing_helpers)) {
    stop("Required helper file(s) missing: ", paste(missing_helpers, collapse = ", "))
  }
  invisible(lapply(helper_paths, source, local = envir))

  config <- list(
    paths = paths,
    baseline_year = as.integer(baseline_year),
    run_date = as.character(run_date)
  )
  class(config) <- c("ucm_analysis_config", "list")

  if (isTRUE(export_legacy_names)) {
    list2env(
      list(
        "dir.g" = paths$data_root,
        "dir.aoi" = paths$aoi_dir,
        "dir.fig" = paths$figure_dir,
        "dir_ucm_out" = paths$ucm_output_dir,
        "dir_prod_new" = paths$work_productivity_dir,
        "year_baseline" = config$baseline_year,
        "ymd" = config$run_date
      ),
      envir = envir
    )
  }

  config
}

print.ucm_analysis_config <- function(x, ...) {
  cat("Urban Cooling analysis configuration\n")
  cat("  data root:     ", x$paths$data_root, "\n", sep = "")
  cat("  baseline year: ", x$baseline_year, "\n", sep = "")
  cat("  run date:      ", x$run_date, "\n", sep = "")
  invisible(x)
}
