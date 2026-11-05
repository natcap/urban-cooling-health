# Run from repository root after setting URBAN_COOLING_DATA_ROOT.
# Reuse the author's plotting notebook; no raster/model reruns or image stitching.
stopifnot(requireNamespace("rmarkdown"), requireNamespace("patchwork"))
out <- "figures/figure2/manuscript"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
build <- tempfile("figure2-panels-")
dir.create(build)
panels <- list()
for (outcome in c("temp_baseline", "TCC")) {
  panel_path <- file.path(build, paste0(outcome, ".rds"))
  Sys.setenv(FIGURE2_OUTCOME = outcome, FIGURE2_PANEL_EXPORT = panel_path)
  rmarkdown::render("code/equity-temp-tcc.Rmd",
    output_file = paste0(outcome, ".html"), output_dir = build,
    envir = new.env(), quiet = TRUE)
  panels[[outcome]] <- readRDS(panel_path)
}
Sys.unsetenv(c("FIGURE2_OUTCOME", "FIGURE2_PANEL_EXPORT"))
# Require identical canonical scores and quintile memberships before assembly.
a <- panels$temp_baseline$svi
b <- panels$TCC$svi
stopifnot(!anyDuplicated(a$geoid), setequal(a$geoid, b$geoid))
b <- b[match(a$geoid, b$geoid), ]
stopifnot(identical(a$SVI, b$SVI), identical(a$SVI_q, b$SVI_q))
combined <- patchwork::wrap_plots(
  panels$temp_baseline$plot, panels$TCC$plot, nrow = 1, widths = c(1, 1)) +
  patchwork::plot_annotation(tag_levels = "a", theme = ggplot2::theme(
    plot.tag = ggplot2::element_text(face = "bold", size = 12)))
stem <- file.path(out, "figure2_baseline_temperature_tree_cover")
ggplot2::ggsave(paste0(stem, ".png"), combined, width = 7.2, height = 3.2,
               units = "in", dpi = 600, bg = "white")
# PNG only, as requested; no PDF or external graphics-library dependency.
stopifnot(file.exists(paste0(stem, ".png")))
# Small build record; source CSVs/geometry remain in their existing locations.
writeLines(c("Panel a: July 2021 CHELSA baseline; panel b: baseline tree cover.",
             "Both panels use the same canonical four-theme SVI quintiles.",
             "Existing notebook boxplot, letter and percentile-zoom rules retained.",
             capture.output(sessionInfo())), paste0(stem, "_session.txt"))
message("Saved ", stem, ".png")
