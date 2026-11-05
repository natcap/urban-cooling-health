# Run only the canonical Figure 6 chunks, not the exploratory notebook sections.
# Rscript code/run_figure6_candidates.R [YYYYMMDD] [--redraw]
# --redraw updates only the two candidate PNGs from saved numerical results.
suppressPackageStartupMessages({library(sf);library(dplyr);library(tidyr);library(readr)
  library(purrr);library(ggplot2);library(here);library(tibble)})
args <- commandArgs(TRUE)
redraw <- "--redraw" %in% args
args <- setdiff(args, "--redraw")
ymd <- if(length(args)) args[1] else format(Sys.Date(),"%Y%m%d")
stopifnot(grepl("^[0-9]{8}$",ymd))
health_file <- here("data/derived/health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv")
population_file <- here("data/derived/lsoa_population_2021_by_lsoa11cd.csv")
v <- st_read(here("data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg"),quiet=TRUE) %>% rename(SVI=overall_score)
h <- read_csv(health_file,show_col_types=FALSE) %>% filter(temperature_setting_c==25) %>%
  select(LSOA11CD,lc_scenario,indicator,cases_abs=deaths_averted)
p <- read_csv(population_file,show_col_types=FALSE) %>% select(LSOA11CD,population_2021)
stopifnot(!anyDuplicated(v$LSOA11CD),!anyDuplicated(p$LSOA11CD),!anyDuplicated(h[c("LSOA11CD","lc_scenario","indicator")]))
health_sf <- v %>% left_join(h,by="LSOA11CD") %>% left_join(p,by="LSOA11CD") %>%
  mutate(cases=1e5*cases_abs/population_2021)
stopifnot(all(is.finite(health_sf$cases)),all(health_sf$population_2021>0))
notebook <- readLines(here("code/equity-health.Rmd"),warn=FALSE)
chunk <- function(label) {
  start <- which(notebook==paste0("```{r ",label,"}"))
  stopifnot(length(start)==1)
  end <- start+which(notebook[(start+1):length(notebook)]=="```")[1]
  parse(text=notebook[(start+1):(end-1)])
}
protected <- list.files(here("figures"),recursive=TRUE,full.names=TRUE)
# A failed run may leave partial numerical files; completed candidates are guarded below.
if (!redraw) protected <- protected[!grepl("statistical_reporting/(population|area)_allocation/",protected)]
if (redraw) {
  candidate_names <- paste0("equity_health_All_cause_gam_Green30_",ymd,
    "_10_per_million_",c("population","area"),"_full_prediction.png")
  protected <- protected[!basename(protected) %in% candidate_names]
}
before <- tools::md5sum(protected)
for(allocation in c("population","area")) {
  target <- here("figures/figure6/statistical_reporting",paste0(allocation,"_allocation"))
  stopifnot(file.exists(file.path(target,"candidate_manifest.txt")) == redraw)
  options(urban_cooling.figure6_allocation=allocation)
  if (redraw) {
    # Reuse canonical data preparation, stopping before any fitting/diagnostics.
    expressions <- chunk("figure-6-production-models")
    for (expression in expressions) {
      if (is.call(expression) && identical(expression[[1]],as.name("<-")) &&
          identical(expression[[2]],as.name("figure6_neighbours"))) break
      eval(expression)
    }
    df_cont_raw <- figure6_sf %>% st_drop_geometry() %>%
      select(all_of(figure6_metrics)) %>%
      pivot_longer(everything(),names_to="Metric",values_to="x") %>%
      mutate(Metric_Label=recode(Metric,!!!figure6_labels))
    df_cont_curve <- read_csv(file.path(target,"figure6_gam_curve_data.csv"),show_col_types=FALSE)
    figure6_contrasts <- read_csv(file.path(target,"figure6_gam_p10_p90_contrasts.csv"),show_col_types=FALSE)
    eval(chunk("figure-6-production-plot"))
    next
  }
  eval(chunk("figure-6-production-models"))
  eval(chunk("figure-6-production-plot"))
  expected <- read_csv(here("figures/figure6/allocation_normalization_comparison/gam_contrasts.csv"),show_col_types=FALSE) %>%
    filter(.data$allocation==.env$allocation,normalization=="rate")
  actual <- filter(figure6_contrasts,Analysis=="Primary: all valid LSOAs")
  stopifnot(max(abs(actual$Difference_P90_minus_P10-expected$contrast[match(actual$Metric,expected$metric)]))<1e-4)
  writeLines(c(paste("Allocation:",allocation),"Full response prediction, intercept included.",
     "All 4,835 LSOAs fitted; display P1-P99; fitting scale per 100,000, export per million.",
     paste("Figure:",file.path(figure6_plot_dir,figure6_plot_name))),file.path(target,"candidate_manifest.txt"))
}
options(urban_cooling.figure6_allocation=NULL)
stopifnot(identical(before,tools::md5sum(protected)))
message(if (redraw) "Two candidate PNGs redrawn; all other figure/data files unchanged." else
  "Two canonical Figure 6 candidates generated; previous figure files unchanged.")
