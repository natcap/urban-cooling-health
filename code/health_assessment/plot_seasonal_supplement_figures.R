# Seasonal companions: reuse the established overlay and map renderers.
# Rscript code/health_assessment/plot_seasonal_supplement_figures.R RUN_FOLDER
suppressPackageStartupMessages({library(sf);library(dplyr);library(tidyr)
  library(readr);library(ggplot2);library(purrr)})
run <- commandArgs(TRUE)[1]
stopifnot(file.exists(file.path(run,"manifest.json")))
destinations <- c(
 "figures/figure6/scenario_comparison/figure6_supplement_green_overlay_seasonal_20261112_v2.png",
 "figures/figure6/scenario_comparison/figure6_supplement_target_overlay_seasonal_20261112_v2.png",
 "figures/figure7/png/fig7_ab_supplement_all_five_seasonal_20261112_v2.png")
stopifnot("--refresh" %in% commandArgs(TRUE) || !any(file.exists(destinations)))
protected <- list.files("figures",recursive=TRUE,full.names=TRUE)
protected <- setdiff(protected,destinations)
before <- tools::md5sum(protected)
chunk <- function(file,label) {
 x<-readLines(file,warn=FALSE);start<-grep(paste0("^```\\{r ",label,"[,}]"),x)
 stopifnot(length(start)==1)
 end<-start+which(x[(start+1):length(x)]=="```")[1]
 x[(start+1):(end-1)]
}
curves <- read_csv(file.path(run,"equity_comparison/gam_curves.csv"),show_col_types=FALSE) %>%
 filter(version=="revised_summer",scenario %in% c("green10","green20","green30","target10","target20","target30")) %>%
 transmute(metric,scenario=paste0(toupper(substr(scenario,1,1)),substring(scenario,2)),x,y=predicted_rate,se)
stopifnot(nrow(distinct(curves,metric,scenario))==30,all(is.finite(curves$y)),all(curves$se>=0))
metrics <- c("pIncomeDeprivation","p5under","p75over","pBAME","SVI")
main_overlay_only <- target_overlay_only <- overlay_only <- FALSE
# Extract only the original overlay block. No old annual curves are read and
# no model is refitted. The existing source checksum still guards SVI inputs.
out <- "figures/figure6/scenario_comparison"
src <- readLines("code/compare_figure6_scenarios.R",warn=FALSE)
start <- which(src=='source("code/func_colors.R")')
end <- grep('^stopifnot\\(identical\\(before',src)
stopifnot(length(start)==1,length(end)==1)
src <- src[start:(end-1)]
src <- gsub('nrow(overlay)==1000*length(overlay_scenarios)',
 'nrow(distinct(overlay,metric,scenario))==5*length(overlay_scenarios)',src,fixed=TRUE)
src <- gsub('Predicted deaths averted per million residents',
 'Predicted deaths averted per million residents per summer',src,fixed=TRUE)
src <- gsub('"_overlay.png"','"_overlay_seasonal_20261112_v2.png"',src,fixed=TRUE)
eval(parse(text=src))

# Share the exact seasonal input adapter used by the main-figure runner.
# Its setup/load/paired/bootstrap section has no persistent output writes.
adapter <- readLines("code/health_assessment/plot_seasonal_manuscript_figures.R",warn=FALSE)
start <- grep('^# Use the Figure 7',adapter)
end <- grep('^# Only the canonical',adapter)
stopifnot(length(start)==1,length(end)==1)
eval(parse(text=adapter[start:(end-1)]))
map_code <- chunk("code/equity-health-fig7-production.Rmd","maps")
map_code <- gsub("Benefit: deaths averted per million residents",
 "Benefit: deaths averted per million residents per summer",map_code,fixed=TRUE)
eval(parse(text=map_code))
# Keep the original two-row, five-indicator bivariate map layout and key.
ggsave(destinations[3],supplement_fig7$ab,width=12.5,height=6.2,units="in",
 dpi=600,device=ragg::agg_png,bg="white")
check <- read_csv(file.path(run,"equity_comparison/figure7_summary.csv"),show_col_types=FALSE) %>%
 filter(version=="revised_summer") %>% left_join(headline_results,by="metric")
stopifnot(max(abs(check$pct_green-check$pct_blue_Green30))<1e-8,
 max(abs(check$pct_target-check$pct_blue_Target30))<1e-8,
 identical(before,tools::md5sum(protected)),all(file.exists(destinations)))
message("Three seasonal supplementary PNGs saved; all earlier figures unchanged.")
