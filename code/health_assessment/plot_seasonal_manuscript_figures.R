# Reuse canonical manuscript plotting chunks; never overwrite annual figures.
# Rscript code/health_assessment/plot_seasonal_manuscript_figures.R RUN_FOLDER
suppressPackageStartupMessages({library(sf);library(dplyr);library(tidyr)
  library(readr);library(ggplot2);library(here);library(purrr)})
run <- commandArgs(TRUE)[1]
stopifnot(file.exists(file.path(run,"manifest.json")))
chunk <- function(file,label) {
  x <- readLines(file,warn=FALSE)
  start <- grep(paste0("^```\\{r ",label,"[,}]"),x)
  stopifnot(length(start)==1)
  end <- start+which(x[(start+1):length(x)]=="```")[1]
  x[(start+1):(end-1)]
}
fig6 <- "figures/figure6/manuscript/equity_health_All_cause_gam_Green30_20261112_10_per_million_population_full_prediction_seasonal_v2.png"
fig7 <- "figures/figure7/png/fig7_main_mapled_SVI_20261112_seasonal_v5.png"
refresh <- "--refresh" %in% commandArgs(TRUE)
stopifnot(refresh || (!file.exists(fig6) && !file.exists(fig7)))
protected <- list.files("figures",recursive=TRUE,full.names=TRUE)
protected <- setdiff(protected,c(fig6,fig7))
before <- tools::md5sum(protected)
v <- st_read("data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg",quiet=TRUE) %>%
  mutate(SVI=overall_score)
figure6_labels <- c(pIncomeDeprivation="Income deprivation",p5under="Age <5",
  p75over="Age ≥75",pBAME="Ethnic-minority population share",SVI="Social Vulnerability Index")
df_cont_raw <- v %>% st_drop_geometry() %>% select(all_of(names(figure6_labels))) %>%
  mutate(across(everything(),~as.numeric(scale(.x)))) %>%
  pivot_longer(everything(),names_to="Metric",values_to="x") %>%
  mutate(Metric_Label=recode(Metric,!!!figure6_labels))
# Saved full-response predictions and SEs retain the exact fitted seasonal models.
df_cont_curve <- read_csv(file.path(run,"equity_comparison/gam_curves.csv"),show_col_types=FALSE) %>%
  filter(version=="revised_summer",scenario=="green30") %>%
  transmute(Metric=metric,x,y=predicted_rate,se,Metric_Label=recode(metric,!!!figure6_labels))
plot_code <- chunk("code/equity-health.Rmd","figure-6-production-plot")
plot_code <- plot_code[seq_len(grep("^figure6_plot_dir <-",plot_code)-1)]
plot_code <- plot_code[plot_code != "figure6_plot"]
plot_code <- gsub('Predicted deaths averted per million residents',
  'Predicted deaths averted per million residents per summer',plot_code,fixed=TRUE)
eval(parse(text=plot_code))
ggsave(fig6,figure6_plot,width=7.5,height=5.5,units="in",dpi=600,device=ragg::agg_png)

# Use the Figure 7 production layout, but supply seasonal rows explicitly.
# Do not load old annual Monte Carlo draws or export annual uncertainty metadata.
eval(parse(text=chunk("code/equity-health-fig7-production.Rmd","setup")))
new <- read_csv(file.path(run,"lsoa_results.csv"),show_col_types=FALSE) %>%
  filter(reconstruction=="worldclim_aligned",ssp=="ssp370",rr_variant=="central",
         cause=="all_cause",scenario %in% c("green30","target30")) %>%
  transmute(LSOA11CD,LSOA11NM=LSOA11CD,Neighborhood_name=LSOA11CD,
    temperature_setting_c=25,cause,indicator="All_cause",
    lc_scenario=if_else(scenario=="green30","Green30","Target30"),deaths_averted)
health_zonal_file <- tempfile(fileext=".csv")
write_csv(new,health_zonal_file)
eval(parse(text=chunk("code/equity-health-fig7-production.Rmd","load-data")))
benefit_unit <- "Deaths averted per million residents per modelled summer"
eval(parse(text=chunk("code/equity-health-fig7-production.Rmd","paired-analysis")))
eval(parse(text=chunk("code/equity-health-fig7-production.Rmd","bootstrap")))
# Only the canonical map-led composition; no legacy supplementary exports.
plot_code <- chunk("code/equity-health-fig7-production.Rmd","save-outputs")
plot_code <- plot_code[seq_len(grep('^dir.create\\(output_dir',plot_code)-1)]
plot_code <- plot_code[!grepl("^svi_balance <-",plot_code)]
plot_code <- gsub('Benefit: deaths averted per million residents.',
  'Benefit: deaths averted per million residents per modelled summer.',plot_code,fixed=TRUE)
eval(parse(text=plot_code))
ggsave(fig7,fig7_mapled,width=180,height=170,units="mm",dpi=600,device=ragg::agg_png,bg="white")
check <- read_csv(file.path(run,"equity_comparison/figure7_summary.csv"),show_col_types=FALSE) %>%
  filter(version=="revised_summer") %>% left_join(headline_results,by="metric")
stopifnot(max(abs(check$pct_green-check$pct_blue_Green30))<1e-8,
          max(abs(check$pct_target-check$pct_blue_Target30))<1e-8,
          identical(before,tools::md5sum(protected)))
message("Seasonal manuscript candidates saved; existing figures unchanged.")
