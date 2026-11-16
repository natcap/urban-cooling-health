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
fig6 <- "figures/figure6/manuscript/figure6_svi_green30_target30_seasonal_v03.png"
fig7 <- "figures/figure7/png/figure7_svi_main_seasonal_v06.png"
fig7_supp <- "figures/figure7/png/figure7_supp_indicator_shares_seasonal_v01.png"
refresh <- "--refresh" %in% commandArgs(TRUE)
stopifnot(refresh || !any(file.exists(c(fig6,fig7,fig7_supp))))
protected <- list.files("figures",recursive=TRUE,full.names=TRUE)
protected <- setdiff(protected,c(fig6,fig7,fig7_supp))
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
  filter(version=="revised_summer",scenario %in% c("green30","target30"),metric=="SVI") %>%
  transmute(Metric=scenario,x,y=predicted_rate,se,
            Metric_Label=if_else(scenario=="green30","a   Green30","b   Target30"))
# Reuse the existing density/arrows style, with identical SVI support in both panels.
# These are saved full-response predictions, not re-fitted or centred smooth terms.
df_cont_raw <- bind_rows(lapply(c("green30","target30"), function(sc)
  df_cont_raw %>% filter(Metric=="SVI") %>% mutate(Metric=sc,
    Metric_Label=if_else(sc=="green30","a   Green30","b   Target30"))))
figure6_labels <- c(green30="a   Green30",target30="b   Target30")
plot_code <- chunk("code/equity-health.Rmd","figure-6-production-plot")
plot_code <- plot_code[seq_len(grep("^figure6_plot_dir <-",plot_code)-1)]
plot_code <- plot_code[plot_code != "figure6_plot"]
plot_code <- gsub('Predicted deaths averted per million residents',
  'Predicted deaths averted per million residents per summer',plot_code,fixed=TRUE)
eval(parse(text=plot_code))
# Match Figure 4's canonical scenario palette; keep density/annotations grey.
palette_env <- new.env()
sys.source("code/func_colors.R", envir=palette_env)
for (i in seq_along(figure6_plot$layers)) {
  layer <- figure6_plot$layers[[i]]
  if (inherits(layer$geom,"GeomLine")) {
    layer$aes_params$colour <- NULL
    layer$mapping$colour <- aes(colour=Metric)$colour
  } else if (inherits(layer$geom,"GeomRibbon") && "se" %in% names(layer$data)) {
    layer$aes_params$fill <- NULL
    layer$mapping$fill <- aes(fill=Metric)$fill
  }
  figure6_plot$layers[[i]] <- layer
}
figure6_plot <- figure6_plot +
  scale_colour_manual(values=palette_env$scenario_colors,guide="none") +
  scale_fill_manual(values=palette_env$scenario_colors,guide="none")
figure6_plot <- figure6_plot + facet_wrap(~Metric_Label,nrow=1,scales="fixed") +
  labs(x="Social Vulnerability Index (standardized z-score)",
       y="Predicted deaths averted per million residents\nper modelled summer",subtitle=NULL) +
  theme(strip.text=element_text(face="bold",hjust=0,size=10))
ggsave(fig6,figure6_plot,width=7.5,height=3.7,units="in",dpi=600,device=ragg::agg_png)
if ("--figure6-only" %in% commandArgs(TRUE)) {
  stopifnot(file.exists(fig6),file.info(fig6)$size>1000,
            identical(before,tools::md5sum(protected)))
  message("Figure 6 refreshed; other figures unchanged.")
  quit(save="no")
}

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
# Include every component in the standalone supplement (the historical main
# panel omitted under-five share). This changes display selection, not estimates.
main_vulnerability_vars <- vulnerability_vars
plot_code <- chunk("code/equity-health-fig7-production.Rmd","save-outputs")
plot_code <- plot_code[seq_len(grep('^dir.create\\(output_dir',plot_code)-1)]
plot_code <- plot_code[!grepl("^svi_balance <-",plot_code)]
plot_code <- gsub('Benefit: deaths averted per million residents.',
  'Benefit: deaths averted per million residents per modelled summer.',plot_code,fixed=TRUE)
eval(parse(text=plot_code))
# Keep a–c only in the main figure; retain the unchanged five-indicator panel
# as a separate supplement. Fixed thresholds and all calculated classes are unchanged.
fig7_mapled <- ggdraw() +
  draw_plot(plot_grid(left_column,
    panel_header(distribution_plot,"c","Benefit distribution"),
    nrow=1,rel_widths=c(1,1.15)),0,0.045,1,0.955) +
  draw_label("Benefit: deaths averted per million residents per modelled summer.",
             x=.03,y=.022,hjust=0,size=6)
ggsave(fig7,fig7_mapled,width=180,height=170,units="mm",dpi=600,device=ragg::agg_png,bg="white")
ggsave(fig7_supp,panel_header(gap_plot,"","Change in low-benefit share"),
       width=125,height=100,units="mm",dpi=600,device=ragg::agg_png,bg="white")
check <- read_csv(file.path(run,"equity_comparison/figure7_summary.csv"),show_col_types=FALSE) %>%
  filter(version=="revised_summer") %>% left_join(headline_results,by="metric")
stopifnot(max(abs(check$pct_green-check$pct_blue_Green30))<1e-8,
          max(abs(check$pct_target-check$pct_blue_Target30))<1e-8,
          all(file.exists(c(fig6,fig7,fig7_supp))),
          all(file.info(c(fig6,fig7,fig7_supp))$size > 1000),
          identical(before,tools::md5sum(protected)))
message("Seasonal manuscript candidates saved; existing figures unchanged.")
