# Replay the original notebook's functions and plotting expression without
# executing its historical file-writing chunks. Only the Target filter is removed.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
repo <- normalizePath(args[1]); out <- args[2]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(stringr)
  library(ggplot2); library(patchwork)})
notebook <- file.path(repo,"code/invest_result_zonal_viz_1_temp.Rmd")
tmp <- tempfile(fileext=".R")
knitr::purl(notebook, output=tmp, quiet=TRUE)
expressions <- parse(tmp); unlink(tmp)
e <- new.env(parent=globalenv())
sys.source(file.path(repo,"code/func_colors.R"), e)
functions <- c("func_calculate_change_from_yri_lc0",
  "func_calculate_change_from_yr0_lc0", "func_combine_two_change_dfs",
  "func_plot_change_point")
for (name in functions) {
  found <- Filter(function(x) is.call(x) && identical(x[[1]],as.name("<-")) &&
    identical(x[[2]],as.name(name)), as.list(expressions))
  stopifnot(length(found)==1); eval(found[[1]], e)
}
e$shp_all_df <- readRDS(file.path(repo,"data/ucm_output_shp_stats_outcomes_by_borough.rds"))
e$year_baseline <- 2021; e$ind <- "avg_tmp_v"; e$ind_scale <- 1
e$ind_label <- "Temperature (°C)"; e$dir.fig <- out; e$ymd <- "20261111_original_v1"
e$aoi <- distinct(e$shp_all_df,GSS_CODE)
# Disable implicit old saves; export only explicit, versioned comparison figures.
e$func_ggsave <- function(...) invisible(NULL)
e$ggsave <- function(...) invisible(NULL)
old <- e$func_calculate_change_from_yri_lc0(e$shp_all_df,e$aoi)
total <- e$func_calculate_change_from_yr0_lc0(e$shp_all_df,e$aoi)
# Audit the original row-order subtraction against an explicit borough/year join.
base <- e$shp_all_df %>% filter(lc_scenario=="scenario0") %>%
  select(GSS_CODE,year,baseline=avg_tmp_v)
stopifnot(!anyDuplicated(base[c("GSS_CODE","year")]))
paired <- e$shp_all_df %>% filter(year>=2021,lc_scenario!="scenario0") %>%
  left_join(base,by=c("GSS_CODE","year")) %>% mutate(delta=avg_tmp_v-baseline)
keyed <- paired %>% group_by(year,lc_scenario) %>%
  summarise(key_mean=mean(delta),key_sd=sd(delta),.groups="drop")
check <- left_join(old,mutate(keyed,year=as.numeric(year)),by=c("year","lc_scenario"))
stopifnot(max(abs(check$value_diff_mean-check$key_mean))<1e-10,
          max(abs(check$sd-check$key_sd))<1e-10,all(old$n==33))
label <- function(x) mutate(x,lc_scenario=factor(lc_scenario,
  levels=names(e$scenario_labels),labels=unname(e$scenario_labels)))
e$stat_both_l <- e$func_combine_two_change_dfs(label(total),label(old))
lines <- readLines(notebook)
start <- grep('^```.*fig.3a}',lines)+1
end <- start+which(lines[start:length(lines)]=='```')[1]-2
panel_code <- parse(text=lines[start:end])
a <- eval(panel_code[[1]], e)  # exact original stem/point plotting expression
b <- e$func_plot_change_point(e$stat_both_l,show_x_label=TRUE,plot_avg=TRUE,plot_h=4)
# v2: restore an internal key like the original panel a; calculations unchanged.
# Nature asks for black text, restrained lines and 5–7 pt sans-serif artwork text.
a <- a + labs(x="Year", y="Temperature change (°C)") +
  theme(text=element_text(size=7,family="sans"),
        axis.text=element_text(size=6), axis.title=element_text(size=7),
        legend.position="inside",legend.position.inside=c(.07,.96),
        legend.justification.inside=c(0,1),
        legend.title=element_text(size=6),legend.text=element_text(size=6),
        legend.key.height=unit(9,"pt"),legend.key.width=unit(9,"pt"),
        panel.grid=element_blank()) + guides(colour=guide_legend(ncol=1))
for (layer in b$layers) {
  if (inherits(layer$geom,"GeomText")) {
    layer$aes_params$fontface <- "plain"; layer$aes_params$colour <- "grey25"
    layer$aes_params$size <- 2.1
    layer$mapping$y <- aes(y=ifelse(chg>=0,ci_upper+.025,ci_lower-.025))$y
    layer$aes_params$vjust <- .5
  }
  if (inherits(layer$geom,"GeomErrorbar")) layer$aes_params$linewidth <- .3
}
b <- b + labs(x=NULL,y="Temperature change (°C)") +
  theme_classic(base_size=7,base_family="sans") +
  theme(axis.text=element_text(size=6),axis.text.x=element_text(angle=45,hjust=1),
        legend.position="none") + scale_y_continuous(expand=expansion(mult=c(.12,.12)))
stem <- "figure3_legacy_replay_v02_20261111"
ggsave(file.path(out,paste0(stem,".png")),((a+b)+plot_annotation(tag_levels="a")) &
       theme(plot.tag=element_text(size=8,face="bold")),
       width=7.2,height=3.7,dpi=450,bg="white")
ggsave(file.path(out,paste0(stem,"_a.png")),a,width=3.6,height=3.7,dpi=450,bg="white")
ggsave(file.path(out,paste0(stem,"_b.png")),b,width=3.6,height=3.7,dpi=450,bg="white")
write.csv(old,file.path(out,paste0(stem,"_by_year.csv")),row.names=FALSE)
write.csv(total,file.path(out,paste0(stem,"_relative_2021.csv")),row.names=FALSE)
legacy <- old %>% filter(year!=2021) %>% group_by(lc_scenario) %>%
  summarise(original_future_mean=mean(value_diff_mean),.groups="drop")
legacy2050 <- old %>% filter(year==2050) %>%
  select(lc_scenario,original_2050=value_diff_mean)
map <- setNames(c("allbuilt","treerisk","treeopp","green10","green20",
  "green30","target10","target20","target30"),
  c("scenario1","scenario2_TR","scenario3_TO","scenario4_10","scenario4_20",
    "scenario4_30","scenario510","scenario520","scenario530"))
revised <- read.csv(file.path(out,"figure3_alternative_area_weighted_v01_20261111_citywide.csv"))
comparison <- legacy %>% left_join(legacy2050,by="lc_scenario") %>%
  mutate(scenario=unname(map[lc_scenario])) %>% left_join(revised,by="scenario") %>%
  transmute(scenario,original_future_mean,original_2050,
    revised_borough=equal_borough_mean_c,revised_citywide=change_c,
    period_averaging_effect=original_2050-original_future_mean,
    revised_inputs_and_mask_effect=revised_borough-original_2050,
    area_weighting_effect=revised_citywide-revised_borough)
write.csv(comparison,file.path(out,paste0(stem,"_comparison.csv")),row.names=FALSE)
print(comparison,digits=5)
jsonlite::write_json(list(notebook=notebook,
  notebook_sha256=digest::digest(file=notebook,algo="sha256"),
  input_sha256=digest::digest(file=file.path(repo,"data/ucm_output_shp_stats_outcomes_by_borough.rds"),algo="sha256"),
  script_sha256=digest::digest(file=file.path(repo,"code/post_processing_layers/reproduce_figure3_original.R"),algo="sha256"),
  checks="Original row-order mean AND SD match keyed borough/year pairing within 1e-10; 33 boroughs per scenario/year",
  changes="Target exclusion removed; internal panel-a legend; panel-b black regular labels and classic theme; combined tags; versioned saves. Original calculations retained.",
  caveat="Legacy year labels denote prescribed settings. Original error bars and their cross-period averages are not propagated model uncertainty."),
  file.path(out,paste0(stem,"_manifest.json")),pretty=TRUE,auto_unbox=TRUE)
