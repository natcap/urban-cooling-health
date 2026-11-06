# Read-only check: Figure 7 per-million rescaling must preserve all classes.
suppressPackageStartupMessages({library(sf);library(dplyr);library(tidyr)})
v<-st_drop_geometry(st_read("data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg",quiet=TRUE))
v$SVI<-v$overall_score
h<-read.csv("data/derived/health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv") %>%
  filter(temperature_setting_c==25,indicator=="All_cause",lc_scenario %in% c("Green30","Target30")) %>%
  select(LSOA11CD,lc_scenario,deaths_averted)
p<-read.csv("data/derived/lsoa_population_2021_by_lsoa11cd.csv")
stopifnot(nrow(v)==4835,!anyDuplicated(v$LSOA11CD),!anyDuplicated(p$LSOA11CD),
          nrow(h)==9670,!anyDuplicated(h[c("LSOA11CD","lc_scenario")]))
d<-h %>% pivot_wider(names_from=lc_scenario,values_from=deaths_averted) %>%
  left_join(p %>% select(LSOA11CD,population_2021),by="LSOA11CD") %>% left_join(v,by="LSOA11CD")
stopifnot(all(d$population_2021>0),all(is.finite(d$Green30)),all(is.finite(d$Target30)))
classify<-function(x,b) ifelse(x<=b[1],1L,ifelse(x<=b[2],2L,3L))
g<-1e5*d$Green30/d$population_2021;t<-1e5*d$Target30/d$population_2021
b<-quantile(g,c(1/3,2/3),names=FALSE)
stopifnot(identical(classify(g,b),classify(g*10,b*10)),
          identical(classify(t,b),classify(t*10,b*10)))
ref<-read.csv("figures/figure7/csv/fig7_quantitative_headline_results.csv")
for(m in ref$metric){
  high<-classify(d[[m]],quantile(d[[m]],c(1/3,2/3),names=FALSE))==3L
  z<-ref[ref$metric==m,]
  stopifnot(sum(high)==z$n_high_vulnerability,
    sum(high & classify(g,b)==1)==z$n_blue_Green30,
    sum(high & classify(t,b)==1)==z$n_blue_Target30)
}
# Independently verify every entry/exit count displayed in the revised panel d.
balance<-read.csv("figures/figure7/csv/fig7_low_benefit_entries_exits.csv")
for(m in balance$metric){
  high<-classify(d[[m]],quantile(d[[m]],c(1/3,2/3),names=FALSE))==3L
  low_g<-classify(g,b)==1L; low_t<-classify(t,b)==1L
  z<-balance[balance$metric==m,]
  stopifnot(sum(high & low_g & low_t)==z$remained_low,
    sum(high & low_g & !low_t)==z$left_low,
    sum(high & !low_g & low_t)==z$newly_low,
    sum(high & !low_g & !low_t)==z$neither_low,
    z$low_Green30-z$left_low+z$newly_low==z$low_Target30,
    z$remained_low+z$left_low+z$newly_low+z$neither_low==z$n_high_vulnerability)
}
meta<-read.csv("figures/figure7/csv/fig7_run_metadata.csv")
stopifnot(grepl("per million",meta$benefit_unit),
  isTRUE(all.equal(c(meta$green30_tertile_1,meta$green30_tertile_2),b*10)))
cat("PASS: all entry/exit groups reconcile; saved thresholds are per million.\n")
cat("PASS: 4,835 matched LSOAs; saved counts reproduced; per-million rescaling changes no classes.\n")
cat("Thresholds per million:",b*10,"\n")
# Optional map-led candidate: recompute the complete SVI composition from inputs.
distribution_path<-"figures/figure7/csv/fig7_mapled_SVI_distribution.csv"
if(file.exists(distribution_path)){
  dist<-read.csv(distribution_path)
  high<-classify(d$SVI,quantile(d$SVI,c(1/3,2/3),names=FALSE))==3L
  for(sc in c("Green30","Target30")){
    cls<-classify(if(sc=="Green30")g else t,b)
    z<-dist[dist$scenario==sc,]
    expected<-vapply(seq_len(3),function(k)sum(high & cls==k),integer(1))
    observed<-z$n[match(c("Low","Moderate","High"),z$benefit)]
    stopifnot(all(observed==expected),abs(sum(z$pct)-100)<1e-8,
              all(abs(z$pct-100*z$n/sum(high))<1e-8))
  }
  cat("PASS: map-led SVI bars reproduce all three benefit classes.\n")
}
