# Read-only comparison of old annual production and new conditional summer run.
# Rscript code/health_assessment/compare_seasonal_health_equity.R RUN_FOLDER
suppressPackageStartupMessages({library(sf);library(dplyr);library(mgcv)})
run <- commandArgs(TRUE)[1];stopifnot(!is.na(run),file.exists(file.path(run,"manifest.json")))
out<-file.path(run,"equity_comparison");stopifnot(!dir.exists(out));dir.create(out)
v<-st_read("data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg",quiet=TRUE) |> arrange(LSOA11CD)
xy<-st_coordinates(st_point_on_surface(st_geometry(v)))/1000
v$E<-xy[,1];v$N<-xy[,2];v$SVI<-v$overall_score
v<-st_drop_geometry(v)
p<-read.csv("data/derived/lsoa_population_2021_by_lsoa11cd.csv") |> select(LSOA11CD,population_2021)
old<-read.csv("data/derived/health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv") |>
 filter(temperature_setting_c==25) |> transmute(LSOA11CD,scenario=tolower(lc_scenario),cause,deaths_averted,version="previous_annual")
new<-read.csv(file.path(run,"lsoa_results.csv")) |> filter(reconstruction=="worldclim_aligned",ssp=="ssp370",rr_variant=="central",scenario!="baseline") |>
 select(LSOA11CD,scenario,cause,deaths_averted) |> mutate(version="revised_summer")
d<-bind_rows(old,new) |> left_join(p,by="LSOA11CD") |> left_join(v,by="LSOA11CD")
stopifnot(all(is.finite(d$deaths_averted)),all(d$population_2021>0),!anyDuplicated(d[,c("version","scenario","cause","LSOA11CD")]))
metrics<-c("pIncomeDeprivation","p5under","p75over","pBAME","SVI")
stats<-list();curves<-list();grouped<-list();fig7<-list();transitions<-list();breaks<-list()
for(version in unique(d$version)) {
 for(scenario in unique(d$scenario[d$version==version])) {
  for(metric in metrics) {
   z<-d[d$version==version & d$scenario==scenario & d$cause=="all_cause",];stopifnot(nrow(z)==4835)
   z$x<-as.numeric(scale(z[[metric]]));z$y<-1e5*z$deaths_averted/z$population_2021
   for(k in if(scenario %in% c("green30","target30")) c(0,100) else 0) {
    form<-if(k==0) y~s(x,k=10,bs="ts") else y~s(x,k=10,bs="ts")+s(E,N,k=100)
    fit<-gam(form,data=z,family=scat(),method="REML",select=TRUE);stopifnot(fit$converged)
    q<-unname(quantile(z$x,c(.1,.9)));nd<-data.frame(x=q,E=median(z$E),N=median(z$N))
    L<-predict(fit,nd,type="lpmatrix");aa<-L[2,]-L[1,];est<-sum(aa*coef(fit))*10;se<-sqrt(drop(aa%*%vcov(fit)%*%aa))*10
    stats[[length(stats)+1]]<-data.frame(version,scenario,metric,spatial_k=k,n=nrow(z),contrast_per_million=est,low=est-1.96*se,high=est+1.96*se,mean_lsoa_rate_per_million=mean(z$y)*10)
    if(k==0) {
     grid<-sort(unique(c(seq(quantile(z$x,.01),quantile(z$x,.99),length.out=160),q)))
     pr<-predict(fit,data.frame(x=grid),type="response",se.fit=TRUE)
     curves[[length(curves)+1]]<-data.frame(version,scenario,metric,x=grid,predicted_rate=pr$fit*10,se=pr$se.fit*10,relative_to_mean=pr$fit/mean(z$y))
    }
   }
  }
  message("GAM comparisons: ",version," ",scenario)
 }
 # Figure 7: fixed Green30 benefit thresholds WITHIN each version. Never apply
 # old annual numerical cutoffs to the new seasonal outcome.
 z<-d[d$version==version & d$cause=="all_cause",]
 g<-z[z$scenario=="green30",] |> arrange(LSOA11CD);t<-z[z$scenario=="target30",] |> arrange(LSOA11CD)
 stopifnot(identical(g$LSOA11CD,t$LSOA11CD))
 gr<-1e6*g$deaths_averted/g$population_2021;tr<-1e6*t$deaths_averted/t$population_2021
 br<-quantile(gr,c(1/3,2/3),type=7);breaks[[version]]<-data.frame(version,lower=br[1],upper=br[2])
 classify<-function(x,cut) ifelse(x<=cut[1],1,ifelse(x<=cut[2],2,3))
 for(metric in metrics) {
  vb<-quantile(g[[metric]],c(1/3,2/3),type=7);high<-g[[metric]]>vb[2]
  blueg<-gr<=br[1];bluet<-tr<=br[1]
  fig7[[length(fig7)+1]]<-data.frame(version,metric,n_high=sum(high),low_green=sum(high&blueg),low_target=sum(high&bluet),pct_green=100*mean(blueg[high]),pct_target=100*mean(bluet[high]),remained=sum(high&blueg&bluet),left_low=sum(high&blueg&!bluet),newly_low=sum(high&!blueg&bluet),delta_cases=sum(t$deaths_averted[high]-g$deaths_averted[high]),pct_change_cases=100*(sum(t$deaths_averted[high])/sum(g$deaths_averted[high])-1))
  transitions[[length(transitions)+1]]<-data.frame(version,metric,LSOA11CD=g$LSOA11CD,high_vulnerability=high,green_class=classify(gr,br),target_class=classify(tr,br))
 }
}
# All five outcomes: high-vulnerability totals and population-weighted rates,
# kept separate from unweighted descriptive GAMs and from Figure 7 classes.
for(metric in metrics) {
 threshold<-quantile(v[[metric]],2/3,type=7)
 grouped[[metric]]<-d |> mutate(high_vulnerability=.data[[metric]]>threshold) |> group_by(version,scenario,cause,high_vulnerability) |>
 summarise(n=n(),deaths_averted=sum(deaths_averted),population=sum(population_2021),rate_per_million=1e6*deaths_averted/population,.groups="drop") |> mutate(metric=metric)
}
for(item in list(c("gam_contrasts", "stats"),c("gam_curves","curves"),c("high_vulnerability_groups","grouped"),c("figure7_summary","fig7"),c("figure7_classes","transitions"),c("figure7_thresholds","breaks"))) {
 write.csv(bind_rows(get(item[2])),file.path(out,paste0(item[1],".csv")),row.names=FALSE)
}
# Prior headline classification must reproduce before comparing new results.
ref<-read.csv("figures/figure7/csv/fig7_quantitative_headline_results.csv")
check<-bind_rows(fig7) |> filter(version=="previous_annual") |> left_join(ref,by="metric")
stopifnot(max(abs(check$pct_green-check$pct_blue_Green30))<1e-8,max(abs(check$pct_target-check$pct_blue_Target30))<1e-8)
writeLines(capture.output(sessionInfo()),file.path(out,"r_session.txt"))
message("Equity comparison complete")
