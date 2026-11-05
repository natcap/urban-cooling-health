# Separate Figure 6 sensitivity plots. Run after compare_figure6_allocation.py.
# All curves include the intercept; production figures are never overwritten.
suppressPackageStartupMessages({library(sf);library(dplyr);library(mgcv);library(ggplot2)})
out <- "figures/figure6/allocation_normalization_comparison"
stopifnot(file.exists(file.path(out,"allocation_lsoa.csv")),!file.exists(file.path(out,"gam_contrasts.csv")))
protected <- list.files("figures",recursive=TRUE,full.names=TRUE)
hashes <- tools::md5sum(protected)
v <- st_drop_geometry(st_read("data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg",quiet=TRUE))
d <- read.csv(file.path(out,"allocation_lsoa.csv")) |>
  left_join(v |> select(-id),by="LSOA11CD") |>
  left_join(read.csv("data/derived/lsoa_population_2021_by_lsoa11cd.csv") |> select(LSOA11CD,population_2021),by="LSOA11CD") |>
  arrange(LSOA11CD)
d$SVI <- d$overall_score
metrics <- c("pIncomeDeprivation","p5under","p75over","pBAME","SVI")
labels <- c("Income deprivation","Age <5","Age 75+","Ethnic-minority share","SVI")
stopifnot(nrow(d)==4835,!anyDuplicated(d$LSOA11CD),all(d$population_2021>0),all(is.finite(as.matrix(d[,c(metrics,"area","population")]))))
design <- expand.grid(metric=metrics,allocation=c("area","population"),normalization=c("total","rate"),stringsAsFactors=FALSE)
res <- lapply(seq_len(nrow(design)),function(i){
  z<-design[i,]; message(i,"/",nrow(design))
  x <- as.numeric(scale(d[[z$metric]])); y<-d[[z$allocation]]
  multiplier<-1
  if(z$normalization=="rate") {y<-y/d$population_2021*1e5;multiplier<-10}
  fit<-gam(y~s(x,k=10,bs="ts"),data=data.frame(x,y),family=scat(),method="REML",select=TRUE)
  stopifnot(fit$converged)
  q<-unname(quantile(x,c(.1,.9)))
  grid<-sort(unique(c(seq(quantile(x,.01),quantile(x,.99),length.out=200),q)))
  pred<-predict(fit,data.frame(x=grid),type="response",se.fit=TRUE)
  # Response predictions include intercept uncertainty and covariance.
  curves<-cbind(z[rep(1,length(grid)),],data.frame(x=grid,y=pred$fit*multiplier,se=pred$se.fit*multiplier,
     percentile=ifelse(grid==q[1],"P10",ifelse(grid==q[2],"P90",NA_character_))))
  L<-predict(fit,data.frame(x=q),type="lpmatrix"); a<-L[2,]-L[1,]
  contrast<-sum(a*coef(fit))*multiplier;se<-sqrt(drop(a%*%vcov(fit)%*%a))*multiplier
  stats<-cbind(z,data.frame(n=nrow(d),contrast,se,low=contrast-1.96*se,high=contrast+1.96*se,
    min_displayed_prediction=min(curves$y),negative_displayed_predictions=sum(curves$y<0)))
  list(curves=curves,stats=stats)
})
curves<-bind_rows(lapply(res,`[[`,"curves")); stats<-bind_rows(lapply(res,`[[`,"stats"))
ref<-read.csv("figures/figure6/statistical_reporting/figure6_gam_p10_p90_contrasts.csv") |> filter(Analysis=="Primary: all valid LSOAs")
test<-filter(stats,allocation=="population",normalization=="rate")
stopifnot(max(abs(test$contrast-ref$Difference_P90_minus_P10[match(test$metric,ref$Metric)]))<1e-4)
write.csv(stats,file.path(out,"gam_contrasts.csv"),row.names=FALSE)
write.csv(curves,file.path(out,"gam_full_prediction_curves.csv"),row.names=FALSE)
curves$metric<-factor(curves$metric,levels=metrics,labels=labels)
curves$allocation<-factor(curves$allocation,levels=c("area","population"),labels=c("V0-style area allocation","V1 population allocation"))
draw<-function(dat,unit,title){
  ggplot(dat,aes(x,y,colour=allocation,fill=allocation,linetype=allocation))+
    geom_ribbon(aes(ymin=y-1.96*se,ymax=y+1.96*se),alpha=.12,colour=NA,linetype=0)+
    geom_line(linewidth=.65)+geom_point(data=dat[!is.na(dat$percentile),],aes(shape=percentile),size=2.3)+
    facet_wrap(~metric,ncol=3)+
    scale_colour_manual(values=c("#777777","#0072B2"),drop=FALSE)+
    scale_fill_manual(values=c("#777777","#0072B2"),drop=FALSE)+
    scale_linetype_manual(values=c("dashed","solid"),drop=FALSE)+
    scale_shape_manual(values=c(P10=16,P90=17))+
    labs(title=title,subtitle="Green30 | same revised cooling and borough deaths | all 4,835 LSOAs fitted",
      x="Vulnerability (standard deviations)",y=unit,colour=NULL,fill=NULL,linetype=NULL,shape=NULL,
      caption=paste("Full predictions include the intercept. Display: predictor P1–P99; no observations removed from fitting.",
        "Bands: conditional pointwise 95% CIs; no spatial adjustment or propagated upstream uncertainty.",
        "V0-style allocation is a controlled reconstruction on the V1 grid, not the historical V0 output.",sep="\n"))+
    theme_bw(base_size=11)+theme(legend.position="bottom",plot.caption=element_text(hjust=0,size=9),
      plot.margin=margin(8,15,8,22))
}
for(mode in c("total","rate")){
  dat<-filter(curves,normalization==mode)
  unit<-if(mode=="total") "Predicted deaths averted per LSOA" else "Predicted deaths averted per million residents"
  title<-if(mode=="total") "Mortality allocation comparison — without population normalization" else "Mortality allocation comparison — with population normalization"
  ggsave(file.path(out,paste0("allocation_comparison_",mode,".png")),draw(dat,unit,title),width=11,height=7,dpi=180)
  for(a in levels(curves$allocation)){
    name<-if(grepl("V0",a)) "v0_style_area" else "v1_population"
    single<-draw(filter(dat,allocation==a),unit,paste(a,"—",if(mode=="total") "without normalization" else "with normalization"))+
      guides(colour="none",fill="none",linetype="none")
    ggsave(file.path(out,paste0("figure6_",name,"_",mode,".png")),single,width=11,height=7,dpi=180)
  }
}
writeLines(capture.output(sessionInfo()),file.path(out,"R_session_info.txt"))
stopifnot(identical(hashes,tools::md5sum(protected)))
print(stats,row.names=FALSE)
