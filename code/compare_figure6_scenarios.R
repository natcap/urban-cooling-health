# Controlled six-scenario companion to Figure 6; no main-figure overwrites.
# Run from repository root: Rscript code/compare_figure6_scenarios.R
suppressPackageStartupMessages({library(sf);library(dplyr);library(mgcv);library(ggplot2)})
out <- "figures/figure6/scenario_comparison"
target_overlay_only <- "--target-overlay-only" %in% commandArgs(TRUE)
main_overlay_only <- "--main-overlay-only" %in% commandArgs(TRUE)
main_overlay_file <- "figures/figure6/manuscript/equity_health_All_cause_gam_Green30_Target30_per_million_population_full_prediction.png"
overlay_only <- "--overlay-only" %in% commandArgs(TRUE) || target_overlay_only || main_overlay_only
redraw <- "--redraw" %in% commandArgs(TRUE) || overlay_only
stopifnot(dir.exists(out)==redraw)
protected <- list.files("figures",recursive=TRUE,full.names=TRUE)
if(main_overlay_only) protected <- setdiff(protected,main_overlay_file)
if(redraw && !main_overlay_only) protected <- setdiff(protected,file.path(out,c(
 if(!overlay_only) c("figure6_supplement_green.png","figure6_supplement_target.png"),
 if(!target_overlay_only) "figure6_supplement_green_overlay.png",
 "figure6_supplement_target_overlay.png")))
before <- tools::md5sum(protected)
metrics <- c("pIncomeDeprivation","p5under","p75over","pBAME","SVI")
labels <- c("Income deprivation","Age <5 share","Age ≥75 share","Ethnic-minority share","SVI")
scenarios <- c("Green10","Green20","Green30","Target10","Target20","Target30")
if(!redraw) {
inputs <- c("data/derived/health_lsoa_revised_equal_area_population_weighted_2021.csv",
 "data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg",
 "data/derived/lsoa_population_2021_by_lsoa11cd.csv")
h <- read.csv(inputs[1]) |> filter(indicator=="All_cause",temperature_setting_c==25)
v <- st_drop_geometry(st_read(inputs[2],quiet=TRUE)) |> mutate(SVI=overall_score)
p <- read.csv(inputs[3]) |> select(LSOA11CD,population_2021)
stopifnot(!anyDuplicated(v$LSOA11CD),!anyDuplicated(p$LSOA11CD),
 !anyDuplicated(h[c("LSOA11CD","lc_scenario")]),setequal(unique(h$lc_scenario),scenarios))
d <- v |> select(LSOA11CD,all_of(metrics)) |> left_join(p,by="LSOA11CD") |> arrange(LSOA11CD)
stopifnot(nrow(d)==4835,all(is.finite(as.matrix(d[,c(metrics,"population_2021")]))),all(d$population_2021>0))
for(m in metrics) d[[m]] <- as.numeric(scale(d[[m]]))
# Verify the six-scenario source exactly reproduces the input of main Figure 6.
main <- read.csv("data/derived/health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv") |>
 filter(indicator=="All_cause",temperature_setting_c==25,lc_scenario=="Green30") |> arrange(LSOA11CD)
g <- h |> filter(lc_scenario=="Green30") |> arrange(LSOA11CD)
stopifnot(identical(g$LSOA11CD,main$LSOA11CD),max(abs(g$deaths_averted-main$deaths_averted))<1e-10)
res <- list()
for(m in metrics) for(scenario in scenarios) {
 message(m," / ",scenario)
 z <- h |> filter(lc_scenario==.env$scenario) |> arrange(LSOA11CD)
 stopifnot(identical(z$LSOA11CD,d$LSOA11CD),all(is.finite(z$deaths_averted)))
 dat <- data.frame(x=d[[m]],y=1e5*z$deaths_averted/d$population_2021)
 fit <- gam(y~s(x,k=10,bs="ts"),data=dat,family=scat(),method="REML",select=TRUE)
 stopifnot(fit$converged)
 q <- unname(quantile(dat$x,c(.1,.9)))
 xx <- seq(quantile(dat$x,.01),quantile(dat$x,.99),length.out=200)
 # Full response includes intercept and covariance; all observations fitted.
 pred <- predict(fit,data.frame(x=xx),type="response",se.fit=TRUE)
 L <- predict(fit,data.frame(x=q),type="lpmatrix"); a <- L[2,]-L[1,]
 contrast <- 10*sum(a*coef(fit)); se <- 10*sqrt(drop(a%*%vcov(fit)%*%a))
 res[[length(res)+1]] <- list(
  curve=data.frame(metric=m,scenario,x=xx,y=10*as.numeric(pred$fit),se=10*as.numeric(pred$se.fit)),
  stat=data.frame(metric=m,scenario,n=nrow(dat),contrast,se,low=contrast-1.96*se,high=contrast+1.96*se,
   edf=summary(fit)$s.table[1,"edf"],min_prediction=10*min(pred$fit)))
}
curves <- bind_rows(lapply(res,`[[`,"curve")); stats <- bind_rows(lapply(res,`[[`,"stat"))
ref <- read.csv("figures/figure6/statistical_reporting/population_allocation/figure6_gam_p10_p90_contrasts.csv") |>
 filter(Analysis=="Primary: all valid LSOAs")
check <- stats |> filter(scenario=="Green30")
stopifnot(max(abs(check$contrast-ref$Difference_P90_minus_P10[match(check$metric,ref$Metric)]))<1e-4)
# Descriptive shape correlation removes level/scale differences, not uncertainty.
# It is not a test of equivalence; inspect complete curves alongside contrasts.
shape <- bind_rows(lapply(metrics,function(m) bind_rows(lapply(scenarios,function(s){
 a<-filter(curves,metric==m,scenario==s);b<-filter(curves,metric==m,scenario=="Green30")
 data.frame(metric=m,scenario=s,curve_correlation_with_Green30=cor(a$y,b$y),
   fraction_increasing=mean(diff(a$y)>0),peak_z=a$x[which.max(a$y)],trough_z=a$x[which.min(a$y)])
}))))
dir.create(out,recursive=TRUE)
write.csv(stats,file.path(out,"contrasts.csv"),row.names=FALSE)
write.csv(curves,file.path(out,"curves.csv"),row.names=FALSE)
write.csv(shape,file.path(out,"shape_checks.csv"),row.names=FALSE)
write.csv(data.frame(path=inputs,md5=unname(tools::md5sum(inputs))),file.path(out,"input_checksums.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"session_info.txt"))
} else {
 curves <- read.csv(file.path(out,"curves.csv"))
 stats <- read.csv(file.path(out,"contrasts.csv"))
}
for(family in if(overlay_only) character(0) else c("Green","Target")) {
 cs <- paste0(family,c(10,20,30)); plotdata <- filter(curves,scenario %in% cs)
 panels <- unlist(lapply(seq_along(metrics),function(i) paste(labels[i],cs,sep=" | ")))
 plotdata$panel <- factor(paste(labels[match(plotdata$metric,metrics)],plotdata$scenario,sep=" | "),levels=panels)
 g <- ggplot(plotdata,aes(x,y))+
  geom_ribbon(aes(ymin=y-1.96*se,ymax=y+1.96*se),fill="#2c7bb6",alpha=.2)+
  geom_line(colour="#2c7bb6",linewidth=.65)+
  geom_vline(xintercept=0,linetype="dotted",colour="grey50",linewidth=.25)+
  # Separate annotation strip avoids overlap with near-zero Target10 benefits.
  annotate("segment",x=-.05,xend=-2,y=-8,yend=-8,colour="grey50",
    arrow=grid::arrow(length=grid::unit(.1,"cm")))+
  annotate("segment",x=.05,xend=2,y=-8,yend=-8,colour="grey50",
    arrow=grid::arrow(length=grid::unit(.1,"cm")))+
  annotate("text",x=-1.2,y=-4,label="Less Vulnerable",size=2.1,colour="grey40",fontface="italic")+
  annotate("text",x=1.2,y=-4,label="More Vulnerable",size=2.1,colour="grey40",fontface="italic")+
  facet_wrap(~panel,ncol=3,scales="free_x")+
  scale_y_continuous(breaks=seq(0,ceiling(max(curves$y+1.96*curves$se)/25)*25,25))+
  coord_cartesian(ylim=c(-11,max(curves$y+1.96*curves$se)+3))+
  labs(x="Vulnerability metric (standardized z-score)",y="Predicted deaths averted per million residents",
       title=paste(family,"scenarios: neighborhood vulnerability and health benefits"))+
  theme_light(base_size=10)+theme(strip.text=element_text(size=8,face="bold"),
    strip.background=element_rect(fill="white",colour="grey65"),
    strip.text.x=element_text(colour="black"),
    panel.grid.minor=element_blank(),plot.margin=margin(10,12,10,12))
 if(family=="Green") {
  # Pool all three scenarios within each metric: free scales between rows,
  # identical limits within a row. Include the entire confidence ribbon.
  bounds <- plotdata %>% group_by(metric) %>%
    summarise(top=max(y+1.96*se)*1.04,.groups="drop")
  ann <- distinct(plotdata,panel,metric) %>% left_join(bounds,by="metric")
  g <- ggplot(plotdata,aes(x,y))+
   geom_ribbon(aes(ymin=y-1.96*se,ymax=y+1.96*se),fill="#2c7bb6",alpha=.2)+
   geom_line(colour="#2c7bb6",linewidth=.65)+
   geom_vline(xintercept=0,linetype="dotted",colour="grey50",linewidth=.25)+
   geom_blank(data=ann,aes(x=0,y=top),inherit.aes=FALSE)+
   geom_blank(data=ann,aes(x=0,y=-.14*top),inherit.aes=FALSE)+
   geom_segment(data=ann,aes(x=-.05,xend=-2,y=-.10*top,yend=-.10*top),
     inherit.aes=FALSE,colour="grey50",arrow=grid::arrow(length=grid::unit(.1,"cm")))+
   geom_segment(data=ann,aes(x=.05,xend=2,y=-.10*top,yend=-.10*top),
     inherit.aes=FALSE,colour="grey50",arrow=grid::arrow(length=grid::unit(.1,"cm")))+
   geom_text(data=ann,aes(x=-1.2,y=-.05*top),label="Less Vulnerable",
     inherit.aes=FALSE,size=2.1,colour="grey40",fontface="italic")+
   geom_text(data=ann,aes(x=1.2,y=-.05*top),label="More Vulnerable",
     inherit.aes=FALSE,size=2.1,colour="grey40",fontface="italic")+
   facet_wrap(~panel,ncol=3,scales="free")+
   scale_y_continuous(breaks=function(limits) {b <- pretty(c(0,limits[2]),n=4); b[b>=0]})+
   labs(x="Vulnerability metric (standardized z-score)",y="Predicted deaths averted per million residents",
     title="Green scenarios: neighborhood vulnerability and health benefits",
     subtitle="Y-axis ranges differ between indicators; shared across scenarios within each row")+
   g$theme
  # Test actual rendered panel ranges, not just the intended limits.
  built <- ggplot_build(g)
  yr <- lapply(built$layout$panel_params,function(p) p$y.range)
  stopifnot(all(vapply(split(seq_along(yr),rep(1:5,each=3)),
    function(i) identical(yr[[i[1]]],yr[[i[2]]]) && identical(yr[[i[1]]],yr[[i[3]]]),logical(1))))
 }
 ggsave(file.path(out,paste0("figure6_supplement_",tolower(family),".png")),g,
   width=10,height=12,dpi=300,device=ragg::agg_png)
}
# Optional compact comparison: reuse saved predictions, never refit the GAMs.
# --overlay-only preserves every existing figure except this optional version.
source("code/func_colors.R")
for(overlay_family in if(main_overlay_only) "Main" else if(target_overlay_only) "Target" else if(overlay_only) "Green" else c("Green","Target")) {
overlay_scenarios <- if(overlay_family=="Main") c("Green30","Target30") else paste0(overlay_family,c(10,20,30))
overlay <- filter(curves,scenario %in% overlay_scenarios)
main_labels <- c("Income deprivation","Age <5","Age ≥75",
                 "Ethnic-minority population share","Social Vulnerability Index")
overlay$indicator <- factor(main_labels[match(overlay$metric,metrics)],levels=main_labels[c(2,3,4,1,5)])
overlay$scenario <- factor(overlay$scenario,levels=overlay_scenarios)
ann <- overlay %>% group_by(indicator) %>% summarise(top=max(y+1.96*se)*1.04,.groups="drop")
# Target10 approaches zero: put labels below the arrows and move density down
# to keep all model curves unobscured. Main Figure 6 and Green remain unchanged.
ann$text_fraction <- if(overlay_family %in% c("Target","Main")) -.05 else .07
ann$floor_fraction <- if(overlay_family %in% c("Target","Main")) -.252 else -.132
# Use the observed 4,835-neighbourhood predictors, not the prediction grid.
# Verify the source still matches the fitted analysis before drawing densities.
vpath <- "data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg"
manifest <- read.csv(file.path(out,"input_checksums.csv"))
stopifnot(unname(tools::md5sum(vpath))==manifest$md5[manifest$path==vpath])
vd <- st_drop_geometry(st_read(vpath,quiet=TRUE)) %>% mutate(SVI=overall_score)
stopifnot(nrow(vd)==4835,!anyDuplicated(vd$LSOA11CD))
dens <- bind_rows(lapply(seq_along(metrics),function(i) {
 xx <- as.numeric(scale(vd[[metrics[i]]]))
 stopifnot(all(is.finite(xx)))
 de <- density(xx,n=256)
 lim <- quantile(xx,c(.01,.99))
 data.frame(indicator=main_labels[i],x=de$x,height=de$y/max(de$y)) %>%
   filter(x>=lim[1],x<=lim[2])
})) %>% left_join(ann,by="indicator") %>%
 mutate(indicator=factor(indicator,levels=levels(overlay$indicator)),
        ymin=floor_fraction*top,ymax=ymin+.12*top*height)
g_overlay <- ggplot(overlay,aes(x,y,colour=scenario,fill=scenario,linetype=scenario))+
 geom_ribbon(data=dens,aes(x=x,ymin=ymin,ymax=ymax),inherit.aes=FALSE,
   fill="grey65",colour="grey45",linewidth=.15,alpha=.65)+
 geom_ribbon(aes(ymin=y-1.96*se,ymax=y+1.96*se),alpha=.14,colour=NA,linetype=0)+
 geom_line(linewidth=.8)+
 geom_vline(xintercept=0,linetype="dotted",colour="grey50",linewidth=.25)+
 geom_blank(data=ann,aes(x=0,y=top),inherit.aes=FALSE)+
 geom_blank(data=ann,aes(x=0,y=(floor_fraction-.038)*top),inherit.aes=FALSE)+
 annotate("segment",x=-.05,xend=-2,y=0,yend=0,
   colour="grey50",arrow=grid::arrow(length=grid::unit(.1,"cm")))+
 annotate("segment",x=.05,xend=2,y=0,yend=0,
   colour="grey50",arrow=grid::arrow(length=grid::unit(.1,"cm")))+
 geom_text(data=ann,aes(x=-1.5,y=text_fraction*top),label="Less Vulnerable",
   inherit.aes=FALSE,size=2.2,colour="grey40",fontface="italic")+
 geom_text(data=ann,aes(x=1.5,y=text_fraction*top),label="More Vulnerable",
   inherit.aes=FALSE,size=2.2,colour="grey40",fontface="italic")+
 facet_wrap(~indicator,ncol=3,scales="free")+
 scale_x_continuous(breaks=scales::breaks_width(1),expand=expansion(mult=c(.1,.05)))+
 scale_colour_manual(values=scenario_colors_new)+scale_fill_manual(values=scenario_colors_new)+
 scale_linetype_manual(values=setNames(if(overlay_family=="Main") c("solid","dashed") else c("dotted","dashed","solid"),overlay_scenarios))+
 scale_y_continuous(breaks=function(limits) {b <- pretty(c(0,limits[2]),n=4); b[b>=0]})+
 labs(x="Vulnerability metric (standardized z-score)",y="Predicted deaths averted per million residents",
   subtitle=paste0("Associations between neighborhood vulnerability and all-cause deaths averted (",
     if(overlay_family=="Main") "Green30 vs Target30" else paste0(overlay_family,"10/20/30"),")"),
   colour=NULL,fill=NULL,linetype=NULL)+
 theme_light(base_size=9)+theme(legend.position="bottom",legend.key.width=grid::unit(1.2,"cm"),
   strip.text=element_text(size=8,face="bold",colour="black"),
   strip.background=element_rect(fill="white",colour="grey65"),
   panel.grid.minor=element_blank(),panel.grid.major=element_line(colour="grey92",linewidth=.2),
   plot.subtitle=element_text(size=9),plot.margin=margin(8,12,8,12))
stopifnot(nrow(overlay)==1000*length(overlay_scenarios),length(unique(overlay$indicator))==5)
# Both scenarios inhabit each panel and therefore share its exact axis limits.
destination <- if(overlay_family=="Main") main_overlay_file else file.path(out,paste0("figure6_supplement_",tolower(overlay_family),"_overlay.png"))
ggsave(destination,g_overlay,
 width=7.5,height=5.5,dpi=600,device=ragg::agg_png)
}
stopifnot(identical(before,tools::md5sum(protected)))
print(stats[,c("metric","scenario","contrast")],row.names=FALSE)
message(if(redraw) "Only supplementary PNGs redrawn; saved numerical results unchanged." else
 "All existing figure files unchanged; Green30 input and contrast parity passed.")
