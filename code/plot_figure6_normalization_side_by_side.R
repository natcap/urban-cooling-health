# Render saved fits only; do not refit or replace existing Figure 6 files.
# Run from repository root: Rscript code/plot_figure6_normalization_side_by_side.R
suppressPackageStartupMessages({library(ggplot2); library(patchwork)})
out <- "figures/figure6/allocation_normalization_comparison"
curves <- read.csv(file.path(out,"gam_full_prediction_curves.csv"))
metrics <- c("pIncomeDeprivation","p5under","p75over","pBAME","SVI")
labels <- c("Income deprivation","Age <5","Age 75+","Ethnic-minority share","SVI")
curves$metric <- factor(curves$metric,levels=metrics,labels=labels)
targets <- file.path(out,paste0("normalization_side_by_side_",c("v1_population","v0_style_area"),".png"))
if (!"--redraw" %in% commandArgs(TRUE)) stopifnot(!any(file.exists(targets)))
# --redraw may replace only these two diagnostic PNGs, never their inputs.
protected <- setdiff(list.files("figures",recursive=TRUE,full.names=TRUE),targets)
hashes <- tools::md5sum(protected)

column <- function(d,mode,colour) {
  z <- subset(d,normalization==mode)
  ggplot(z,aes(x,y))+
    geom_ribbon(aes(ymin=y-1.96*se,ymax=y+1.96*se),fill=colour,alpha=.15)+
    geom_line(colour=colour,linewidth=.65)+
    geom_point(data=z[!is.na(z$percentile),],aes(shape=percentile),colour=colour,size=2.3)+
    scale_shape_manual(values=c(P10=16,P90=17),name=NULL)+
    facet_wrap(~metric,ncol=1)+
    labs(title=if(mode=="total") "Without population normalization" else "With population normalization",
      x="Vulnerability (standard deviations)",
      y=if(mode=="total") "Predicted deaths averted per LSOA" else "Predicted deaths averted per million residents")+
    theme_bw(base_size=11)+theme(plot.title=element_text(size=12,face="bold"),
      strip.text=element_text(size=10),legend.position="bottom",plot.margin=margin(5,8,5,8))
}
for(i in seq_along(targets)) {
  allocation <- c("population","area")[i]
  d <- curves[curves$allocation==allocation,]
  stopifnot(nrow(d)>0,all(table(d$metric,d$normalization)>0))
  for(m in levels(d$metric)) {
    a <- subset(d,metric==m & normalization=="total")
    b <- subset(d,metric==m & normalization=="rate")
    stopifnot(identical(a$x,b$x))
  }
  colour <- if(i==1) "#0072B2" else "#777777"
  name <- if(i==1) "V1 population-weighted mortality allocation" else "V0-style uniform-area mortality allocation"
  p <- (column(d,"total",colour) | column(d,"rate",colour))+
    plot_layout(guides="collect")+
    plot_annotation(title=paste("Figure 6 normalization comparison:",name),
      subtitle="Matched rows and x-axes | same health inputs and 4,835 LSOAs | full predictions include the intercept",
      caption=paste("Different outcome units: compare curve shapes, not numerical heights across columns. Circle = P10; triangle = P90.",
        "All LSOAs fitted; display limited to predictor P1–P99. Bands: conditional pointwise 95% CIs, not spatially adjusted.",
        if(i==1) "Population is fixed at 2021; normalization does not change the underlying total deaths averted." else
          "V0-style allocation is reconstructed on the revised grid with conserved borough totals, not the historical V0 output.",sep="\n"),
      theme=theme(plot.title=element_text(size=14,face="bold"),plot.subtitle=element_text(size=10),
        plot.caption=element_text(size=9,hjust=0),plot.margin=margin(12,16,12,16))) &
    theme(legend.position="bottom")
  ggsave(targets[i],p,width=11,height=13,dpi=180,bg="white",device=ragg::agg_png)
}
stopifnot(identical(hashes,tools::md5sum(protected)))
message("Saved two side-by-side PNGs; all pre-existing figure files unchanged.")
