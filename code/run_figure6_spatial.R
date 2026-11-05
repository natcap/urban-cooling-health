# Run from repository root: Rscript code/run_figure6_spatial.R [output_directory]
# Matched Green30, 25 C all-cause sample; k=0 reproduces the original GAM.
suppressPackageStartupMessages({library(sf); library(dplyr); library(mgcv)})
out <- commandArgs(TRUE)[1]
if (is.na(out)) out <- "figures/figure6/statistical_reporting"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
root <- "data/derived"
v <- st_read(file.path(root,"fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg"),quiet=TRUE)
h <- read.csv(file.path(root,"health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv")) |>
  filter(lc_scenario=="Green30",temperature_setting_c==25,indicator=="All_cause")
p <- read.csv(file.path(root,"lsoa_population_2021_by_lsoa11cd.csv"))
stopifnot(!anyDuplicated(h$LSOA11CD), !anyDuplicated(p$LSOA11CD),
          !anyDuplicated(v$LSOA11CD), setequal(v$LSOA11CD,h$LSOA11CD))
v <- v |> left_join(select(h,LSOA11CD,deaths_averted),by="LSOA11CD") |>
  left_join(select(p,LSOA11CD,population_2021),by="LSOA11CD") |> arrange(LSOA11CD)
metrics <- c("pIncomeDeprivation","p5under","p75over","pBAME","overall_score")
stopifnot(st_crs(v)$epsg==27700, all(v$population_2021>0))
xy <- st_coordinates(st_point_on_surface(st_geometry(v)))/1000 # BNG kilometres
nb <- st_touches(v); i <- rep(seq_along(nb),lengths(nb)); j <- unlist(nb)
grid <- expand.grid(metric=metrics,k=c(0,50,100,200),stringsAsFactors=FALSE)
fit <- function(idx) {
  metric <- grid$metric[idx]; k <- grid$k[idx]
  message("Fitting ",metric," (spatial k=",k,")")
  d <- data.frame(y=1e5*v$deaths_averted/v$population_2021,
                  x=as.numeric(scale(v[[metric]])),E=xy[,1],N=xy[,2])
  stopifnot(all(is.finite(as.matrix(d))))
  f <- if(k==0) y~s(x,k=10,bs="ts") else
    as.formula(sprintf("y~s(x,k=10,bs='ts')+s(E,N,k=%d)",k))
  m <- gam(f,data=d,family=scat(),method="REML",select=TRUE)
  stopifnot(m$converged)
  # Contrast at the same location cancels the additive spatial term.
  nd <- data.frame(x=quantile(d$x,c(.1,.9)),E=median(d$E),N=median(d$N))
  L <- predict(m,nd,type="lpmatrix"); a <- L[2,]-L[1,]
  est <- sum(a*coef(m)); se <- sqrt(drop(a%*%vcov(m)%*%a))
  r <- residuals(m,type="deviance"); r <- r-mean(r)
  moran <- function(z) length(z)/length(i)*sum(z[i]*z[j])/sum(z^2)
  I <- moran(r); expected <- -1/(length(r)-1)
  set.seed(20260908+match(metric,metrics))
  pv <- (1+sum(replicate(999,abs(moran(sample(r))-expected))>=abs(I-expected)))/1000
  nd <- data.frame(x=sort(unique(c(seq(min(d$x),max(d$x),length.out=200),
                                  unname(quantile(d$x,c(.1,.9)))))),
                   E=median(d$E),N=median(d$N))
  curve <- predict(m,nd,type="terms",se.fit=TRUE)
  # Convert response-scale exports only; spatial diagnostics are unitless.
  list(stats=data.frame(metric,k,n=nrow(d),rate_denominator=1000000L,contrast=10*est,low=10*(est-1.96*se),high=10*(est+1.96*se),
                        moran=I,moran_p=pv,overlap=if(k) concurvity(m)["estimate","s(x)"] else NA_real_),
       curves=data.frame(metric,k,x=nd$x,y=10*curve$fit[,1],se=10*curve$se.fit[,1],rate_denominator=1000000L),
       residuals=data.frame(metric,k,LSOA11CD=v$LSOA11CD,E=d$E,N=d$N,residual=r))
}
results <- parallel::mclapply(seq_len(nrow(grid)),fit,mc.cores=3)
stopifnot(all(vapply(results,is.list,logical(1))))
writeLines(capture.output(sessionInfo()),file.path(out,"figure6_spatial_session.txt"))
inputs <- file.path(root,c("fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg",
                          "health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv",
                          "lsoa_population_2021_by_lsoa11cd.csv"))
write.csv(data.frame(input=inputs,md5=unname(tools::md5sum(inputs))),
          file.path(out,"figure6_spatial_inputs.csv"),row.names=FALSE)
for (key in c("stats","curves","residuals"))
  write.csv(bind_rows(lapply(results,"[[",key)),file.path(out,paste0("figure6_spatial_",key,".csv")),row.names=FALSE)

# Diagnostic companion: centred vulnerability curves and residual patterns.
render_diagnostics <- function(out, curves_only=FALSE) {
  # Report illustrations are separate from narrative docs and numerical exports.
  report_figures <- "figures/figure6/spatial_sensitivity"
  dir.create(report_figures, recursive=TRUE, showWarnings=FALSE)
  curves <- read.csv(file.path(out,"figure6_spatial_curves.csv"))
  resid <- read.csv(file.path(out,"figure6_spatial_residuals.csv"))
  refs <- read.csv(file.path(out,"figure6_gam_p10_p90_contrasts.csv"))
  refs <- subset(refs, Analysis=="Primary: all valid LSOAs")
  refs$Metric[refs$Metric=="SVI"] <- "overall_score"
  stats <- read.csv(file.path(out,"figure6_spatial_stats.csv"))
  stopifnot("rate_denominator" %in% names(stats), "rate_denominator" %in% names(curves),
            all(stats$rate_denominator==1e6), all(curves$rate_denominator==1e6))
  labels <- c("No spatial adjustment","Spatial: k = 50","Spatial: k = 100","Spatial: k = 200")
  page <- 0L
  save_panel <- function(plot) {
    page <<- page + 1L
    ggplot2::ggsave(file.path(report_figures, sprintf("figure6_diagnostic-%02d.png",page)),
                    plot, width=9,height=6,dpi=108,bg="white")
  }
  for (m in unique(curves$metric)) {
    ref <- subset(refs,Metric==m)
    stopifnot(nrow(ref)==1)
    px <- c(ref$Predictor_P10,ref$Predictor_P90)
    dots <- do.call(rbind,lapply(split(subset(curves,metric==m),~k),function(z) {
      # Historical exports sampled the curve on a grid. New exports include these x's.
      y <- splinefun(z$x,z$y)(px)
      check <- stats$contrast[stats$metric==m & stats$k==z$k[1]]
      stopifnot(length(check)==1,abs(diff(y)-check)<0.01)
      data.frame(x=px,y=y,k=z$k[1],percentile=c("P10","P90"))
    }))
    save_panel(ggplot2::ggplot(subset(curves,metric==m),
      ggplot2::aes(x,y,colour=factor(k))) + ggplot2::geom_line() +
      ggplot2::geom_vline(xintercept=px,linetype="dotted",colour="grey50") +
      ggplot2::geom_point(data=dots,ggplot2::aes(shape=percentile),size=3) +
      ggplot2::scale_shape_manual(values=c(P10=16,P90=17),name="Vulnerability") +
      ggplot2::scale_colour_discrete(labels=labels) +
      ggplot2::labs(title=m,x="Vulnerability (SD)",y="Relative deaths averted per million residents",
                    colour="Model") + ggplot2::theme_bw())
    if (curves_only) { page <- page + 1L; next }
    save_panel(ggplot2::ggplot(subset(resid,metric==m),
      ggplot2::aes(E,N,colour=residual)) + ggplot2::geom_point(size=.35) +
      ggplot2::scale_colour_gradient2() + ggplot2::coord_equal() +
      ggplot2::facet_wrap(~k,labeller=ggplot2::as_labeller(setNames(labels,c("0","50","100","200")))) + ggplot2::labs(title=m,x="BNG east (km)",y="BNG north (km)") +
      ggplot2::theme_bw())
  }
}
render_diagnostics(out)
