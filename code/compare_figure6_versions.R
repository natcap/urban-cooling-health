# Read-only inputs; separate diagnostic outputs, never production figures.
# Run from repository root: Rscript code/compare_figure6_versions.R
suppressPackageStartupMessages({library(sf); library(dplyr); library(mgcv); library(ggplot2)})
args <- commandArgs(TRUE)
out <- if(length(args)) args[1] else "figures/figure6/version_comparison"
stopifnot(!dir.exists(out)) # Avoid silently replacing a previous comparison.
protected <- list.files("figures", recursive=TRUE, full.names=TRUE)
before <- tools::md5sum(protected)
root <- Sys.getenv("URBAN_COOLING_DATA_ROOT", unset=paste0(
  "/Users/yingjiel/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/",
  "Shared drives/Wellcome Trust Project Data"))
oldfile <- file.path(root,"2_postprocess_intermediate/UCM_official_runs/health_output_stats/health_london_45_zonal_stats_long_LSOA_2050_2050.csv")
inputs <- c(oldfile,"figures/equity_map_biscale/health_sf.rds",
  "data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg",
  "data/derived/health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv",
  "data/derived/lsoa_population_2021_by_lsoa11cd.csv")
v <- st_drop_geometry(st_read(inputs[3],quiet=TRUE)) |> arrange(LSOA11CD)
v$SVI <- v$overall_score
old <- read.csv(oldfile) |> filter(lc_scenario=="health_output_s0_s43_2050_2050",raster=="Excess_all_cause.tif")
cache <- st_drop_geometry(readRDS(inputs[2])) |> filter(lc_scenario=="Green30",indicator=="All_cause")
new <- read.csv(inputs[4]) |> filter(lc_scenario=="Green30",temperature_setting_c==25,indicator=="All_cause")
pop <- read.csv(inputs[5])
metrics <- c("pIncomeDeprivation","p5under","p75over","pBAME","SVI")
stopifnot(nrow(v)==4835,!anyDuplicated(v$id),!anyDuplicated(old$id),!anyDuplicated(cache$id),
  !anyDuplicated(new$LSOA11CD),!anyDuplicated(pop$LSOA11CD),setequal(old$id,v$id),setequal(cache$id,v$id),
  setequal(v$LSOA11CD,new$LSOA11CD))
old <- old[match(v$id,old$id),]; cache <- cache[match(v$id,cache$id),]
# Verify the historical ID mapping against every predictor and polygon,
# rather than assuming matching row numbers mean matching neighbourhoods.
old$SVI <- old$overall_score
stopifnot(max(abs(-old$sum-cache$cases))<1e-10)
for (m in metrics) stopifnot(max(abs(v[[m]]-old[[m]]))<1e-10,
                            max(abs(v[[m]]-cache[[m]]))<1e-10)
oldgeom <- st_as_sfc(old$geometry,crs=27700) |> st_zm()
newgeom <- st_geometry(st_read(inputs[3],quiet=TRUE))[match(v$id,st_read(inputs[3],quiet=TRUE)$id)]
stopifnot(all(vapply(seq_len(nrow(v)),function(i) length(st_equals(oldgeom[i],newgeom[i])[[1]])==1,logical(1))))
v$old <- -old$sum
v$revised <- new$deaths_averted[match(v$LSOA11CD,new$LSOA11CD)]
v$population <- pop$population_2021[match(v$LSOA11CD,pop$LSOA11CD)]
stopifnot(all(is.finite(as.matrix(v[,c(metrics,"old","revised","population")]))),all(v$population>0))
dir.create(out,recursive=TRUE)
write.csv(data.frame(input=inputs,md5=unname(tools::md5sum(inputs))),file.path(out,"input_checksums.csv"),row.names=FALSE)
design <- expand.grid(metric=metrics,input=c("old","revised"),normalization=c("total","rate"),sample=c("trimmed","all"),stringsAsFactors=FALSE)
results <- lapply(seq_len(nrow(design)),function(i){
  z <- design[i,]; message(i,"/",nrow(design),": ",paste(z,collapse=" / "))
  x <- as.numeric(scale(v[[z$metric]]))
  y <- v[[z$input]]; mult <- 1
  # Fit rates in the production per-100,000 units, then export per million.
  if(z$normalization=="rate") {y <- 1e5*y/v$population; mult <- 10}
  limits <- quantile(x,c(.01,.99)); keep <- if(z$sample=="all") rep(TRUE,length(x)) else x>=limits[1]&x<=limits[2]
  model <- gam(y~s(x,k=10,bs="ts"),data=data.frame(x=x[keep],y=y[keep]),family=scat(),method="REML",select=TRUE)
  stopifnot(model$converged)
  # Identical full-sample P10/P90 in all eight fits; no moving endpoints.
  q <- unname(quantile(x,c(.1,.5,.9))); L <- predict(model,data.frame(x=q),type="lpmatrix")
  a <- L[3,]-L[1,]; contrast <- sum(a*coef(model))*mult
  se <- sqrt(drop(a%*%vcov(model)%*%a))*mult
  # Shared central support avoids extrapolating trimmed fits into omitted tails.
  grid <- seq(min(x[ x>=limits[1] & x<=limits[2] ]),max(x[ x>=limits[1] & x<=limits[2] ]),length.out=300)
  pred <- as.numeric(predict(model,data.frame(x=grid),type="response"))
  ref <- as.numeric(predict(model,data.frame(x=q[2]),type="response"))
  curve <- cbind(z[rep(1,length(grid)),],data.frame(x=grid,y=(pred-ref)*mult,standardized=(pred-ref)/sd(y)))
  stats <- cbind(z,data.frame(n=sum(keep),contrast,se,low=contrast-1.96*se,high=contrast+1.96*se,
    contrast_outcome_sd=contrast/(sd(y)*mult),outcome_sd=sd(y)*mult,edf=summary(model)$s.table[1,"edf"],p10=q[1],p90=q[3]))
  list(stats=stats,curve=curve)
})
stats <- bind_rows(lapply(results,`[[`,"stats")); curves <- bind_rows(lapply(results,`[[`,"curve"))
saved <- read.csv("figures/figure6/statistical_reporting/figure6_gam_p10_p90_contrasts.csv") |> filter(Analysis=="Primary: all valid LSOAs")
current <- stats |> filter(input=="revised",normalization=="rate",sample=="all")
error <- max(abs(current$contrast-saved$Difference_P90_minus_P10[match(current$metric,saved$Metric)]))
stopifnot(error<1e-6)
write.csv(stats,file.path(out,"all_eight_combinations.csv"),row.names=FALSE)
write.csv(curves,file.path(out,"comparison_curves.csv"),row.names=FALSE)
steps <- list(c("old","total","trimmed"),c("old","rate","trimmed"),c("revised","rate","trimmed"),c("revised","rate","all"))
pick <- function(d,z) filter(d,input==z[1],normalization==z[2],sample==z[3])
chain <- bind_rows(lapply(seq_along(steps),function(i) mutate(pick(stats,steps[[i]]),step=i)))
write.csv(chain,file.path(out,"stepwise_contrasts.csv"),row.names=FALSE)
labels <- c(pIncomeDeprivation="Income deprivation",p5under="Age <5",p75over="Age 75+",pBAME="Ethnic-minority share",SVI="SVI")
changes <- list()
for(i in 1:3){
  pair <- bind_rows(mutate(pick(curves,steps[[i]]),version="Before"),mutate(pick(curves,steps[[i+1]]),version="After"))
  pair$display <- if(i==1) pair$standardized else pair$y
  pair$metric <- factor(pair$metric,levels=metrics,labels=labels)
  title <- c("1. Population normalization (old inputs, trimmed)","2. Revised health inputs (rates, trimmed)","3. Retaining all LSOAs (revised inputs, rates)")[i]
  p <- ggplot(pair,aes(x,display,colour=version))+geom_hline(yintercept=0,colour="grey80")+geom_line(linewidth=.7)+
    facet_wrap(~metric,ncol=3,scales="free_y")+scale_colour_manual(values=c(Before="#666666",After="#0072B2"))+
    labs(title=title,subtitle="Same central 98% of predictor range; curves referenced to the median vulnerability",
      x="Vulnerability (standard deviations)",y=if(i==1) "Difference from P50 (outcome SD units)" else "Difference from P50 (deaths averted per million)",
      caption="Diagnostic point estimates, not spatially adjusted. Normalization panel rescales each outcome by its full-sample SD.",colour=NULL)+
    theme_bw(base_size=11)+theme(legend.position="bottom")
  ggsave(file.path(out,paste0("step",i,"_curves.png")),p,width=11,height=7,dpi=160)
  for(m in metrics){
    a <- pick(curves,steps[[i]]) |> filter(metric==m); b <- pick(curves,steps[[i+1]]) |> filter(metric==m)
    changes[[length(changes)+1]] <- data.frame(step=i,metric=m,curve_correlation=cor(a$y,b$y),
      rms_difference_outcome_sd=sqrt(mean((b$standardized-a$standardized)^2)),
      rms_difference_per_million=if(i==1) NA_real_ else sqrt(mean((b$y-a$y)^2)))
  }
}
write.csv(bind_rows(changes),file.path(out,"stepwise_curve_changes.csv"),row.names=FALSE)
writeLines(c("Controlled comparison: Green30, all-cause mortality; 4,835 matched LSOAs.",
  "Historical CSV agrees with cached historical cases, all five predictors, and current polygons.",
  "Historical code reference: git commit 95383f8. Exact image-producing runtime is not archived.",
  "All eight combinations retain fixed vulnerability scores, standardization, population2021 and GAM settings.",
  "Trimming removes observations outside each predictor's P1-P99; fixed full-sample P10/P90 contrasts avoid moving endpoints.",
  "Curve comparisons use shared central support and subtract the fitted value at P50, eliminating arbitrary centering differences.",
  "Normalization changes the estimand, not merely units: total deaths/LSOA versus deaths per resident.",
  "Step 1 uses each outcome's full-sample SD only for dimensionless visual comparison; this is not a mortality rate.",
  "No percentages of total change are attributed: factors interact and sequential attribution depends on order.",
  "Revised health inputs bundle upstream changes; this analysis does not separate UCM, mortality allocation or scenario effects.",
  "Conditional model intervals omit spatial dependence and upstream uncertainty; no inferential claims from between-fit differences.",
  paste("Maximum discrepancy from current saved per-million contrasts:",format(error,digits=12)),
  "Current figures were not regenerated. All pre-existing figure files checked unchanged."),file.path(out,"README.txt"))
writeLines(capture.output(sessionInfo()),file.path(out,"session_info.txt"))
after <- tools::md5sum(protected)
stopifnot(identical(before,after))
write.csv(data.frame(file=protected,before=unname(before),after=unname(after)),file.path(out,"existing_figures_unchanged.csv"),row.names=FALSE)
print(chain[,c("metric","step","normalization","n","contrast","contrast_outcome_sd")],row.names=FALSE)
