# Reproduce the existing Figure 1 borough estimand, not a new climate method.
# Usage: Rscript code/post_processing_layers/audit_figure1_climate_summary.R [repo]
args <- commandArgs(trailingOnly=TRUE)
repo <- if(length(args)) args[1] else "."
base_dir <- file.path(repo,"data/02_preprocess")
baseline <- read.csv(file.path(base_dir,"chelsa02/clipped_london_stats",
  "borough_stats_CHELSA_tasmax_07_2021_V.2.1_London.csv"))
files <- list.files(file.path(base_dir,"GCMs/clipped/stats"),
  "^borough_stats_wc2.*tmax_.*ssp.*csv$",full.names=TRUE)
records <- do.call(rbind,lapply(files,function(path){
  x <- read.csv(path)
  stopifnot(nrow(x)==33,!anyDuplicated(x$borough),setequal(x$borough,baseline$borough))
  meta <- regmatches(basename(path),regexec(
    "tmax_(.*)_(ssp[0-9]+)_([0-9-]+)_London",basename(path)))[[1]]
  data.frame(model=meta[2],ssp=meta[3],period=meta[4],future=mean(x$mean),
    difference=mean(x$mean-baseline$mean[match(x$borough,baseline$borough)]))
}))
stopifnot(nrow(records)==18,all(table(records$model)==6),
          all(table(records$ssp,records$period)==3))
print(table(records$model))
cat("Equal-weight borough baseline mean:",format(mean(baseline$mean),digits=10),"\n")
print(aggregate(cbind(future,difference)~ssp+period,records,mean),digits=10)
# These differences retain the original cross-product/single-year comparator.
# They are NOT a newly harmonized long-term climate anomaly.
