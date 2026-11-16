# Read-only final-input audit. Rscript code/lc_scenarios/audit_table1_canopy.R DATA_ROOT OUTPUT_DIR
# No UCM/health reruns. Count all canopy classes, not just overlay code 100.
suppressPackageStartupMessages({library(terra);library(jsonlite)})
a <- commandArgs(TRUE); stopifnot(length(a)==2)
root <- normalizePath(a[1]); out <- a[2]
diag <- file.path(root,"2_postprocess_intermediate/UCM_official_runs/diagnostics")
scenarios <- c("baseline","green10","green20","green30","target10","target20","target30")
args_files <- setNames(vapply(scenarios,function(sc) file.path(diag,
 if(sc %in% c("baseline","green30")) "nodata_correction_2026-11-12" else
 if(sc=="target30") "nodata_target30_2026-11-12" else "remaining_corrected_temperatures_2026-11-12_v1",
 paste0(sc,"_nodata0"),"arguments.json"),character(1)),scenarios)
paths <- vapply(args_files,function(p) fromJSON(p)$lulc_raster_path,character(1))
boundary <- fromJSON(args_files[[1]])$aoi_vector_path
sources <- unique(c(args_files,paths,list.files(dirname(boundary),
 pattern="^London_Borough_aoi\\.",full.names=TRUE)))
before <- tools::md5sum(sources)
b <- rast(paths[[1]]); stopifnot(grepl("27700",crs(b)),all(res(b)==10))
mask <- rasterize(project(vect(boundary),crs(b)),b,field=1,background=0,
                  touches=FALSE,filename=tempfile(fileext=".tif"),overwrite=TRUE)
rows <- list()
for(sc in scenarios) {
 s <- rast(paths[[sc]]);stopifnot(compareGeom(b,s,stopOnError=FALSE))
 counts <- matrix(0,2,6,dimnames=list(c("valid_raster_footprint","london_boundary_pixel_centres"),
 c("valid_pixels","baseline_canopy_pixels","scenario_canopy_pixels","added_pixels","removed_pixels","ineligible_added_pixels")))
 readStart(b);readStart(s);readStart(mask)
 for(start in seq(1,nrow(b),by=128)) {
  n <- min(128,nrow(b)-start+1)
  bv <- readValues(b,start,n);sv <- readValues(s,start,n);mv <- readValues(mask,start,n)
  stopifnot(identical(is.na(bv),is.na(sv)))
  bt <- bv %in% c(1,2,100);st <- sv %in% c(1,2,100)
  for(k in 1:2) {
   valid <- !is.na(bv);if(k==2) valid <- valid & mv==1
   added <- valid & !bt & st
   counts[k,] <- counts[k,]+c(sum(valid),sum(valid & bt),sum(valid & st),sum(added),
    sum(valid & bt & !st),sum(added & !(bv %in% c(4,20,21) & sv==100)))
  }
 }
 readStop(b);readStop(s);readStop(mask)
 for(k in 1:2) {
  r <- as.list(counts[k,]);r$scenario <- sc;r$domain <- rownames(counts)[k]
  r$area_km2 <- r$valid_pixels*100/1e6
  r$baseline_canopy_km2 <- r$baseline_canopy_pixels*100/1e6
  r$canopy_km2 <- r$scenario_canopy_pixels*100/1e6
  r$added_canopy_km2 <- r$added_pixels*100/1e6
  r$relative_increase_pct <- 100*(r$scenario_canopy_pixels/r$baseline_canopy_pixels-1)
  r$final_canopy_share_pct <- 100*r$scenario_canopy_pixels/r$valid_pixels
  r$change_percentage_points <- 100*(r$scenario_canopy_pixels-r$baseline_canopy_pixels)/r$valid_pixels
  rows[[length(rows)+1]] <- as.data.frame(r)
 }
 message("Counted ",sc)
}
result <- do.call(rbind,rows)
stopifnot(all(result$removed_pixels==0),all(result$ineligible_added_pixels==0))
budget_checks <- list()
for(domain in unique(result$domain)) {
 z <- result[result$domain==domain,];stopifnot(length(unique(z$valid_pixels))==1)
 for(tier in c(10,20,30)) {
  g <- z$added_pixels[z$scenario==paste0("green",tier)]
  t <- z$added_pixels[z$scenario==paste0("target",tier)]
  budget_checks[[length(budget_checks)+1]] <- data.frame(domain,tier,green_added_pixels=g,
    target_added_pixels=t,difference_pixels=t-g,difference_pct=100*(t-g)/g)
 }
}
stopifnot(identical(before,tools::md5sum(sources)))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(result,file.path(out,"table1_canopy_audit.csv"),row.names=FALSE)
write.csv(do.call(rbind,budget_checks),file.path(out,"table1_canopy_budget_checks.csv"),row.names=FALSE)
write_json(list(canopy_codes=c(1,2,100),eligible_codes=c(4,20,21),pixel_area_m2=100,
 boundary=boundary,raster_paths=as.list(paths),arguments=as.list(args_files),
 input_md5=as.list(before),checks="Same grid and valid footprint; zero removals or ineligible additions; sources unchanged. Domain-specific pair differences reported separately, not assumed zero."),
 file.path(out,"table1_canopy_provenance.json"),pretty=TRUE,auto_unbox=TRUE)
print(result[,c("scenario","domain","area_km2","baseline_canopy_km2","added_canopy_km2","relative_increase_pct","final_canopy_share_pct")],row.names=FALSE)
