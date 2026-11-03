# Reproduce the scoring audit (not a new SVI definition). Run from project root.
library(dplyr)
x <- sf::st_drop_geometry(readRDS("data/temp_svi.RDS")) |>
  filter(scenario=="ssp370",year=="2050")
ref <- sf::st_drop_geometry(sf::st_read("data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg",quiet=TRUE))
stopifnot(!anyDuplicated(x$geoid), !anyDuplicated(ref$id), setequal(x$geoid,as.character(ref$id)))
ref <- ref[match(x$geoid,ref$id),]
stopifnot(all(x$Neighborhood_name==ref$Neighborhood_name))
canonical <- ref$overall_score
four_themes <- rowMeans(ref[c("mean_Social","mean_Household","mean_Racial","mean_Housing")],na.rm=TRUE)
p <- sapply(ref[c("pIncomeDeprivation","p5under","p75over","pNotEnglishProficient",
                 "pSocial_housing","pBAME")], percent_rank)
from_indicators <- rowMeans(cbind(p[,1],rowMeans(p[,2:4],na.rm=TRUE),p[,6],p[,5]),na.rm=TRUE)
stopifnot(max(abs(canonical-from_indicators)) < 1e-12, all(x$SVI==canonical))
metrics <- c("pIncomeDeprivation","p5under","p75over","pBAME")
x <- mutate(x,across(all_of(metrics),percent_rank,.names="{.col}_pct"))
old <- rowMeans(x[endsWith(names(x),"_pct")],na.rm=TRUE)
print(c(max_theme_error=max(abs(canonical-four_themes)),correlation=cor(old,canonical),
        mean_absolute_difference=mean(abs(canonical-old))))
print(table(former=ntile(old,5),canonical=ntile(canonical,5)))
for (score in list(former=old,canonical=canonical))
  print(tapply(x$mean_diff_avg,ntile(score,5),median,na.rm=TRUE))
