## This script is used to create joined tables from Shapefiles
## of UCM runs. 

library(sf)
library(stringr)
library(purrr)
library(dplyr)
library(tidyr)

# set path to outputs
root <- "G:\\Shared drives\\Wellcome Trust Project Data\\2_postprocess_intermediate\\UCM_preliminary_scenarios\\EP_borough_comparisons"

# list shapefiles we want
shp_files <- list.files(path = root, pattern = "\\.shp$", full.names = TRUE)

# read in the shapefiles
shps <- map(shp_files, st_read) %>%  
  set_names(str_remove_all(basename(shp_files), "uhi_results_|\\_boroughlevel_uhi2_23deg.shp$|\\_25m_borough_level_23deg_2uhi.shp$")) %>%
  map(~.x %>% st_drop_geometry(.)) %>%
  map(~.x %>% subset(., select = -c(ONS_INNER, SUB_2009, SUB_2006, avd_eng_cn, avg_wbgt_v, avg_ltls_v, avg_hvls_v))) %>%
  bind_rows(., .id = "year")

# subset the different values
avg_cc <- shps %>% select(., year, NAME, GSS_CODE, HECTARES, avg_cc) %>% 
  pivot_wider(names_from = year, values_from = avg_cc) %>%
  mutate(diff2021 = ESA2021-UKECH2021,
         diff2015 = UKECH2015-UKECH2000,
         diff2023_resolutions = UKECH2023-UKECH2021)

avg_tmp_v <- shps %>% select(., year, NAME, GSS_CODE, HECTARES, avg_tmp_v) %>% 
  pivot_wider(names_from = year, values_from = avg_tmp_v) %>%
  mutate(diff2021 = ESA2021-UKECH2021,
         diff2015 = UKECH2015-UKECH2000,
         diff2023_resolutions = UKECH2023-UKECH2021)

avg_temp_an <- shps %>% select(., year, NAME, GSS_CODE, HECTARES, avg_tmp_an) %>% 
  pivot_wider(names_from = year, values_from = avg_tmp_an) %>%
  mutate(diff2021 = ESA2021-UKECH2021,
         diff2015 = UKECH2015-UKECH2000,
         diff2023_resolutions = UKECH2023-UKECH2021)

# write the files
write.csv(avg_cc, paste0(root, "/avg_cc_comparisons_23deg_2UHI.csv"))
write.csv(avg_tmp_v, paste0(root, "/avg_tmp_v_comparisons_23deg_2UHI.csv"))
write.csv(avg_temp_an, paste0(root, "/avg_temp_an_comparisons_23deg_2UHI.csv"))



