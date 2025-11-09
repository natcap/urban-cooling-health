library(sf)
library(dplyr)

base <- "Z:\\users\\epavia\\WellcomeTrustProjectData\\2_postprocess_intermediate\\UCM_official_runs"

aggregated <- read.csv(paste0(base, "\\cleaned-data\\aggregated_building_data\\buildings_by_borough_25deg_5uhi.csv"))

uhi_shapefile <- read_sf(paste0(base, "\\current_lulc\\work_and_energy_runs\\uhi_results_london_scenario_25.0deg_5.0uhi_45.0hum_energy_productivity.shp")) %>%
  select(c(1, 9:14)) %>%
  st_drop_geometry()


# group and summarise building data
by_borough <- aggregated %>% 
  group_by(NAME) %>%
  summarise(energy_sav_sum = sum(energy_sav),
            mean_t_air = mean(mean_t_air),
            num_of_buildings = n())

join <- full_join(uhi_shapefile, by_borough, by = "NAME")

write.csv(join, paste0(base, "\\cleaned-data\\aggregated_building_data\\aggregated_building_shp_25deg_5uhi.csv"))
