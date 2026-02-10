

scenario_colors = c(
  "scenario0" = "gray20", 
  "scenario1" = "#d95f0e", # #78C679
  "scenario2_TR" = "#fec44f", # tree risk
  "scenario3_TO" = "#d9f0a3", # tree opportunity
  "scenario4_10" = "#78c679",
  "scenario4_20" = "#238443",
  "scenario4_30" = "#004529")


scenario_abbr <- c(
  'Baseline',
  'AllBuilt', 
  'TreeRisk',
  'TreeOpp',
  'Green10',
  'Green20',
  'Green30'
)


# update the scenario name abbreviation 
scenario_colors_new <- scenario_colors
names(scenario_colors_new) <- scenario_abbr


# Define mapping once at the top of your script
scenario_labels <- c(
  'scenario0'    = 'Baseline',
  'scenario1'    = 'AllBuilt', 
  'scenario2_TR' = 'TreeRisk',
  'scenario3_TO' = 'TreeOpp',
  'scenario4_10' = 'Green10',
  'scenario4_20' = 'Green20',
  'scenario4_30' = 'Green30'
)