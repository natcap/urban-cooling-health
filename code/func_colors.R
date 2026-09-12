

scenario_colors <- c(
  # Canonical names written by the revised production runner.
  "baseline" = "gray20",
  "allbuilt" = "#d95f0e",
  "treerisk" = "#fec44f",
  "treeopp" = "#d9f0a3",
  "green10" = "#78c679",
  "green20" = "#238443",
  "green30" = "#004529",
  "target10" = "#8c96c6",
  "target20" = "#8856a7",
  "target30" = "#810f7c",
  # Historical names retained only for archived result tables.
  "scenario0" = "gray20", 
  "scenario1" = "#d95f0e", # #78C679
  "scenario2_TR" = "#fec44f", # tree risk
  "scenario3_TO" = "#d9f0a3", # tree opportunity
  "scenario4_10" = "#78c679",
  "scenario4_20" = "#238443",
  "scenario4_30" = "#004529",
  "scenario510"  = "#8c96c6",
  "scenario520"  = "#8856a7",
  "scenario530"  = "#810f7c"
)

# Canonical and historical label mapping. New outputs must use the lower-case
# canonical keys; scenario510/520/530 are accepted only when reading archives.
scenario_labels <- c(
  'baseline'     = 'Baseline',
  'allbuilt'     = 'AllBuilt',
  'treerisk'     = 'TreeRisk',
  'treeopp'      = 'TreeOpp',
  'green10'      = 'Green10',
  'green20'      = 'Green20',
  'green30'      = 'Green30',
  'target10'     = 'Target10',
  'target20'     = 'Target20',
  'target30'     = 'Target30',
  'scenario0'    = 'Baseline',
  'scenario1'    = 'AllBuilt', 
  'scenario2_TR' = 'TreeRisk',
  'scenario3_TO' = 'TreeOpp',
  'scenario4_10' = 'Green10',
  'scenario4_20' = 'Green20',
  'scenario4_30' = 'Green30',
  "scenario510"  = "Target10",
  "scenario520"  = "Target20",
  "scenario530"  = "Target30"
)

# Display-name palette used after labels have been applied.
scenario_colors_new <- c(
  'Baseline' = 'gray20',
  'AllBuilt' = '#d95f0e',
  'TreeRisk' = '#fec44f',
  'TreeOpp' = '#d9f0a3',
  'Green10' = '#78c679',
  'Green20' = '#238443',
  'Green30' = '#004529',
  'Target10' = '#8c96c6',
  'Target20' = '#8856a7',
  'Target30' = '#810f7c'
)
