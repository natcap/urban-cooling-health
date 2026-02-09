## This repository contains scripts pertaining to the Wellcome Trust Project (Extreme heat, green space and mental health).


## Directory Structure

```
├── data/
│   ├── tree_list_GiGL_Pre2023_2050.csv            # tree specices will be at risk by 2050
│   ├── GiGL_GLATrees_Pre2023_risk_2050.shp        # at-risk tree shapefile data
│   └── 
│
├── code/
|    |
|    |__ gcm-data-clip-stats-viz.Rmd              # figure 1a
|
│    ├── *lc_scenarios*
|    |    |   # for the manuscript
|    |    |__ scenario_1_pavement_and_2_opportunity_trees.ipynb  # scenario 1 and 2
|    |    |__ london-climate-scenario.Rmd            # scenario 3 - filter global tree-climate-scenario pairs for London
|    |    |__ london-tree-climate-risk.Rmd           # scenario 3 - match London tree data to the scenario pairs
|    |    |__ london-tree-climate-risk-GiGL.ipynb    # scenario 3 - link at-risk list to London tree shapefile for map
|    |    |__                                        # scenario 4 - InVEST Scenario Generator (Proximity Based) model rather than script 
|    |    |
|    |    |   # for our London collaborators
|    |    |__ jlg_0_tree_scenarios_to_shp.Rmd            # Use J&L Gibbons data for Camden to generate tree planting sceanrio rasters
|    |    |__ jlg_1_lc_clip_to_aoi.py
|    |    |__ jlg_2_scenario_engine.py
|    |    |__ jlg_3_plot_scenarios.py
|    |
│    ├── *health_assessment*                      # Health outcome estimates based on lc and climate scenarios
|    |    |__ health-model-01-prep-input-ONS-mortality-data.Rmd    # baseline data
|    |    |__ health-modeling.py                                   # main code for modeling  
|    |    |__ health-modeling_*.bat                                # batch run for each scenario
|    |    |__ health-modeling-output-plot.ipynb                    # (not updated)
|    |    |__ health-modeling-output-plot-city.Rmd                 # overall stats at city level 
|    |    |__ health-modeling-zonal-stats.ipynb                    # zonal stats at borough or LSOA level 
|    |    |__ health-modeling-zonal-stats-viz-borough.Rmd          # viz zonal stats: bar plots + maps
|
|
├── 
│   ├── 
│   ├── 
│   ├── 
│   ├── func_*.py                                  # Various data processing functions in Python
│   └── func_*.R                                   # Various data processing functions in R
│      

```
