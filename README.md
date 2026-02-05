## This repository contains scripts pertaining to the Wellcome Trust Project (Extreme heat, green space and mental health).


## Directory Structure

```
├── data/
│   ├── tree_list_GiGL_Pre2023_2050.csv            # tree specices will be at risk by 2050
│   ├── GiGL_GLATrees_Pre2023_risk_2050.shp        # at-risk tree shapefile data
│   └── 
│
├── code/
│   ├── lc_scenarios
|   |    |__ london-climate-scenario.Rmd                # filter global tree-climate-scenario pairs for London
|   |    |__ london-tree-climate-risk.Rmd               # match London tree data to the scenario pairs
|   |    |__ london-tree-climate-risk-GiGL.ipynb        # link at-risk list to London tree shapefile for map
|   |    |__
|   |    |__
|   |    |__
|   |    |__ jlg_0_tree_scenarios_to_shp.Rmd     # Use J&L Gibbons data for Camden to generate tree planting sceanrio rasters
|   |    |__ jlg_1_lc_clip_to_aoi.py
|   |    |__ jlg_2_scenario_engine.py
|   |    |__ jlg_3_plot_scenarios.py
|   |
│   ├── health_assessment                        # Health outcome estimates based on lc and climate scenarios
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
