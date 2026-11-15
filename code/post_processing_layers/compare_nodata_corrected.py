"""Compare isolated corrected baseline/Green30/Target30 with production.

Writes only a new comparison JSON in the specified diagnostic output folder.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from osgeo import gdal
from valuation_accounting import london_grid, masked_means

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_root',type=Path)
    p.add_argument('baseline_green_diagnostic',type=Path)
    p.add_argument('target_diagnostic',type=Path)
    a=p.parse_args();gdal.UseExceptions()
    prod=a.data_root/'2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/01_temperature_invest3202'
    paths={};checks=[]
    for s in ['baseline','green30','target30']:
        folder=a.target_diagnostic if s=='target30' else a.baseline_green_diagnostic
        for n in [0,255]: paths[f'{s}_{n}']=folder/f'{s}_nodata{n}/T_air_test.tif'
        old=list((prod/s).glob('T_air_*25deg_5uhi*.tif'));assert len(old)==1
        paths[s+'_production']=old[0]
        ds=[gdal.Open(str(paths[f'{s}_{n}'])) for n in [0,255]]
        assert ds[0].GetGeoTransform()==ds[1].GetGeoTransform()
        x,y=[d.ReadAsArray() for d in ds]
        nx,ny=[d.GetRasterBand(1).GetNoDataValue() for d in ds]
        vx=np.isfinite(x)&(x!=nx);vy=np.isfinite(y)&(y!=ny)
        difference=y[vx&vy].astype(float)-x[vx&vy].astype(float)
        checks.append(dict(scenario=s,mask_differences=int((vx!=vy).sum()),max_abs_difference=float(abs(difference).max())))
        assert checks[-1]['mask_differences']==0 and checks[-1]['max_abs_difference']<1e-6
    _,zones,counts=london_grid(a.data_root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp',list(paths.values()))
    means={k:masked_means(v,zones,counts) for k,v in paths.items()}
    out={}
    for version in ['0','255','production']:
        b=means['baseline_'+version];g=means['green30_'+version];t=means['target30_'+version]
        out[version]=dict(baseline_city=b[0],green30_city=g[0],target30_city=t[0],
            green_cooling_city=b[0]-g[0],target_cooling_city=b[0]-t[0],
            green_cooling_borough=float((b[1]-g[1]).mean()),target_cooling_borough=float((b[1]-t[1]).mean()),
            target_advantage_city=g[0]-t[0],target_advantage_borough=float((g[1]-t[1]).mean()),
            boroughs_target_cooler=int((t[1]<g[1]).sum()))
    result=dict(common_london_cells=int(counts.sum()),sentinel_checks=checks,comparison=out,
                source_paths={k:str(v) for k,v in paths.items()})
    path=a.target_diagnostic/'three_scenario_comparison.json'
    with path.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
