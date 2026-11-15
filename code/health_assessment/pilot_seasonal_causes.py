"""Exploratory national-seasonality transfer; not validated health-burden estimates.

Usage: python pilot_seasonal_causes.py DATA_ROOT FRACTIONS_JSON NEW_OUTPUT_DIR
FRACTIONS_JSON is stdout from inspect_ons_monthly_causes.py. Retains existing
coefficients solely to isolate seasonal scaling; known applicability gaps remain.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
from pilot_seasonal_health import avoided, london_grid, sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('fractions',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();gdal.UseExceptions()
    if a.output.exists():raise FileExistsError(a.output)
    cfgpath=Path(__file__).with_name('health-analysis-revised-equal-area.example.json')
    cfg=json.loads(cfgpath.read_text());fractions=json.loads(a.fractions.read_text())
    runs=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    temps={s:runs/('nodata_target30_2026-11-12' if s=='target30' else 'nodata_correction_2026-11-12')/f'{s}_nodata0/T_air_test.tif' for s in ['baseline','green30','target30']}
    paths={**temps,'population':a.root/cfg['population_2021'],**{c['name']:a.root/c['mortality_raster'] for c in cfg['causes']}}
    rows,zones,counts=london_grid(a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp',list(paths.values()))
    valid=zones>0;z=zones[valid]-1
    base=gdal.Open(str(temps['baseline'])).ReadAsArray()[valid].astype(float)
    deltas={s:gdal.Open(str(v)).ReadAsArray()[valid].astype(float)-base for s,v in temps.items()}
    london=pd.read_csv(runs/'nodata_correction_2026-11-12/ons_seasonal_2021_boroughs.csv').set_index('code').loc[[r[0] for r in rows]]
    output=[]
    for c in cfg['causes']:
        d=gdal.Open(str(paths[c['name']])).ReadAsArray()[valid].astype(float)
        f=fractions['results'][c['name']]['summer_share']
        for method,scale in [('annual',1.0),('england_cause_summer',f),('london_allcause_summer_sensitivity',london.summer_share.values[z])]:
            for s,delta in deltas.items():
                values=[float(avoided(d*scale,delta,c[k]).sum()) for k in ['rr_low_1c','rr_1c','rr_high_1c']]
                output.append(dict(cause=c['name'],scenario=s,method=method,avoided=values[1],rr_low_result=values[0],rr_high_result=values[2],deaths_denominator=float((d*scale).sum()),national_summer_fraction=f))
    manifest={'status':'EXPLORATORY SCALING ONLY; not manuscript-ready',
      'limitations':['England occurrence fractions transferred to London annual registrations','constant July cooling for summer; no threshold or daily lag model','Yang pools metrics and selects maximum-risk lag; disease-pooled endpoint is not verified all-cause','cardiovascular and respiratory coefficients are age-65+ but applied to all-age deaths for diagnostic comparability only','do not sum cause estimates with all-cause'],
      'fraction_source':fractions,'common_cells':int(valid.sum()),
      'sources':{k:{'path':str(v),'sha256':sha256(v)} for k,v in {**paths,'config':cfgpath,'fractions':a.fractions}.items()}}
    a.output.mkdir(parents=True)
    pd.DataFrame(output).to_csv(a.output/'conditional_cause_comparison.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(output).query("scenario != 'baseline' and method == 'england_cause_summer'").to_string(index=False))

if __name__=='__main__':main()
