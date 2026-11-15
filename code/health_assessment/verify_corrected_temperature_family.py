"""Verify completed corrected temperature family against saved offset footprint.

Usage: InVEST Python verify_corrected_temperature_family.py DATA_ROOT RUN_FOLDER
Adds one exclusive verification JSON; no temperature or health outputs modified.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
from pilot_spatial_reconstructed_health import london_grid,sha256
from signed_health_response import signed_sum

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('run',type=Path);a=ap.parse_args()
    target=a.run/'health_readiness_verification.json'
    if target.exists():raise FileExistsError(target)
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    repo=Path(__file__).resolve().parents[2];cfg=json.loads((repo/'code/health_assessment/health-analysis-revised-equal-area.example.json').read_text())
    m=json.loads((a.run/'manifest.json').read_text());paths={k:Path(v) for k,v in m['scenario_paths'].items()}
    assert len(paths)==10
    for k,p in paths.items():assert sha256(p)==m['outputs'][k]
    for p,h in m['preserved_sources'].items():assert sha256(p)==h
    pop=a.root/cfg['population_2021'];boundary=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    deaths={c['name']:a.root/c['mortality_raster'] for c in cfg['causes']}
    _,z,_=london_grid(boundary,list(paths.values())+[pop]+list(deaths.values()))
    _,reference,_=london_grid(boundary,[paths['baseline'],pop]);assert np.array_equal(z,reference)
    valid=z>0;count=int(valid.sum());assert count==14258588
    b=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float)
    offsets_folder=a.run.parent/'nine_scenario_health_readiness_2026-11-12_v2'
    om=json.loads((offsets_folder/'manifest.json').read_text())
    assert abs(b.mean()-om['corrected_baseline_area_mean_c'])<1e-10
    assert count==om['common_cells']
    d=pd.read_csv(offsets_folder/'shared_daily_offsets.csv');o=np.r_[-50.,50.,d.lag01_offset_c.to_numpy()[::50]-24.7]
    rng=np.random.default_rng(20261112);ix=rng.choice(len(b),1500,replace=False)
    rr=pd.read_csv(repo/'docs/manuscript/manuscript_health_response_candidates.csv').set_index('endpoint')
    tests=[]
    for name,p in paths.items():
        s=gdal.Open(str(p)).ReadAsArray()[valid].astype(float);c=b-s
        for cause,mp in deaths.items():
            w=gdal.Open(str(mp)).ReadAsArray()[valid][ix].astype(float)
            for col in ['rr_per_1c','ci95_lower','ci95_upper']:
                r=float(rr.loc[cause,col]);got=signed_sum(b[ix],c[ix],w,r,o)
                expected=np.array([np.sum(w*-np.expm1(np.log(r)*(np.maximum(s[ix]+x,0)-np.maximum(b[ix]+x,0)))) for x in o])
                err=float(abs(got-expected).max());assert err<1e-8
                if name in ['allbuilt','treerisk']:assert got.min()<0 and got.max()<1e-8
                elif name=='baseline':assert (got==0).all()
                else:assert got.min()>-1e-8
                tests.append(dict(scenario=name,cause=cause,coefficient=col,max_abs_error=err))
    out=dict(status='corrected temperature family and shared offsets compatible; no health rerun',common_cells=count,offset_file=str(offsets_folder/'shared_daily_offsets.csv'),offset_sha256=sha256(offsets_folder/'shared_daily_offsets.csv'),baseline_mean_c=float(b.mean()),checks=len(tests),max_direct_error=max(t['max_abs_error'] for t in tests),tests=tests,script_sha256=sha256(Path(__file__)))
    with target.open('x') as f:json.dump(out,f,indent=2)
    print({k:v for k,v in out.items() if k!='tests'})

if __name__=='__main__':main()
