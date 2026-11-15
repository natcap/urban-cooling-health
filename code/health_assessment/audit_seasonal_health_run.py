"""Post-run completeness, conservation, signed-RR and old/new accounting checks."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from osgeo import gdal
from pilot_spatial_reconstructed_health import sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('run',type=Path);a=ap.parse_args()
    output=a.run/'final_accounting_checks.json';assert not output.exists()
    m=json.loads((a.run/'manifest.json').read_text())
    for p,h in m['sources'].items():assert sha256(p)==h,p
    city=pd.read_csv(a.run/'citywide_results.csv');lsoa=pd.read_csv(a.run/'lsoa_results.csv');borough=pd.read_csv(a.run/'borough_results.csv')
    keys=['scenario','cause','reconstruction','ssp','rr_variant']
    assert len(city)==300 and not city.duplicated(keys).any() and np.isfinite(city.deaths_averted).all()
    assert len(lsoa)==300*4835 and not lsoa.duplicated(keys+['LSOA11CD']).any()
    assert len(borough)==300*33 and not borough.duplicated(keys+['borough_id']).any()
    errors={}
    for name,frame in [('lsoa',lsoa),('borough',borough)]:
        z=frame.groupby(keys).deaths_averted.sum().sort_index()-city.set_index(keys).deaths_averted.sort_index()
        errors[name]=float(abs(z).max());assert errors[name]<1e-7
    primary=city.query("reconstruction=='worldclim_aligned' and ssp=='ssp370'")
    rr=primary.pivot(index=['scenario','cause'],columns='rr_variant',values='deaths_averted')
    for (scenario,cause),row in rr.iterrows():
        if scenario in ['allbuilt','treerisk']:assert row.rr_high<=row.central<=row.rr_low
        else:assert row.rr_low<=row.central<=row.rr_high
    old=[]
    prod=a.root/'2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/02_health_population2021'
    for s in city.scenario.unique():
        if s=='baseline':continue
        p=prod/f'{s}_25c/city_totals_deterministic.csv'
        x=pd.read_csv(p);x['scenario']=s;x['source']=str(p);old.append(x[['scenario','cause','deaths_averted','source']])
    merged=primary.query("rr_variant=='central' and scenario!='baseline'").merge(pd.concat(old),on=['scenario','cause'],suffixes=('_summer','_previous_annual'),validate='one_to_one')
    merged['interpretation']='Different estimands and inputs; not a controlled percentage effect'
    merged.to_csv(a.run/'previous_annual_vs_revised_summer.csv',index=False)
    ranking=[]
    for (branch,ssp,variant),group in city.groupby(['reconstruction','ssp','rr_variant']):
        for level in [10,20,30]:
            pair=group[group.scenario.isin([f'green{level}',f'target{level}'])]
            p=pair.pivot(index='cause',columns='scenario',values='deaths_averted')
            for cause,row in p.iterrows():ranking.append(dict(reconstruction=branch,ssp=ssp,rr_variant=variant,level=level,cause=cause,target_minus_green=float(row[f'target{level}']-row[f'green{level}'])))
    pd.DataFrame(ranking).to_csv(a.run/'green_target_ranking.csv',index=False)
    rasterchecks=[];gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    files=sorted((a.run/'rasters').glob('*.tif'));assert len(files)==45
    for p in files:
        ds=gdal.Open(str(p));v=ds.ReadAsArray();nd=ds.GetRasterBand(1).GetNoDataValue();valid=v!=nd;assert np.isfinite(v[valid]).all() and valid.sum()==m['common_cells']
        scenario=p.name.split('_')[0];cause=p.name[len(scenario)+1:].replace('_summer_deaths_averted.tif','')
        total=primary.query('scenario==@scenario and cause==@cause and rr_variant=="central"').deaths_averted.item()
        e=float(v[valid].sum(dtype=float)-total);assert abs(e)<1e-5
        rasterchecks.append(dict(file=str(p),sum_minus_float64=e,sha256=sha256(p)))
    out=dict(city_rows=len(city),lsoa_rows=len(lsoa),borough_rows=len(borough),primary_rasters=len(files),reconciliation=errors,all_target_exceed_green=all(r['target_minus_green']>0 for r in ranking),ranking_comparisons=len(ranking),rr_endpoints_ordered_with_signed_harm=True,raster_checks=rasterchecks)
    output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='raster_checks'})

if __name__=='__main__':main()
