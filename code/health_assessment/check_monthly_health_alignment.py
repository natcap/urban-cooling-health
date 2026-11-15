"""Partition existing seasonal health by month, preserving lag and shared support.

Diagnostic only: no production outputs or assumptions are changed. Evaluate
daily risks first; do not divide summer deaths by three. England monthly cause
fractions remain the existing proxy for seasonal London mortality allocation.
"""
import argparse, calendar, json
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
from pilot_spatial_reconstructed_health import london_grid, sha256, integrate


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    repo=Path(__file__).resolve().parents[2]
    diag=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    mf=diag/'remaining_corrected_temperatures_2026-11-12_v1/manifest.json'
    m=json.loads(mf.read_text());paths={s:Path(p) for s,p in m['scenario_paths'].items()}
    for s,p in paths.items():assert sha256(p)==m['outputs'][s]
    cf=Path(__file__).with_name('health-analysis-revised-equal-area.example.json')
    cfg=json.loads(cf.read_text());mort={c['name']:a.root/c['mortality_raster'] for c in cfg['causes']}
    pop=a.root/cfg['population_2021'];vector=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    _,zones,_=london_grid(vector,[*paths.values(),pop,*mort.values()]);valid=zones>0
    assert valid.sum()==14258588
    b=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float);ob=np.argsort(b)
    of=diag/'nine_scenario_health_readiness_2026-11-12_v2/shared_daily_offsets.csv'
    d=pd.read_csv(of,parse_dates=['date']).query("reconstruction=='worldclim_aligned' and ssp=='ssp370'").copy()
    d['month']=d.date.dt.month
    assert len(d)==1380 and not d.duplicated(['gcm','weather_year','date']).any()
    for _,g in d.groupby(['gcm','weather_year']):assert g.groupby('month').size().to_dict()=={6:30,7:31,8:31}
    ff=diag/'five_outcome_thresholds_2026-11-12/monthly_cause_fractions.csv'
    fractions=pd.read_csv(ff).set_index(['cause','month'])
    rf=repo/'docs/manuscript/manuscript_health_response_candidates.csv';rrs=pd.read_csv(rf).set_index('endpoint')
    oldfile=diag/'seasonal_health_all_scenarios_2026-11-12_v1/citywide_results.csv'
    old=pd.read_csv(oldfile).query("reconstruction=='worldclim_aligned' and ssp=='ssp370' and rr_variant=='central'").set_index(['scenario','cause'])
    sources=[mf,cf,of,ff,rf,oldfile,Path(__file__),Path(__file__).with_name('pilot_spatial_reconstructed_health.py'),pop,vector,vector.with_suffix('.dbf'),*paths.values(),*mort.values()]
    hashes={str(p):sha256(p) for p in sources};records=[];checks=[]
    # The already-prepared lag01 offsets include the preceding day at month
    # boundaries. Filtering by outcome day must not reset the lag on July 1.
    off=d.lag01_offset_c.to_numpy()-24.7
    for scenario in ['baseline','green30','target30']:
        s=gdal.Open(str(paths[scenario])).ReadAsArray()[valid].astype(float);cool=b-s
        assert cool.min()>-1e-8
        order=np.argsort(s)
        for cause,mp in mort.items():
            deaths=gdal.Open(str(mp)).ReadAsArray()[valid].astype(float)
            rr=float(rrs.loc[cause,'rr_per_1c']);assert float(rrs.loc[cause,'threshold_c'])==24.7
            values=np.zeros(len(d)) if scenario=='baseline' else integrate(b,cool,deaths,rr,off,ob,order)
            # Independent direct daily arithmetic on sampled pixels verifies
            # the reused prefix-sum integration without pixel/day binning.
            ix=np.random.default_rng(20261113).choice(len(b),1000,replace=False)
            test=integrate(b[ix],cool[ix],deaths[ix],rr,off[:20],np.argsort(b[ix]),np.argsort(s[ix]))
            direct=(deaths[ix,None]*-np.expm1(np.log(rr)*(np.maximum(s[ix,None]+off[:20],0)-np.maximum(b[ix,None]+off[:20],0)))).sum(axis=0)
            assert np.max(abs(test-direct))<1e-8
            x=d[['gcm','weather_year','date','month']].copy();x['scenario']=scenario;x['cause']=cause
            weights=np.array([float(fractions.loc[(cause,int(mm)),'fraction'])/calendar.monthrange(2021,int(mm))[1] for mm in x.month])
            x['deaths_averted']=values*weights
            total=x.deaths_averted.sum()/15;reference=float(old.loc[(scenario,cause),'deaths_averted'])
            assert abs(total-reference)<1e-5,(scenario,cause,total,reference)
            checks.append(dict(scenario=scenario,cause=cause,summer_replay_error=total-reference,direct_sample_error=float(np.max(abs(test-direct)))))
            records.append(x)
        print('DONE',scenario,flush=True)
    daily=pd.concat(records,ignore_index=True)
    members=daily.groupby(['scenario','cause','gcm','weather_year','month'],as_index=False).deaths_averted.sum()
    members['period']=members.month.map({6:'June',7:'July',8:'August'});members['days']=members.month.map({6:30,7:31,8:31})
    summer=members.groupby(['scenario','cause','gcm','weather_year'],as_index=False).deaths_averted.sum();summer['period']='JJA';summer['days']=92
    members=pd.concat([members,summer],ignore_index=True);members['deaths_averted_per_day']=members.deaths_averted/members.days
    summary=members.groupby(['scenario','cause','period','days'],as_index=False).agg(deaths_averted=('deaths_averted','mean'),deaths_averted_per_day=('deaths_averted_per_day','mean'),member_min_per_day=('deaths_averted_per_day','min'),member_max_per_day=('deaths_averted_per_day','max'),members=('gcm','size'))
    assert (summary.members==15).all()
    for p,h in hashes.items():assert sha256(p)==h
    a.output.mkdir(parents=True)
    daily.to_csv(a.output/'daily_health.csv',index=False);members.to_csv(a.output/'monthly_health_members.csv',index=False);summary.to_csv(a.output/'monthly_health_summary.csv',index=False)
    pd.DataFrame(checks).to_csv(a.output/'checks.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps(dict(inputs_unchanged=True,sources=hashes,health_cells=int(valid.sum()),primary='WorldClim aligned SSP370 2041-2060; central response; fixed2021 mortality/population',lag='retained precomputed lag01 offsets, including month-boundary preceding days',weighting='equal15 GCM-weather members; actual calendar-day month lengths',limitations=['July UCM cooling held across JJA; this tests reconstructed exposure, not month-specific cooling validation','England monthly cause fractions transferred to London; not observed London monthly cause totals','member range is scenario/weather variability, not a confidence interval']),indent=2))


if __name__=='__main__':main()
