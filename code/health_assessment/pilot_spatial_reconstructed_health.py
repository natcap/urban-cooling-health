"""Five-outcome pixel-specific health diagnostic; never overwrites production.

Usage (InVEST Python): python pilot_spatial_reconstructed_health.py DATA_ROOT NEW_OUTPUT
Reuses the verified temperature reconstruction offsets, not city-mean exposures.
"""
import argparse
import calendar
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'post_processing_layers'))
from valuation_accounting import london_grid, sha256
gdal.UseExceptions()
gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')


def tail(values, order, k):
    prefix = np.r_[0., np.cumsum(values[order])]
    return prefix[-1] - prefix[k]


def integrate(b, c, deaths, rr, offsets, ob, os):
    """Exact pixel sum D*[1-exp(-beta*min(c,max(b+offset-h,0)))].

    Split pixels into baseline-cold, threshold-crossing and scenario-hot sets.
    Two sorted arrays replace a huge pixel-by-day matrix, without binning.
    Offsets include -threshold. Cooling must be non-negative.
    """
    beta = np.log(rr)
    kb = np.searchsorted(b[ob], -offsets, side='right')
    ks = np.searchsorted((b-c)[os], -offsets, side='right')
    first = tail(deaths, ob, kb)
    second = tail(deaths*np.exp(-beta*c), os, ks)
    w = deaths*np.exp(-beta*b)
    cross = tail(w, ob, kb)-tail(w, os, ks)
    result = first-second-np.exp(-beta*offsets)*cross
    assert result.min() > -1e-6
    return np.maximum(result,0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    repo=Path(__file__).resolve().parents[2]
    diag=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    temp=diag/'temperature_reconstructions_2050_2026-11-12_v1'
    mf=temp/'manifest.json'; manifest=json.loads(mf.read_text())
    for x in manifest['inputs']:
        assert sha256(x['path'])==x['sha256'],f"Temperature input changed: {x['path']}"
    cfgpath=Path(__file__).with_name('health-analysis-revised-equal-area.example.json')
    cfg=json.loads(cfgpath.read_text())
    candidate=repo/'docs/manuscript/manuscript_health_response_candidates.csv'
    rrtable=pd.read_csv(candidate).set_index('endpoint')
    assert (rrtable.threshold_c==24.7).all()
    fractions=diag/'five_outcome_thresholds_2026-11-12/monthly_cause_fractions.csv'
    monthly=pd.read_csv(fractions).set_index(['cause','month'])
    offset_path=temp/'daily_temperature_exposure.csv'
    # Baseline offsets are shared by all land-use scenarios and transfer fractions.
    d=pd.read_csv(offset_path,parse_dates=['date']).query("scenario=='baseline'").reset_index(drop=True)
    keys=['reconstruction','ssp','gcm','weather_year']
    assert len(d)==5520 and not d.duplicated(keys+['date']).any()
    paths={s:diag/('nodata_target30_2026-11-12' if s=='target30' else 'nodata_correction_2026-11-12')/f'{s}_nodata0/T_air_test.tif' for s in ['baseline','green30','target30']}
    poppath=a.root/cfg['population_2021']
    boundary=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    _,zones,_=london_grid(boundary,[*paths.values(),poppath]);valid=zones>0
    first=gdal.Open(str(paths['baseline']));gt=first.GetGeoTransform()
    bounds=[gt[0],gt[3]+first.RasterYSize*gt[5],gt[0]+first.RasterXSize*gt[1],gt[3]]
    for item in manifest['inputs']:
        if 'wc2.1_30s_tmax_' not in item['path']:continue
        for month in [5,6,7,8]:
            band=gdal.Translate('',item['path'],format='MEM',bandList=[month])
            w=gdal.Warp('',band,format='MEM',dstSRS=first.GetProjection(),outputBounds=bounds,
                width=first.RasterXSize,height=first.RasterYSize,resampleAlg='bilinear',dstNodata=-9999,outputType=gdal.GDT_Float32)
            v=w.ReadAsArray();valid &= np.isfinite(v)&(v!=-9999);w=band=None
    assert int(valid.sum())==manifest['common_cells']
    pop=gdal.Open(str(poppath)).ReadAsArray()[valid].astype(float)
    assert np.isclose(pop.sum(),manifest['population_on_common_mask'],rtol=0,atol=1e-6)
    del pop,zones
    b=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float)
    assert abs(b.mean()-manifest['baseline_ucm_area_mean_c'])<1e-10
    ob=np.argsort(b);offsets=d.lag01_offset_c.to_numpy()-24.7
    out=[];audit=[];sources=[mf,offset_path,cfgpath,candidate,fractions,Path(__file__)]
    # Random synthetic and real-pixel checks compare with direct risk differences.
    rng=np.random.default_rng(20261112);sample=rng.choice(len(b),size=1500,replace=False)
    for scenario,path in paths.items():
        cool=b-gdal.Open(str(path)).ReadAsArray()[valid].astype(float)
        assert cool.min()>=0
        for transfer in ([1.] if scenario=='baseline' else [1.,.5]):
            c=cool*transfer;os=np.argsort(b-c)
            for cause in cfg['causes']:
                name=cause['name'];p=a.root/cause['mortality_raster'];sources.append(p)
                ds=gdal.Open(str(p))
                assert ds.GetGeoTransform()==gt and ds.GetProjection()==first.GetProjection()
                death=ds.ReadAsArray()[valid].astype(float)
                nd=ds.GetRasterBand(1).GetNoDataValue()
                assert np.isfinite(death).all() and (death>=0).all() and (nd is None or (death!=nd).all())
                scale=np.array([monthly.loc[(name,t.month),'fraction']/calendar.monthrange(2021,t.month)[1] for t in d.date])
                # Fixed 2021 deaths, independent of the weather-pattern year.
                for label,col in [('central','rr_per_1c'),('rr_low','ci95_lower'),('rr_high','ci95_upper')]:
                    rr=float(rrtable.loc[name,col]);values=integrate(b,c,death,rr,offsets,ob,os)
                    if scenario=='baseline':
                        assert values.max()<1e-6
                        values[:]=0
                    bs,cs,ws=b[sample],c[sample],death[sample]
                    xx=np.r_[-50.,0.,50.,offsets[::500]]
                    exact=np.array([np.sum(ws*-np.expm1(np.log(rr)*(np.maximum(bs-cs+x,0)-np.maximum(bs+x,0)))) for x in xx])
                    fast=integrate(bs,cs,ws,rr,xx,np.argsort(bs),np.argsort(bs-cs))
                    err=float(np.max(np.abs(exact-fast)));assert err<1e-8
                    audit.append(dict(scenario=scenario,cooling_transfer=transfer,cause=name,rr_variant=label,max_direct_error=err))
                    frame=d[keys+['date']].copy();frame['avoided']=values*scale
                    result=frame.groupby(keys,as_index=False).avoided.sum()
                    for r in result.to_dict('records'):
                        out.append(dict(**r,scenario=scenario,cooling_transfer=transfer,cause=name,
                            rr_variant=label,rr=rr,annual_deaths_common_mask=float(death.sum()),
                            summer_deaths_common_mask=float(death.sum()*sum(monthly.loc[(name,m),'fraction'] for m in [6,7,8]))))
                print(scenario,transfer,name,'complete',flush=True)
    result=pd.DataFrame(out)
    assert len(result)==4500
    group=['reconstruction','ssp','scenario','cooling_transfer','cause','rr_variant']
    avg=result.groupby(group,as_index=False).avoided.agg(['mean','min','max']).reset_index()
    # Check structural expectations for every model and weather pattern, not just averages.
    kk=keys+['scenario','cause','rr_variant']
    f=result.query("scenario!='baseline'").pivot(index=kk,columns='cooling_transfer',values='avoided')
    assert (f[1.]>=f[.5]-1e-6).all()
    rr=result.pivot(index=keys+['scenario','cause','cooling_transfer'],columns='rr_variant',values='avoided')
    assert (rr.rr_low<=rr.central+1e-6).all() and (rr.central<=rr.rr_high+1e-6).all()
    paired=result.pivot(index=['ssp','gcm','weather_year','scenario','cooling_transfer','cause','rr_variant'],columns='reconstruction',values='avoided').reset_index()
    paired['anchored_minus_aligned']=paired.ucm_25_5_anchored-paired.worldclim_aligned
    assert (paired.anchored_minus_aligned>=-1e-6).all()
    a.output.mkdir(parents=True)
    result.to_csv(a.output/'citywide_health_comparison.csv',index=False)
    avg.to_csv(a.output/'ensemble_summary.csv',index=False)
    paired.to_csv(a.output/'paired_reconstruction_difference.csv',index=False)
    pd.DataFrame(audit).to_csv(a.output/'direct_calculation_checks.csv',index=False)
    record=dict(status='diagnostic only; not adopted production health estimates',preferred='worldclim_aligned',
        temperature_manifest=str(mf),common_cells=int(valid.sum()),
        formula='D_i_month/day * [1-exp(log(RR)*(max(Tscenario_lag01-24.7,0)-max(Tbaseline_lag01-24.7,0)))]',
        interpretation='Avoided deaths per modelled June-August summer conditional on fixed 2021 population and baseline mortality',
        baseline_mortality_assumption='Observed 2021 baseline daily counts allocated seasonally are held fixed as counterfactual future-baseline counts; not projected mortality risk or demographic growth.',
        checks=['temperature input hashes unchanged','identical temperature/health footprint and population denominator',
                'direct pixel formula agrees with exact cumulative sums','baseline zero','RR endpoint ordering',
                'full >= half cooling benefit','anchored >= aligned benefit','4500 unique result combinations'],
        limitations=['Regional pooled all-age coefficients and London regional threshold applied to pixels; spatial transfer not epidemiologically validated',
                     'England occurrence-month cause fractions approximate London registration-month distribution',
                     'Fixed 2021 mortality counts are not adjusted for future baseline warming; results are conditional intervention comparisons',
                     'RR endpoint ranges exclude structural, climate, mortality allocation and temperature-transfer uncertainty',
                     'Temperature reconstruction limitations and common-footprint loss carry forward; no renormalization',
                     'No LSOA equity curves or new spatial health maps; exact pixel sums reported citywide'],
        sources=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in sorted(set(sources))])
    (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(avg.query("rr_variant=='central' and cooling_transfer==1 and scenario!='baseline'").to_string(index=False))


if __name__=='__main__':main()
