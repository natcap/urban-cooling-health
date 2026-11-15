"""Isolated London-response sensitivity; NOT a calibrated future health projection.

Usage: python pilot_london_heat_response.py DATA_ROOT NEW_OUTPUT_DIRECTORY
Arbuthnott 2020 DOI 10.1016/j.envint.2020.105966: all-cause RR 1.039
(1.035–1.043), daily mean (Tmax+Tmin)/2, threshold 18.9 C, lag 0–2.
Hourly station extrema approximate, but do not reproduce, source daily extrema.
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


def response(cooling, weights, rr, excess):
    """Exact weighted sum of 1-exp(-beta*min(cooling, heat excess)).

    Sorted prefix sums avoid an enormous pixel-by-day array. No cooling bins.
    Input cooling must be non-negative. Threshold crossing stops benefit at h.
    """
    order = np.argsort(cooling)
    c, w = cooling[order], weights[order]
    cw = np.r_[0., np.cumsum(w)]
    ca = np.r_[0., np.cumsum(w * -np.expm1(-np.log(rr)*c))]
    q = np.asarray(excess)
    k = np.searchsorted(c, q, side='right')
    return ca[k] + (cw[-1]-cw[k]) * -np.expm1(-np.log(rr)*q)


def weather(path):
    with path.open() as f:
        header = next(i for i,line in enumerate(f) if line.startswith('ob_time,'))
    x = pd.read_csv(path, skiprows=header, usecols=[
        'ob_time','version_num','air_temperature','air_temperature_q','meto_stmp_time'])
    x['date_time'] = pd.to_datetime(x.ob_time, errors='coerce')
    for col in ['version_num','air_temperature','air_temperature_q']:
        x[col] = pd.to_numeric(x[col], errors='coerce')
    x = x[(x.version_num == 1) & ((x.air_temperature_q//100)%10 == 0)
          & x.date_time.notna() & x.air_temperature.notna()]
    x = x.sort_values('meto_stmp_time').drop_duplicates('date_time', keep='last')
    d = x.groupby(x.date_time.dt.floor('D')).air_temperature.agg(['count','min','max','mean'])
    d = d.reindex(pd.date_range('2021-05-30','2021-08-31'))
    # Fail rather than silently impute missing days or inflate remaining weights.
    assert (d['count'] >= 21).all(), d[d['count'] < 21]
    d['extrema_mean'] = (d['min']+d['max'])/2
    d['lag02_mean'] = d.extrema_mean.rolling(3).mean()
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root',type=Path); parser.add_argument('output',type=Path)
    a=parser.parse_args(); gdal.UseExceptions()
    if a.output.exists(): raise FileExistsError(a.output)
    # Check prefix integration against a direct threshold-crossing calculation.
    c=np.array([0.,.2,1.,2.]); w=np.array([.1,.2,.3,.4]); q=np.array([0.,.5,3.])
    assert np.allclose(response(c,w,1.039,q),
        [np.sum(w*-np.expm1(-np.log(1.039)*np.minimum(c,v))) for v in q])
    inputs=a.data_root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs'
    runs=a.data_root/'2_postprocess_intermediate/UCM_official_runs'
    diag=runs/'diagnostics/nodata_correction_2026-11-12'
    station=next((inputs/'climate/uhi_station_pilot_20261111_v1/hourly/00697').glob('*_2021.csv'))
    daily=weather(station); summer=daily.loc['2021-06-01':].copy()
    assert len(summer)==92 and summer.lag02_mean.notna().all()
    seasonal=diag/'ons_seasonal_2021_boroughs.csv'
    source=pd.read_csv(seasonal).set_index('code')
    paths={'population':inputs/'pop_raster/resampled_10m/gbr_pop_2021_10m_count_preserved_bng.tif'}
    for s in ['baseline','green30','target30']:
        folder=runs/'diagnostics/nodata_target30_2026-11-12' if s=='target30' else diag
        paths[s]=folder/f'{s}_nodata0/T_air_test.tif'
    vector=inputs/'AOIs/London_Borough_aoi.shp'
    rows,fullzones,_=london_grid(vector,[paths['population']])
    pop=gdal.Open(str(paths['population'])).ReadAsArray().astype(float)
    denominators=np.bincount(fullzones.ravel(),weights=pop.ravel(),minlength=34)[1:]
    rows,zones,_=london_grid(vector,list(paths.values()))
    valid=zones>0; z=zones[valid]-1; weights=pop[valid]/denominators[z]
    base=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float)
    # These transfer fractions are illustrative assumptions, NOT confidence bounds.
    settings=[('yang_unrestricted',1.02,1.,False),('london_slope_only',1.039,1.,False),
              ('yang_threshold_diagnostic',1.02,1.,True)]
    for f in [1.,.5]:
        for rr in [1.035,1.039,1.043]:
            settings.append(('london_threshold',rr,f,True))
    records=[]
    for scenario in ['baseline','green30','target30']:
        cooling=base-gdal.Open(str(paths[scenario])).ReadAsArray()[valid].astype(float)
        assert np.all(cooling>=0), 'Warming requires general risk-difference integration'
        for j,(code,name,_) in enumerate(rows):
            take=z==j; c=cooling[take]; w=weights[take]
            # Registered monthly deaths spread uniformly within each month.
            dd=np.array([source.loc[code,{6:'june',7:'july',8:'august'}[t.month]] /
                         calendar.monthrange(2021,t.month)[1] for t in summer.index])
            for label,rr,f,threshold in settings:
                excess=np.maximum(summer.lag02_mean.to_numpy()-18.9,0) if threshold else np.full(92,100.)
                benefit=response(c*f,w,rr,excess)
                records.append(dict(scenario=scenario,method=label,rr=rr,mean_cooling_fraction=f,
                    code=code,name=name,avoided=float(np.dot(dd,benefit))))
        print('Completed '+scenario,flush=True)
    b=pd.DataFrame(records)
    city=b.groupby(['scenario','method','rr','mean_cooling_fraction'],as_index=False).avoided.sum()
    # Reproduce the prior same-mask seasonal pilot as a regression check.
    reference=pd.read_csv(runs/'diagnostics/seasonal_health_2026-11-12/citywide_comparison.csv')
    for s in ['green30','target30']:
        old=reference[(reference.scenario==s)&(reference.period=='june_august_ons')].avoided.item()
        new=city[(city.scenario==s)&(city.method=='yang_unrestricted')].avoided.item()
        assert abs(old-new)<.001,(s,old,new)
    coverage=np.bincount(z,weights=weights,minlength=33)
    sources={**paths,'weather':station,'monthly_deaths':seasonal,
             **{'boundary_'+ext:vector.with_suffix('.'+ext) for ext in ['shp','shx','dbf','prj']},
             'script':Path(__file__)}
    manifest={'status':'hybrid historical-weather response sensitivity; not production or a 2050 forecast',
      'response_doi':'10.1016/j.envint.2020.105966','response_table':'Table 2, p5; Methods 2.1–2.2 pp2–3',
      'station':'00697 St James\u2019s Park','days':92,'minimum_hourly_count':int(daily['count'].min()),
      'days_lag02_above_18_9':int((summer.lag02_mean>18.9).sum()),
      'population_coverage_min':float(coverage.min()),'baseline_registered_jja_deaths':float(source.summer.sum()),
      'assumptions':['station daily (hourly max+min)/2 proxy; not original gridded daily extrema',
        'threshold applied after 3-day mean; equal lag weights, not refitted distributed-lag coefficients',
        'single station baseline exposure assigned throughout London',
        'constant corrected 25/5 UCM cooling transferred at 1 or 0.5 to daily mean, including pre-June lag days',
        '2021 monthly registered deaths uniform within month, fixed 2021 population',
        'heat-only effects; no cold mortality or future acclimatisation'],
      'uncertainty':'RR endpoints only; transfer fractions are scenarios, not uncertainty intervals',
      'sources':{k:{'path':str(v),'sha256':sha256(v)} for k,v in sources.items()}}
    a.output.mkdir(parents=True)
    daily.to_csv(a.output/'station_daily_temperature.csv',index_label='date')
    b.to_csv(a.output/'borough_comparison.csv',index=False)
    city.to_csv(a.output/'citywide_comparison.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(city.to_string(index=False)); print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},indent=2))


if __name__=='__main__': main()
