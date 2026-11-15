"""Controlled five-outcome pilot; preserves production and refuses output reuse.

Extract monthly fractions with bundled Python (openpyxl):
  python pilot_five_outcome_thresholds.py extract WORKBOOK NEW_CSV
Run with the InVEST Python environment:
  python pilot_five_outcome_thresholds.py run DATA_ROOT MONTHLY_CSV NEW_OUTPUT
"""
import argparse
import calendar
import csv
import json
from pathlib import Path


def extract(workbook, output):
    import openpyxl
    from collections import defaultdict
    if output.exists():
        raise FileExistsError(output)
    totals = defaultdict(lambda: defaultdict(float))
    seen = set()
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    for year, month, code, *ages in wb['1'].iter_rows(min_row=6, values_only=True):
        if year != 2021:
            continue
        assert (month, code) not in seen
        seen.add((month, code))
        assert all(isinstance(x, (int, float)) and x >= 0 for x in ages)
        groups = ['all_cause'] if code == 'Neonate' or 'A00' <= code <= 'R99' or 'U00' <= code <= 'Y89' else []
        for name, lo, hi in [('mental_disorder','F00','F99'), ('cardiovascular','I00','I99'),
                             ('respiratory','J00','J99'), ('self_harm','X60','X84')]:
            if lo <= code <= hi:
                groups.append(name)
        for name in groups:
            totals[name][month] += sum(ages)
    with output.open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['cause','month','count','annual','fraction'])
        for name, months in totals.items():
            assert len(months) == 12
            annual = sum(months.values())
            for m in range(1,13):
                n = months[calendar.month_name[m]]
                writer.writerow([name,m,n,annual,n/annual])


def run(root, monthly, output):
    import numpy as np
    import pandas as pd
    from osgeo import gdal
    from pilot_london_heat_response import london_grid, sha256
    if output.exists():
        raise FileExistsError(output)
    gdal.UseExceptions()
    here = Path(__file__).resolve()
    cfgpath = here.with_name('health-analysis-revised-equal-area.example.json')
    cfg = json.loads(cfgpath.read_text())
    candidates = here.parents[2]/'docs/manuscript/manuscript_health_response_candidates.csv'
    rrtable = pd.read_csv(candidates).set_index('endpoint')
    months = pd.read_csv(monthly).set_index(['cause','month'])
    diag = root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    station = diag/'london_heat_response_2026-11-12/station_daily_temperature.csv'
    daily = pd.read_csv(station, parse_dates=['date']).set_index('date')
    daily['lag01_tmax'] = daily['max'].rolling(2).mean()
    summer = daily.loc['2021-06-01':'2021-08-31'].copy()
    assert len(summer) == 92 and summer.lag01_tmax.notna().all()
    excess = np.maximum(summer.lag01_tmax.to_numpy()-24.7, 0)
    temps = {s:diag/('nodata_target30_2026-11-12' if s=='target30' else 'nodata_correction_2026-11-12')/f'{s}_nodata0/T_air_test.tif' for s in ['baseline','green30','target30']}
    paths = {**temps, **{c['name']:root/c['mortality_raster'] for c in cfg['causes']}, 'population':root/cfg['population_2021']}
    vector = root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    _, zones, _ = london_grid(vector, list(paths.values()))
    valid = zones > 0
    base = gdal.Open(str(temps['baseline'])).ReadAsArray()[valid].astype(float)
    records = []

    def integrate(sorted_c, weights, rr, q):
        # Exact cumulative sums; no cooling bins and no pixel-by-day array.
        cw = np.r_[0., np.cumsum(weights)]
        ca = np.r_[0., np.cumsum(weights * -np.expm1(-np.log(rr)*sorted_c))]
        k = np.searchsorted(sorted_c, q, side='right')
        return ca[k] + (cw[-1]-cw[k]) * -np.expm1(-np.log(rr)*q)

    c = np.array([0., .2, 1., 2.]); w = np.array([1.,2.,3.,4.]); q = np.array([0.,.5,3.])
    assert np.allclose(integrate(c,w,1.03,q), [sum(w*-np.expm1(-np.log(1.03)*np.minimum(c,x))) for x in q])
    for scenario in temps:
        cooling = base-gdal.Open(str(temps[scenario])).ReadAsArray()[valid].astype(float)
        assert np.all(cooling >= 0), 'Negative cooling requires general integration'
        order = np.argsort(cooling); sorted_c = cooling[order]
        for cause in cfg['causes']:
            name = cause['name']; candidate = rrtable.loc[name]
            deaths = gdal.Open(str(paths[name])).ReadAsArray()[valid].astype(float)[order]
            assert np.isfinite(deaths).all() and (deaths >= 0).all()
            # Same national cause-specific monthly allocation in all seasonal arms.
            scale = np.array([months.loc[(name,t.month),'fraction']/calendar.monthrange(2021,t.month)[1] for t in summer.index])
            settings = [('current_rr_annual',cause['rr_1c'],1.,False,True),
                        ('current_rr_seasonal',cause['rr_1c'],1.,False,False),
                        ('candidate_rr_seasonal_no_threshold',candidate.rr_per_1c,1.,False,False)]
            for transfer in [1., .5]:
                for label, rr in [('central',candidate.rr_per_1c),('rr_low',candidate.ci95_lower),('rr_high',candidate.ci95_upper)]:
                    settings.append((f'candidate_threshold_{label}',rr,transfer,True,False))
            for method, rr, transfer, threshold, annual in settings:
                qs = excess if threshold else np.full(92,100.)
                benefit = integrate(sorted_c*transfer, deaths, rr, qs)
                value = float(benefit[0] if annual else benefit @ scale)
                records.append(dict(cause=name,scenario=scenario,method=method,rr=float(rr),cooling_transfer=transfer,
                    avoided=value,annual_deaths_common_mask=float(deaths.sum()),summer_deaths_common_mask=float(deaths.sum()*scale.sum())))
        print('Completed '+scenario,flush=True)
    result = pd.DataFrame(records)
    assert (result.loc[result.scenario=='baseline','avoided']==0).all()
    # Reproduce previous scaling pilot; this isolates response changes from grid effects.
    refpath=diag/'seasonal_cause_sensitivity_2026-11-12/conditional_cause_comparison.csv'
    ref=pd.read_csv(refpath)
    for r in result.query("method == 'current_rr_seasonal'").itertuples():
        old=ref.query('cause == @r.cause and scenario == @r.scenario and method == "england_cause_summer"').avoided.item()
        assert abs(old-r.avoided)<.001,(r.cause,r.scenario,old,r.avoided)
    inputs={**paths,'monthly_fractions':monthly,'station_daily':station,'candidates':candidates,'config':cfgpath,'script':here,'regression_reference':refpath,
            **{'boundary_'+ext:vector.with_suffix('.'+ext) for ext in ['shp','shx','dbf','prj']}}
    manifest=dict(status='conditional diagnostic; not production or future projection',
        response_doi='10.1136/oem.2010.059782',days=92,threshold_c=24.7,
        days_above_threshold=int((excess>0).sum()),common_cells=int(valid.sum()),
        checks=['direct integration equivalence','baseline zero','no negative cooling','previous seasonal pilot reproduced within 0.001 deaths'],
        assumptions=['2021 St James Park hourly maxima proxy daily Tmax; single weather series across London',
         'lag 0-1 arithmetic average before threshold; constant cooling also on pre-June lag day',
         'England cause-specific monthly occurrence fractions applied to London annual registered-death rasters; uniform mortality within month',
         'fixed 2021 population-based mortality allocation; missing cells not renormalized',
         'corrected 25/5 UCM contrast transferred to daily Tmax at 1 or 0.5; unvalidated scenarios not confidence bounds',
         '1993-2006 pooled all-age slopes transferred to London; no new acclimatisation or cold effect',
         'annual arm uses corrected rasters and corrected risk formula, not exact original production totals'],
        uncertainty='RR endpoints only, not total model uncertainty',
        sources={k:dict(path=str(p),sha256=sha256(p)) for k,p in inputs.items()})
    output.mkdir(parents=True)
    result.to_csv(output/'citywide_comparison.csv',index=False)
    summer.to_csv(output/'daily_exposure.csv',index_label='date')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(result.query("scenario != 'baseline' and (method == 'current_rr_seasonal' or method == 'candidate_threshold_central')").to_string(index=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['extract','run']);p.add_argument('paths',nargs='+',type=Path)
    a=p.parse_args()
    if a.mode=='extract':extract(*a.paths)
    else:run(*a.paths)
