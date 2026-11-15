"""Isolated, conditional all-cause seasonal diagnostic; never replaces production.

Usage: python pilot_seasonal_health.py DATA_ROOT NEW_OUTPUT_DIRECTORY
Assumes the July-parameterised cooling contrast persists throughout each period.
No daily threshold/lag or compatible exposure metric has yet been established.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'post_processing_layers'))
from valuation_accounting import london_grid, sha256


def avoided(deaths, delta, rr, legacy=False):
    # D is baseline-state mortality: Ds=D*RR**delta, hence avoided=D-Ds.
    # The legacy AF formula instead treats D as scenario-state mortality.
    beta = np.log(rr)
    return deaths * (np.expm1(-beta * delta) if legacy else -np.expm1(beta * delta))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_root', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args(); gdal.UseExceptions()
    if a.output.exists(): raise FileExistsError(a.output)
    assert avoided(1000, 0, 1.02) == 0
    assert np.isclose(avoided(1000, -1, 1.02), 19.6078431372549)
    assert np.isclose(avoided(1000, 1, 1.02), -20)
    inputs = a.data_root / '1_preprocess/UrbanCoolingModel/OfficialWorkingInputs'
    runs = a.data_root / '2_postprocess_intermediate/UCM_official_runs'
    diag = runs / 'diagnostics/nodata_correction_2026-11-12'
    paths = {'population': inputs/'pop_raster/resampled_10m/gbr_pop_2021_10m_count_preserved_bng.tif',
             'mortality': inputs/'health_rasters_population_weighted_2021_count_preserved/baseline_deaths_all_cause_population_weighted_2021.tif'}
    for s in ['baseline', 'green30', 'target30']:
        folder = runs/'diagnostics/nodata_target30_2026-11-12' if s == 'target30' else diag
        paths['corrected_'+s] = folder/f'{s}_nodata0/T_air_test.tif'
        found = list((runs/f'production/equal_area_2026-09/01_temperature_invest3202/{s}').glob('T_air_*25deg_5uhi*.tif'))
        assert len(found) == 1
        paths['production_'+s] = found[0]
    seasonal = diag/'ons_seasonal_2021_boroughs.csv'
    source = pd.read_csv(seasonal).set_index('code')
    vector = inputs/'AOIs/London_Borough_aoi.shp'
    rows, fullzones, _ = london_grid(vector, [paths['population']])
    codes = [r[0] for r in rows]
    assert set(codes) == set(source.index) and source.index.is_unique
    source = source.loc[codes]
    pop = gdal.Open(str(paths['population'])).ReadAsArray().astype(float)
    poptot = np.bincount(fullzones.ravel(), weights=pop.ravel(), minlength=34)[1:]
    rows, zones, counts = london_grid(vector, list(paths.values()))
    valid = zones > 0; z = zones[valid]-1; pop = pop[valid]
    coverage = np.bincount(z, weights=pop, minlength=33)/poptot
    # Never renormalize after temperature masking: disclose any lost coverage.
    weights = pop/poptot[z]
    annual = gdal.Open(str(paths['mortality'])).ReadAsArray()[valid].astype(float)
    allocated = np.bincount(z, weights=annual, minlength=33)
    qa = pd.DataFrame({'code':codes, 'name':[r[1] for r in rows],
                       'population_coverage':coverage, 'production_annual_on_mask':allocated,
                       'ons_annual':source.annual.values})
    qa['annual_difference'] = qa.production_annual_on_mask-qa.ons_annual
    summaries=[]; borough=[]
    for version, period, legacy, denominator in [
        ('production','annual_production',True,annual),
        ('corrected','annual_production',True,annual),
        ('corrected','annual_production',False,annual),
        ('corrected','annual_ons',False,weights*source.annual.values[z]),
        ('corrected','june_august_ons',False,weights*source.summer.values[z]),
        ('corrected','july_ons',False,weights*source.july.values[z])]:
        base=gdal.Open(str(paths[version+'_baseline'])).ReadAsArray()[valid].astype(float)
        for scenario in ['baseline','green30','target30']:
            delta=gdal.Open(str(paths[version+'_'+scenario])).ReadAsArray()[valid].astype(float)-base
            totals=[]
            for rr in [1.014,1.020,1.026]:
                values=avoided(denominator,delta,rr,legacy)
                b=np.bincount(z,weights=values,minlength=33); totals.append(float(b.sum()))
                if rr==1.020:
                    for code,name,value in zip(codes,qa.name,b):
                        borough.append(dict(version=version,period=period,formula='legacy' if legacy else 'baseline_risk',scenario=scenario,code=code,name=name,avoided=value))
            summaries.append(dict(version=version,period=period,formula='legacy' if legacy else 'baseline_risk',scenario=scenario,denominator=float(denominator.sum()),avoided=totals[1],rr_low_result=totals[0],rr_high_result=totals[2],warming_cells=int((delta>0).sum())))
    files={**paths,'seasonal_csv':seasonal,**{f'boundary_{ext}':vector.with_suffix('.'+ext) for ext in ['shp','shx','dbf','prj']}}
    manifest={'status':'conditional diagnostic, not validated seasonal heat-attributable deaths',
              'assumptions':['constant July-parameterised contrast throughout period','log-linear all-age Oceanic RR applied to every contrast; metric/threshold/lag compatibility unresolved','fixed 2021 population and registered deaths; no demographic projection'],
              'uncertainty':'RR endpoint sensitivity only; not total model uncertainty',
              'common_positive_population_cells':int(valid.sum()),'population_on_mask':float(pop.sum()),
              'population_coverage_min':float(coverage.min()),
              'sources':{k:{'path':str(v),'sha256':sha256(v)} for k,v in files.items()}}
    a.output.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(a.output/'citywide_comparison.csv',index=False)
    pd.DataFrame(borough).to_csv(a.output/'borough_comparison.csv',index=False)
    qa.to_csv(a.output/'denominator_qa.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(qa.to_string(index=False))


if __name__ == '__main__': main()
