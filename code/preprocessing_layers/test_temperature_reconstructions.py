"""Temperature-only 2050 reconstruction comparison; never reads mortality inputs.

Run in the InVEST Python environment with DATA_ROOT NEW_OUTPUT_DIRECTORY.
Existing rasters are read-only; output directories must not already exist.
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

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')  # Do not create input sidecars.


def weather(paths):
    series, qc = [], []
    for path in paths:
        year = int(path.stem[-4:])
        with path.open() as f:
            skip = next(i for i, line in enumerate(f) if line.startswith('ob_time,'))
        x = pd.read_csv(path, skiprows=skip, usecols=[
            'ob_time', 'version_num', 'air_temperature', 'air_temperature_q', 'meto_stmp_time'],
            low_memory=False)
        for c in ['version_num', 'air_temperature', 'air_temperature_q']:
            x[c] = pd.to_numeric(x[c], errors='coerce')
        x['date_time'] = pd.to_datetime(x.ob_time, errors='coerce')
        x = x[(x.version_num == 1) & ((x.air_temperature_q // 100) % 10 == 0)
              & x.date_time.notna() & x.air_temperature.notna()]
        x = x.sort_values('meto_stmp_time').drop_duplicates('date_time', keep='last')
        d = x.groupby(x.date_time.dt.floor('D')).air_temperature.agg(['count', 'max'])
        d = d.reindex(pd.date_range(f'{year}-05-01', f'{year}-08-31'))
        bad = d['count'].fillna(0) < 21
        qc.append(dict(year=year, low_coverage_days=int(bad.sum()), included=not bad.any(),
                       excluded_dates=';'.join(d.index[bad].strftime('%Y-%m-%d'))))
        # Exclude incomplete years rather than filling gaps or selecting only hot days.
        if bad.any():
            continue
        d['year'] = year
        d['month'] = d.index.month
        d['anomaly_c'] = d['max'] - d.groupby('month')['max'].transform('mean')
        assert d.groupby('month').anomaly_c.mean().abs().max() < 1e-12
        series.append(d)
    return pd.concat(series).rename_axis('date'), pd.DataFrame(qc)


def exceedance(sorted_u, prefix_pop, offsets, threshold):
    """Exact fractions, avoiding storage of pixel-by-day rasters."""
    k = np.searchsorted(sorted_u, threshold - np.asarray(offsets), side='right')
    return (len(sorted_u) - k) / len(sorted_u), (prefix_pop[-1] - prefix_pop[k]) / prefix_pop[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data_root', type=Path)
    ap.add_argument('output', type=Path)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    repo = Path(__file__).resolve().parents[2]
    inputs = a.data_root / '1_preprocess/UrbanCoolingModel/OfficialWorkingInputs'
    diag = a.data_root / '2_postprocess_intermediate/UCM_official_runs/diagnostics'
    raster_paths = {s: diag / ('nodata_target30_2026-11-12' if s == 'target30'
                    else 'nodata_correction_2026-11-12') / f'{s}_nodata0/T_air_test.tif'
                    for s in ['baseline', 'green30', 'target30']}
    pop_path = inputs / 'pop_raster/resampled_10m/gbr_pop_2021_10m_count_preserved_bng.tif'
    boundary = inputs / 'AOIs/London_Borough_aoi.shp'
    _, zones, _ = london_grid(boundary, [*raster_paths.values(), pop_path])
    valid = zones > 0
    first = gdal.Open(str(raster_paths['baseline']))
    gt = first.GetGeoTransform()
    bounds = [gt[0], gt[3] + first.RasterYSize * gt[5],
              gt[0] + first.RasterXSize * gt[1], gt[3]]
    original_cells = int(valid.sum())
    full_pop = gdal.Open(str(pop_path)).ReadAsArray()
    original_population = float(full_pop[valid].sum(dtype=float))
    # Native-resolution London clipping leaves small boundary gaps. Use one
    # intersection footprint for both anchors, rather than extrapolating climate.
    for ssp in ['ssp370', 'ssp585']:
        for model in ['ACCESS-CM2', 'IPSL-CM6A-LR', 'MRI-ESM2-0']:
            p = repo / f'data/02_preprocess/GCMs/clipped/wc2.1_30s_tmax_{model}_{ssp}_2041-2060_London.tif'
            for month in [5, 6, 7, 8]:
                band = gdal.Translate('', str(p), format='MEM', bandList=[month])
                w = gdal.Warp('', band, format='MEM', dstSRS=first.GetProjection(),
                    outputBounds=bounds, width=first.RasterXSize, height=first.RasterYSize,
                    resampleAlg='bilinear', dstNodata=-9999, outputType=gdal.GDT_Float32)
                values = w.ReadAsArray()
                valid &= np.isfinite(values) & (values != -9999)
                w = band = None
    assert valid.sum() / original_cells > .95, 'Material climate-footprint loss requires review'
    pop = full_pop[valid].astype(float)
    del full_pop
    assert (pop >= 0).all() and pop.sum() > 0
    arrays = {s: gdal.Open(str(p)).ReadAsArray()[valid].astype(float) for s, p in raster_paths.items()}
    u0_mean = float(arrays['baseline'].mean())
    paths = sorted((inputs / 'climate/uhi_station_pilot_20261111_v1/hourly/00697').glob('*.csv'))
    daily, qc = weather(paths)
    assert 2021 in daily.year.unique()
    climate, sources = [], list(paths) + list(raster_paths.values()) + [pop_path]
    sources += [boundary.with_suffix('.' + ext) for ext in ['shp', 'shx', 'dbf', 'prj']]
    # Same equal-area London footprint for centring UCM and averaging WorldClim.
    # Interpolation provides integration weights, not newly resolved climate detail.
    for ssp in ['ssp370', 'ssp585']:
        for model in ['ACCESS-CM2', 'IPSL-CM6A-LR', 'MRI-ESM2-0']:
            p = repo / f'data/02_preprocess/GCMs/clipped/wc2.1_30s_tmax_{model}_{ssp}_2041-2060_London.tif'
            sources.append(p)
            for month in [5, 6, 7, 8]:
                band = gdal.Translate('', str(p), format='MEM', bandList=[month])
                w = gdal.Warp('', band, format='MEM', dstSRS=first.GetProjection(),
                              outputBounds=bounds, width=first.RasterXSize, height=first.RasterYSize,
                              resampleAlg='bilinear', dstNodata=-9999, outputType=gdal.GDT_Float32)
                values = w.ReadAsArray()[valid]
                assert np.isfinite(values).all() and (values != -9999).all(), 'Incomplete climate coverage'
                assert 5 < values.mean() < 40, 'Check climate units'
                climate.append(dict(ssp=ssp, gcm=model, month=month, mean_tmax_c=float(values.mean(dtype=float))))
                w = band = None
            print('Climate extracted:', ssp, model, flush=True)
    climate = pd.DataFrame(climate)
    offsets, monthly, summary, cooling = [], [], [], []
    # Retain each GCM separately; an ensemble average of nonlinear exposure is
    # not interchangeable with calculating exposure using an ensemble-mean field.
    for scenario, u in arrays.items():
        c = arrays['baseline'] - u
        assert c.min() >= 0
        cooling.append(dict(scenario=scenario, area_mean_c=float(c.mean()),
                            population_mean_c=float(c @ pop / pop.sum()), maximum_c=float(c.max())))
        for transfer in ([1.] if scenario == 'baseline' else [1., .5]):
            spatial = arrays['baseline'] - transfer * c
            order = np.argsort(spatial)
            sorted_u = spatial[order]
            cp = np.r_[0., np.cumsum(pop[order])]
            spatial_mean = float(spatial.mean())
            spatial_pop = float(spatial @ pop / pop.sum())
            quant = np.quantile(spatial, [.05, .5, .95])
            # Validate exact threshold counts against direct pixel arithmetic.
            fa, fp = exceedance(sorted_u, cp, [0.2], 28.)
            mask = spatial + .2 > 28.
            assert abs(fa[0] - mask.mean()) < 1e-12
            assert abs(fp[0] - pop[mask].sum() / pop.sum()) < 1e-10
            for (ssp, model), cl in climate.groupby(['ssp', 'gcm']):
                means = cl.set_index('month').mean_tmax_c
                for branch in ['worldclim_aligned', 'ucm_25_5_anchored']:
                    shift = means - (u0_mean if branch == 'worldclim_aligned' else means.loc[7])
                    for year, d in daily.groupby('year'):
                        raw = d.anomaly_c + d.month.map(shift)
                        lagged = raw.rolling(2).mean()
                        ix = d.month.isin([6, 7, 8])
                        dates = d.index[ix]
                        raw, lagged = raw.loc[ix].to_numpy(), lagged.loc[ix].to_numpy()
                        assert len(raw) == 92 and np.isfinite(lagged).all()
                        fa, fp = exceedance(sorted_u, cp, lagged, 24.7)
                        item = dict(reconstruction=branch, ssp=ssp, gcm=model, weather_year=int(year),
                                    scenario=scenario, cooling_transfer=transfer)
                        summary.append(dict(**item, mean_tmax_c=float(spatial_mean + raw.mean()),
                            population_mean_tmax_c=float(spatial_pop + raw.mean()),
                            area_mean_days_above=float(fa.sum()), population_mean_days_above=float(fp.sum()),
                            exposed_person_days=float(fp.sum() * pop.sum()),
                            min_daily_pixel_tmax_c=float(sorted_u[0] + raw.min()),
                            max_daily_pixel_tmax_c=float(sorted_u[-1] + raw.max())))
                        for j, date in enumerate(dates):
                            offsets.append(dict(**item, date=str(date.date()), offset_c=float(raw[j]),
                                lag01_offset_c=float(lagged[j]), area_mean_tmax_c=spatial_mean+raw[j],
                                area_pct_above_24_7=100*fa[j], population_pct_above_24_7=100*fp[j]))
                        for month in [6, 7, 8]:
                            sh = float(shift.loc[month])
                            achieved = float(spatial_mean + raw[dates.month == month].mean())
                            target = float(spatial_mean + sh)
                            assert abs(achieved-target) < 1e-10
                            if branch == 'worldclim_aligned' and scenario == 'baseline':
                                assert abs(achieved - means.loc[month]) < 1e-10
                            monthly.append(dict(**item, month=month, mean_tmax_c=achieved,
                                spatial_p05_c=quant[0]+sh, spatial_median_c=quant[1]+sh,
                                spatial_p95_c=quant[2]+sh))
            print('Exposure complete:', scenario, transfer, flush=True)
    result = pd.DataFrame(summary)
    keys = ['ssp','gcm','weather_year','scenario','cooling_transfer']
    pair = result.pivot(index=keys, columns='reconstruction', values=['mean_tmax_c','population_mean_days_above'])
    difference = pd.DataFrame(index=pair.index)
    for metric in ['mean_tmax_c','population_mean_days_above']:
        difference[metric+'_anchored_minus_aligned'] = pair[(metric,'ucm_25_5_anchored')] - pair[(metric,'worldclim_aligned')]
    # Branch offset is constant over months, weather days, scenarios and transfer.
    for (ssp, gcm), v in difference.groupby(level=['ssp','gcm']):
        expected = u0_mean - climate.query('ssp == @ssp and gcm == @gcm and month == 7').mean_tmax_c.item()
        assert np.allclose(v.mean_tmax_c_anchored_minus_aligned, expected, atol=1e-10)
    a.output.mkdir(parents=True)
    for name, frame in [('station_daily_anomalies', daily), ('weather_qc',qc), ('worldclim_monthly_means',climate),
                        ('scenario_cooling',pd.DataFrame(cooling)), ('monthly_spatial_summary',pd.DataFrame(monthly)),
                        ('daily_temperature_exposure',pd.DataFrame(offsets)), ('seasonal_comparison',result),
                        ('paired_reconstruction_difference',difference.reset_index())]:
        frame.to_csv(a.output / (name+'.csv'), index=name=='station_daily_anomalies')
    sources.append(Path(__file__))
    sources.append(repo/'code/post_processing_layers/valuation_accounting.py')
    manifest = dict(status='temperature-only diagnostic; no health calculation', common_cells=int(valid.sum()),
        ucm_population_valid_cells_before_climate=original_cells,
        population_before_climate_mask=original_population,
        population_on_common_mask=float(pop.sum()), baseline_ucm_area_mean_c=u0_mean,
        included_weather_years=sorted(map(int,daily.year.unique())), preferred_candidate='worldclim_aligned',
        threshold_c=24.7, lag='arithmetic mean of current and previous day; formed after reconstruction',
        formula_aligned='U_baseline(i) - f*(U_baseline(i)-U_scenario(i)) - mean(U_baseline) + M2050(month) + anomaly(day)',
        formula_anchored='U_baseline(i) - f*(U_baseline(i)-U_scenario(i)) + M2050(month)-M2050(July) + anomaly(day)',
        factorized_storage='Daily offsets plus source UCM rasters exactly reconstruct every pixel/day; no massive daily raster copies.',
        checks=['identical masks and population denominators','complete months and pre-June lag history',
                'zero monthly weather anomalies','WorldClim mean reproduced','direct versus cumulative threshold exposure',
                'constant analytic difference between reconstructions','non-negative cooling'],
        limitations=['station UTC hourly maxima proxy daily Tmax; one station supplies synchronous daily anomalies',
                     'monthly anomalies centred separately by observed year; future variability held historical',
                     'centred UCM spatial field and 5 C UHI are not validated daily Tmax patterns',
                     'monthly step changes possible; no seasonal ET or daily UHI modulation',
                     'common valid footprint not full boundary; population not renormalized',
                     'no mortality inputs, RR calculation, future population, or health outputs'],
        inputs=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in sources])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(result.query("scenario=='baseline' and weather_year==2021").to_string(index=False))


if __name__ == '__main__':
    main()
