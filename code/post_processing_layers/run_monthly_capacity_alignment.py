"""Fresh June/July/August/JJA diagnostic, not a production replacement.

Uses corrected UCM spatial temperatures plus the health reconstruction's shared
daily offsets, without recentering. Daily-Tmax-based capacity at assumed RH45
is not a working-day average, worker-hour estimate or earnings measure.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from osgeo import gdal
from valuation_accounting import london_grid, sha256
from run_corrected_capacity_candidate import daily_mean, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    diag = a.data_root / '2_postprocess_intermediate/UCM_official_runs/diagnostics'
    mf = diag / 'remaining_corrected_temperatures_2026-11-12_v1/manifest.json'
    of = diag / 'nine_scenario_health_readiness_2026-11-12_v2/shared_daily_offsets.csv'
    manifest = json.loads(mf.read_text())
    paths = {s: Path(v) for s, v in manifest['scenario_paths'].items()}
    for s, path in paths.items():
        assert sha256(path) == manifest['outputs'][s]
    with of.open() as f:
        records = [r for r in csv.DictReader(f)
                   if r['reconstruction'] == 'worldclim_aligned' and r['ssp'] == 'ssp370']
    groups = {}
    for r in records:
        groups.setdefault((r['gcm'], r['weather_year']), []).append(r)
    assert len(records) == 1380 and len(groups) == 15
    assert all(len(v) == len({r['date'] for r in v}) == 92 for v in groups.values())
    vector = a.data_root / '1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    _, zones, counts = london_grid(vector, list(paths.values()))
    valid = zones > 0
    n = int(valid.sum())
    assert int(counts.sum()) == n
    scenarios = ['baseline', 'green30', 'target30']
    temperatures = {s: gdal.Open(str(paths[s])).ReadAsArray()[valid].astype(float)
                    for s in scenarios}
    lo = min(float(x.min()) for x in temperatures.values())
    hi = max(float(x.max()) for x in temperatures.values())
    grid = np.arange(np.floor(lo * 1000) / 1000 - .001,
                     np.ceil(hi * 1000) / 1000 + .002, .001)
    # Linear-interpolation weights aggregate exactly the per-pixel interpolant.
    # Equal-area cells give area weighting, not population/worker weighting.
    weights = {}
    for s, t in temperatures.items():
        k = np.searchsorted(grid, t, side='right') - 1
        f = (t - grid[k]) / (grid[k + 1] - grid[k])
        weights[s] = (np.bincount(k, weights=1-f, minlength=len(grid)) +
                      np.bincount(k+1, weights=f, minlength=len(grid))) / n
        assert abs(weights[s].sum() - 1) < 1e-10
    baseline_mean = float(temperatures['baseline'].mean())
    test = np.r_[lo, hi, np.random.default_rng(20261113).uniform(lo, hi, 400)]
    periods = [('June', '06', 30), ('July', '07', 31), ('August', '08', 31), ('JJA', None, 92)]
    rows = []
    max_error = 0.
    for (gcm, year), values in sorted(groups.items()):
        for period, month, days in periods:
            offsets = np.array([float(r['offset_c']) for r in values
                                if month is None or r['date'][5:7] == month])
            assert len(offsets) == days
            lut = np.concatenate([daily_mean(grid[i:i+256], offsets, 45)
                                  for i in range(0, len(grid), 256)])
            exact = daily_mean(test, offsets, 45)
            error = float(np.abs(np.interp(test, grid, lut)-exact).max())
            max_error = max(max_error, error)
            assert error * 200 < .0001
            means = {s: float(np.dot(weights[s], lut)) for s in scenarios}
            for s in scenarios:
                rows.append(dict(gcm=gcm, weather_year=year, period=period,
                                 scenario=s, days_per_member=days,
                                 baseline_tmax_c=baseline_mean + float(offsets.mean()),
                                 capacity_fraction=means[s],
                                 change_pp=100 * (means[s]-means['baseline']),
                                 rh_percent=45, valid_cells=n))
        print(gcm, year, 'complete', flush=True)
    # JJA is day-weighted, not an equally weighted mean of the three months.
    for key in groups:
        for s in scenarios:
            r = [x for x in rows if (x['gcm'], x['weather_year']) == key and x['scenario'] == s]
            for field in ['capacity_fraction', 'change_pp', 'baseline_tmax_c']:
                weighted = sum(x[field] * x['days_per_member'] for x in r if x['period'] != 'JJA') / 92
                assert abs(weighted-next(x[field] for x in r if x['period'] == 'JJA')) < 1e-10
    summary = []
    for period, _, days in periods:
        for s in scenarios:
            members = [x for x in rows if x['period'] == period and x['scenario'] == s]
            row = dict(period=period, scenario=s, days_per_member=days, members=len(members),
                       rh_percent=45, valid_cells=n)
            for field in ['baseline_tmax_c', 'capacity_fraction', 'change_pp']:
                x = np.array([r[field] for r in members])
                row[field] = float(x.mean())
                row[field+'_median'] = float(np.median(x))
                row[field+'_min'] = float(x.min())
                row[field+'_max'] = float(x.max())
            summary.append(row)
    inputs = [mf, of, Path(__file__), Path(__file__).with_name('run_corrected_capacity_candidate.py'),
              Path(__file__).with_name('valuation_accounting.py'),
              vector, vector.with_suffix('.dbf'), vector.with_suffix('.shx'), *paths.values()]
    hashes = {str(x): sha256(x) for x in inputs}
    a.output.mkdir(parents=True)
    write_csv(a.output/'monthly_capacity_members.csv', rows)
    write_csv(a.output/'monthly_capacity_summary.csv', summary)
    metadata = dict(status='diagnostic only; production unchanged',
                    estimand='Area-weighted mean daily-Tmax-based modelled heavy-work capacity',
                    spatial_support='Common valid footprint of all ten corrected temperatures and London borough cell centres',
                    valid_cells=n, offset_centring='Original health shared offset_c unchanged; no monthly or scenario recentering',
                    temporal_weighting='Equal days within period; equal 3 GCM x 5 weather-year members; JJA uses 30/31/31 day weights',
                    rh_percent=45, uncertainties='Member min-max describes ensemble spread, not confidence intervals',
                    caveats='Fixed UCM spatial cooling pattern and fixed RH; not monthly ET0 reruns or working-day productivity',
                    lookup_step_c=.001, max_lookup_error_fraction=max_error,
                    max_paired_change_error_bound_pp=200*max_error,
                    jja_day_weight_reconciliation='passed', input_sha256=hashes)
    (a.output/'manifest.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
