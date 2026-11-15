"""Isolated coefficient sensitivity; never overwrite production outputs.

Hold existing building temperatures fixed to isolate scaling and allocation.
JSON records are diagnostics, NOT a replacement production consumption table.
Run with the project's InVEST environment (OGR/NumPy dependencies).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from osgeo import ogr


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--regional-csv', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    root = args.data_root / '2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity'
    table = args.data_root / '1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/energy_buildings/_UCM_Energy Consumption Table.csv'
    with args.regional_csv.open(encoding='utf-8-sig') as f: regional = list(csv.DictReader(f))
    shares = {}
    source_rows = []
    for year in (2016, 2017, 2018, 2021):
        rows = {r['Code']: r for r in regional if int(r['Year']) == year}
        london, gb = rows['E12000007'], rows['K03000001']
        source_rows.extend([london, gb])
        shares[str(year)] = {c: float(london[c])/float(gb[c]) for c in ('All_consumption_GWh', 'Non_domestic_consumption_GWh')}
    scale = shares['2021']['All_consumption_GWh']
    with table.open(encoding='latin1') as f: inputs = list(csv.DictReader(f))
    old = {int(r['Type']): float(r['Consumption']) for r in inputs}
    tariff = {int(r['Type']): float(r['cost']) for r in inputs}
    cache = root/'summary/building_borough_allocation_weights.npz'
    z = np.load(cache); n = int(z['n']); inside = np.unique(z['fid'])
    keep = np.zeros(n, bool); keep[inside] = True
    area = np.zeros(n); types = np.zeros(n, int); ids = np.zeros(n)
    paths = {s: next((root/s).glob('buildings_with_stats_*.shp')) for s in ('baseline','allbuilt','treerisk','treeopp','green10','green20','green30','target10','target20','target30')}
    # Allocation cache must correspond to these exact baseline geometries.
    assert list(z['key'][:2]) == [digest(paths['baseline']), digest(paths['baseline'].with_suffix('.shx'))]
    files = [table, cache, args.regional_csv] + [p.with_suffix('.dbf') for p in paths.values()]
    before = {str(p): digest(p) for p in files}
    ogr.UseExceptions()
    d = ogr.Open(str(paths['baseline'])); l = d.GetLayer()
    assert l.GetFeatureCount() == n
    base_t = np.zeros(n); base_energy = np.zeros(n)
    for f in l:
        i = f.GetFID(); area[i] = f.GetGeometryRef().Area(); types[i] = f['type']; ids[i] = f['fid_attr']
        base_t[i] = f['mean_t_air']; base_energy[i] = f['energy_sav']
    assert np.isfinite(base_t).all() and len(np.unique(ids)) == n
    print('Baseline loaded', n, flush=True)
    # Known occupied non-residential classes. Unknown, vacant, agriculture and
    # minerals are outside this proxy; exclusion is not evidence of zero demand.
    nonres = [1,2,3,5,6,8,9,11]
    total = 350*24*30*1000*scale
    groups = [(nonres, .99), ([7], .01)]
    def coefficients(groups):
        c = {t: 0.0 for t in old}
        for ts, share in groups:
            denom = area[keep & np.isin(types, ts)].sum()
            assert denom > 0
            for t in ts: c[t] = total*share/denom
        assert np.isclose(sum(c[t]*area[keep & (types==t)].sum() for t in c), total)
        return c
    pooled = coefficients(groups)
    # Rounded source shares sum to 100.1%; normalize within non-domestic pool.
    # Broad-class crosswalk is a sensitivity assumption, not observed use.
    sectors = [([1,6], .062+.038+.010+.090+.055), ([2], .016), ([3,5], .512+.030), ([8], .188)]
    sector = coefficients([(ts, .99*w/1.001) for ts,w in sectors]+[([7],.01)])
    nonres_only = coefficients([(nonres,1.0)])
    variants = {'legacy':old, 'scale_only':{t:v*scale for t,v in old.items()}, 'pooled_candidate':pooled, 'sector_sensitivity':sector, 'pooled_no_residential':nonres_only}
    vectors = {k: np.array([c[int(t)]*tariff[int(t)] for t in types])*area for k,c in variants.items()}
    out = []
    summary = {r['scenario']:r for r in csv.DictReader((root/'summary/citywide_energy_productivity_summary.csv').open())}
    for s,p in paths.items():
        d = ogr.Open(str(p)); l = d.GetLayer()
        assert l.GetFeatureCount() == n
        l.SetIgnoredFields(['OGR_GEOMETRY'])
        temp = np.zeros(n); energy = np.zeros(n)
        for f in l:
            i=f.GetFID()
            assert f['fid_attr']==ids[i] and f['type']==types[i]
            temp[i]=f['mean_t_air']; energy[i]=f['energy_sav']
        assert np.isfinite(temp).all()
        delta=base_t-temp
        record={'scenario':s, 'gbp_per_conditional_30day_month':{k:float((v[keep]*delta[keep]).sum()) for k,v in vectors.items()}}
        direct=float((energy[keep]-base_energy[keep]).sum())
        assert np.isclose(record['gbp_per_conditional_30day_month']['legacy'],direct,rtol=1e-8,atol=.01)
        assert np.isclose(direct,float(summary[s]['energy_change_vs_baseline']),rtol=1e-8,atol=.01)
        record['legacy_replay_residual_gbp']=record['gbp_per_conditional_30day_month']['legacy']-direct
        record['by_type_area_cooling_m2_c']={str(t):float((area[keep & (types==t)]*delta[keep & (types==t)]).sum()) for t in old}
        out.append(record); print(s, record['gbp_per_conditional_30day_month'],flush=True)
    after={str(p):digest(p) for p in files}; assert before==after
    result={'status':'coefficient sensitivity only; unchanged legacy temperature contrasts', 'source_url':'https://assets.publishing.service.gov.uk/media/6942957636f089d38be1f1f8/elec_region_stacked_2005-2024.csv', 'regional_source_rows':source_rows, 'london_gb_shares':shares, 'assumed_residential_share':.01, 'london_kwh_per_degree_30days':total, 'coefficients_kwh_m2_degree_30days':variants, 'type_area_m2':{str(t):float(area[keep & (types==t)].sum()) for t in old}, 'inside_buildings':int(keep.sum()), 'input_sha256':before, 'inputs_unchanged':True, 'results':out}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(result,f,indent=2)


if __name__=='__main__': main()
