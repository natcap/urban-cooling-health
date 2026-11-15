"""Versioned work-capacity candidates; production outputs are never replaced.

Main candidate: corrected fixed-condition 25/5 C temperature, RH45.
Sensitivity: equal-day/GCM/weather-year mean of daily-Tmax-based capacity.
Neither estimates average working-day productivity or worker-hours.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from osgeo import gdal
try:
    from .valuation_accounting import london_grid, sha256
except ImportError:
    from valuation_accounting import london_grid, sha256


def capacity(t, rh):
    """Unmodified InVEST simplified WBGT, then study Hothaps response."""
    t = np.asarray(t, dtype=float)
    vapour = rh / 100 * 6.105 * np.exp(17.27 * t / (237.7 + t))
    w = .567 * t + .393 * vapour + 3.94
    if np.any(w < 0): raise ValueError('WBGT outside supported positive domain')
    return .1 + .9 / (1 + (w / 30.94) ** 16.64)


def daily_mean(t, offsets, rh):
    return capacity(np.asarray(t)[:, None] + offsets[None, :], rh).mean(axis=1)


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenarios', nargs='+', default=['baseline','green30','target30'])
    p.add_argument('--rasters', action='store_true', help='Save corrected fixed RH45 change rasters')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    gdal.UseExceptions(); gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    ucm = a.data_root/'2_postprocess_intermediate/UCM_official_runs'
    mf = ucm/'diagnostics/remaining_corrected_temperatures_2026-11-12_v1/manifest.json'
    m = json.loads(mf.read_text()); paths = {k:Path(v) for k,v in m['scenario_paths'].items()}
    if not set(a.scenarios).issubset(paths): raise ValueError('Unknown scenario')
    if 'baseline' not in a.scenarios: a.scenarios.insert(0,'baseline')
    for k,v in paths.items(): assert sha256(v) == m['outputs'][k]
    of = ucm/'diagnostics/nine_scenario_health_readiness_2026-11-12_v2/shared_daily_offsets.csv'
    with of.open() as f: records = [r for r in csv.DictReader(f) if r['reconstruction']=='worldclim_aligned' and r['ssp']=='ssp370']
    assert len(records)==1380
    groups = {}
    for r in records: groups.setdefault((r['gcm'],r['weather_year']), []).append(r)
    assert len(groups)==15 and all(len(v)==92 for v in groups.values())
    assert all(len({r['date'] for r in v})==92 for v in groups.values())
    offsets=np.array([float(r['offset_c']) for r in records])
    vector=a.data_root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    boroughs,zones,counts=london_grid(vector,list(paths.values()))
    valid=zones>0; bz=zones[valid]; base=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float)
    # Keep original health reconstruction centring. Do not re-centre on this
    # broader capacity mask or separately by scenario (which removes cooling).
    lo=float(base.min()); hi=float(base.max())
    for path in paths.values():
        t=gdal.Open(str(path)).ReadAsArray()[valid];lo=min(lo,float(t.min()));hi=max(hi,float(t.max()))
    grid=np.arange(np.floor(lo*1000)/1000-.001,np.ceil(hi*1000)/1000+.002,.001)
    rng=np.random.default_rng(20261112); test=np.r_[lo,hi,rng.uniform(lo,hi,500),grid[:-1:7]+.0005]
    lookups={};checks=[]
    for rh in [35,45,55]:
        lut=np.concatenate([daily_mean(grid[i:i+256],offsets,rh) for i in range(0,len(grid),256)])
        exact=np.concatenate([daily_mean(test[i:i+256],offsets,rh) for i in range(0,len(test),256)])
        err=float(abs(np.interp(test,grid,lut)-exact).max())
        assert err*200 < 1e-4  # conservative paired-difference error in pp
        lookups[rh]=lut;checks.append({'check':'daily_lookup','rh':rh,'max_capacity_fraction_error':err,'samples':len(test)})
    sources=[mf,of,vector,vector.with_suffix('.dbf'),Path(__file__),*paths.values()]
    before={str(x):sha256(x) for x in sources}
    a.output.mkdir(parents=True); city=[]; br=[]; raster_files=[]
    prod=ucm/'production/equal_area_2026-09/03_energy_work_capacity'
    oldpaths={s:next((prod/s).glob('T_air_*.tif')) for s in a.scenarios}
    before.update({str(x):sha256(x) for x in oldpaths.values()})
    ref=gdal.Open(str(paths['baseline']))
    def old_temperature(path):
        ds=gdal.Open(str(path))
        assert (ds.RasterXSize,ds.RasterYSize)==(ref.RasterXSize,ref.RasterYSize)
        assert ds.GetGeoTransform()==ref.GetGeoTransform() and ds.GetProjection()==ref.GetProjection()
        t=ds.ReadAsArray()[valid].astype(float);nd=ds.GetRasterBand(1).GetNoDataValue()
        assert np.isfinite(t).all()
        if nd is not None:assert np.all(t!=nd)
        return t
    oldbase=old_temperature(oldpaths['baseline'])
    saved_summary=prod/'summary/citywide_energy_productivity_summary.csv'
    with saved_summary.open() as f: saved={r['scenario']:r for r in csv.DictReader(f)}
    health_summary=ucm/'diagnostics/seasonal_health_all_scenarios_2026-11-12_v1/manifest.json'
    health=json.loads(health_summary.read_text())
    # The corrected upstream manifest independently records the health footprint.
    health_cells=int(m['common_cells'])
    assert health_cells==int(health['common_cells'])==14258588
    old_capacity_paths=[next((prod/'derived_heavy_work_capacity'/s).glob('high_work_productivity_*.tif')) for s in paths]
    old_capacity_by_scenario=dict(zip(paths,old_capacity_paths))
    _,old_zones,_=london_grid(vector,old_capacity_paths)
    same_support=np.array_equal(zones,old_zones)
    before.update({str(x):sha256(x) for x in old_capacity_paths})
    del old_zones
    saved_baseline_capacity=gdal.Open(str(old_capacity_by_scenario['baseline'])).ReadAsArray()[valid].astype(float)
    before.update({str(saved_summary):sha256(saved_summary),str(health_summary):sha256(health_summary)})
    definitions=[('legacy_fixed',45),('corrected_fixed',45)]+[('daily_tmax_sensitivity',r) for r in [35,45,55]]
    base_values={('legacy_fixed',45):capacity(oldbase,45),('corrected_fixed',45):capacity(base,45)}
    for rh in lookups:base_values[('daily_tmax_sensitivity',rh)]=np.interp(base,grid,lookups[rh])
    for scenario in a.scenarios:
        t=gdal.Open(str(paths[scenario])).ReadAsArray()[valid].astype(float)
        old=old_temperature(oldpaths[scenario])
        for method,rh in definitions:
            c=capacity(old,rh) if method=='legacy_fixed' else capacity(t,rh) if method=='corrected_fixed' else np.interp(t,grid,lookups[rh])
            assert np.isfinite(c).all() and c.min()>=.1 and c.max()<=1
            change=100*(c-base_values[(method,rh)])
            if scenario=='baseline':assert np.all(change==0)
            if method!='legacy_fixed':
                assert np.all(change[t>base]<=1e-10) and np.all(change[t<base]>=-1e-10)
            total=np.bincount(bz,weights=change,minlength=34)[1:];means=total/counts
            citymean=float(change.mean());reconciled=float(np.dot(means,counts)/counts.sum())
            assert abs(citymean-reconciled)<1e-10
            if method=='legacy_fixed':
                if same_support:assert int(saved[scenario]['valid_productivity_pixels'])==int(valid.sum())
                residual=citymean-float(saved[scenario]['productivity_change_vs_baseline_pp'])
                if same_support:assert abs(residual)<1e-4
                checks.append({'check':'legacy_production_replay','scenario':scenario,'same_support':same_support,'residual_pp':residual,'tolerance_pp':.0001,'note':'Direct Float64 WBGT/Hothaps versus saved Float32 intermediates; exact common support checked'})
                saved_capacity=gdal.Open(str(old_capacity_by_scenario[scenario])).ReadAsArray()[valid].astype(float)
                matched_residual=citymean-float(100*(saved_capacity-saved_baseline_capacity).mean())
                assert abs(matched_residual)<1e-4
                checks.append({'check':'legacy_matched_support_replay','scenario':scenario,'residual_pp':matched_residual,'tolerance_pp':.0001})
            city.append(dict(scenario=scenario,method=method,rh_percent=rh,capacity_fraction=float(c.mean()),change_pp=citymean,unweighted_borough_mean_pp=float(means.mean()),valid_cells=int(valid.sum()),reconciliation_pp=citymean-reconciled))
            absolute_means=np.bincount(bz,weights=c,minlength=34)[1:]/counts
            for i,(code,name,_) in enumerate(boroughs):br.append(dict(scenario=scenario,method=method,rh_percent=rh,borough_code=code,borough_name=name,capacity_fraction=float(absolute_means[i]),change_pp=float(means[i]),valid_cells=int(counts[i])))
            if a.rasters and method=='corrected_fixed':
                rp=a.output/f'{scenario}_corrected_fixed_rh45_change_pp.tif'
                dst=gdal.GetDriverByName('GTiff').Create(str(rp),ref.RasterXSize,ref.RasterYSize,1,gdal.GDT_Float32,options=['COMPRESS=DEFLATE','TILED=YES'])
                dst.SetGeoTransform(ref.GetGeoTransform());dst.SetProjection(ref.GetProjection())
                out=np.full(valid.shape,-9999,dtype=np.float32);out[valid]=change
                dst.GetRasterBand(1).SetNoDataValue(-9999);dst.GetRasterBand(1).WriteArray(out);dst=None;raster_files.append(str(rp))
            print(scenario,method,rh,citymean,flush=True)
    after={s:sha256(Path(s)) for s in before};assert before==after
    write_csv(a.output/'citywide_capacity.csv',city);write_csv(a.output/'borough_capacity.csv',br)
    manifest={'status':'candidate; not adopted production','scenarios':a.scenarios,'capacity_mask':'all ten corrected temperatures and London borough cell centres; not health mask','valid_cells':int(valid.sum()),'recorded_health_cells':health_cells,'capacity_minus_health_cell_count':int(valid.sum())-health_cells,'grid_cell_area_m2':abs(ref.GetGeoTransform()[1]*ref.GetGeoTransform()[5]),'offset_centring':'unchanged shared health offset_c; no recentering on capacity support','daily_weighting':'equal 92 days x 3 GCM x 5 weather years','daily_label':'summer mean of daily-Tmax-based capacity; not working-day average','rh_sensitivities':'assumed 35/45/55 percent; not confidence limits','lookup_step_c':.001,'lookup_checks':checks,'input_sha256':before,'inputs_unchanged':True,'primary_rasters':raster_files,'units':'scenario-minus-baseline percentage points','runtime_formula':'WBGT=.567*T+.393*(RH/100*6.105*exp(17.27*T/(237.7+T)))+3.94; capacity=.1+.9/(1+(WBGT/30.94)^16.64)'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
