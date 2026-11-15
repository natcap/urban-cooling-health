"""Versioned conditional energy candidates on corrected building temperatures.

Use the InVEST environment. No production writes. Same polygon zonal method as
InVEST; this is not a centroid or health-mask approximation. Absolute values
are model counterfactual values relative to 30 C, NOT electricity consumption.
Half transfer applies to scenario cooling relative to baseline, not baseline T.
"""
import argparse, csv, hashlib, json, logging
from pathlib import Path
import numpy as np
from osgeo import gdal, ogr
import pygeoprocessing
from valuation_accounting import boroughs


def numeric_dbf(path):
    """Read fixed-width numeric DBF columns without millions of OGR calls."""
    raw=path.read_bytes(); n=int.from_bytes(raw[4:8],'little'); header=int.from_bytes(raw[8:10],'little'); stride=int.from_bytes(raw[10:12],'little')
    assert len(raw)>=header+n*stride
    deletion=np.ndarray((n,),dtype='S1',buffer=raw,offset=header,strides=(stride,));assert (deletion==b' ').all()
    offset=1; result={}
    for start in range(32,header-1,32):
        field=raw[start:start+32];name=field[:11].split(b'\0')[0].decode();width=field[16]
        if name in ('fid_attr','type','mean_t_air','energy_sav'):
            result[name]=np.ndarray((n,),dtype=f'S{width}',buffer=raw,offset=header+offset,strides=(stride,)).astype(float)
        offset+=width
    return result


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def write_csv(p, rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    a=argparse.ArgumentParser(); a.add_argument('--data-root',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True); a.add_argument('--stage',choices=['pilot','pilot_verified','allnine'],default='pilot')
    a=a.parse_args(); gdal.UseExceptions(); ogr.UseExceptions(); gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    logging.basicConfig(level=logging.INFO)
    repo=Path(__file__).resolve().parents[2]
    stage=a.output/a.stage
    if stage.exists(): raise FileExistsError(stage)
    stage.mkdir(parents=True); cache=a.output/'cache'; cache.mkdir(exist_ok=True)
    runs=a.data_root/'2_postprocess_intermediate/UCM_official_runs'
    production=runs/'production/equal_area_2026-09/03_energy_work_capacity'
    manifest_path=runs/'diagnostics/remaining_corrected_temperatures_2026-11-12_v1/manifest.json'
    m=json.loads(manifest_path.read_text()); paths={s:Path(p) for s,p in m['scenario_paths'].items()}
    scenarios=list(paths) if a.stage=='allnine' else ['baseline','green30','target30']
    boundary=a.data_root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    vector=next((production/'baseline').glob('buildings_with_stats*.shp'))
    allocation=production/'summary/building_borough_allocation_weights.npz'
    previous_path=repo/'data/diagnostics/energy_scaling_20261112_v01/comparison.json'
    previous=json.loads(previous_path.read_text())
    table=a.data_root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/energy_buildings/_UCM_Energy Consumption Table.csv'
    source_paths=[manifest_path,previous_path,table,allocation,vector,vector.with_suffix('.shx'),vector.with_suffix('.dbf'),boundary,boundary.with_suffix('.dbf')]+[paths[s] for s in scenarios]
    old_dbfs={s:next((production/s).glob('buildings_with_stats*.dbf')) for s in scenarios}
    source_paths+=list(old_dbfs.values())
    before={str(p):sha(p) for p in source_paths}
    for s in scenarios: assert before[str(paths[s])]==m['outputs'][s]
    reference=gdal.Open(str(paths['baseline']));ref_array=reference.ReadAsArray();ref_nodata=reference.GetRasterBand(1).GetNoDataValue()
    ref_valid=np.isfinite(ref_array)&(ref_array!=ref_nodata)
    for s in scenarios:
        raster=gdal.Open(str(paths[s]));arr=raster.ReadAsArray();nd=raster.GetRasterBand(1).GetNoDataValue()
        assert raster.GetGeoTransform()==reference.GetGeoTransform() and raster.GetProjection()==reference.GetProjection()
        assert np.array_equal(np.isfinite(arr)&(arr!=nd),ref_valid),f'Pixel support differs: {s}; paired raster masking required'
    del ref_array,arr,ref_valid
    z=np.load(allocation); n=int(z['n']); fid=z['fid']; zone=z['zone']; weight=z['weight']
    assert list(z['key'])==[sha(vector),sha(vector.with_suffix('.shx')),sha(boundary),sha(boundary.with_suffix('.dbf'))]
    inside=np.zeros(n,bool);inside[np.unique(fid)]=True
    attributes=cache/'attributes.npz'
    geometry_key=sha(vector)+sha(vector.with_suffix('.shx'))
    cache_key=cache/'geometry_key.json'
    if cache_key.exists():assert json.loads(cache_key.read_text())['key']==geometry_key
    elif attributes.exists():
        pilot_manifest=a.output/'pilot/manifest.json'
        if not pilot_manifest.exists():raise RuntimeError('Unverified pre-existing geometry cache')
        proof=json.loads(pilot_manifest.read_text())['input_sha256']
        assert proof[str(vector)]+proof[str(vector.with_suffix('.shx'))]==geometry_key
        cache_key.write_text(json.dumps({'key':geometry_key}))
    else:cache_key.write_text(json.dumps({'key':geometry_key}))
    if not attributes.exists():
        ids=np.zeros(n);types=np.zeros(n,int);area=np.zeros(n);old_t=np.zeros(n);old_energy=np.zeros(n)
        d=ogr.Open(str(vector));l=d.GetLayer();assert l.GetFeatureCount()==n
        indices={k:l.GetLayerDefn().GetFieldIndex(k) for k in ['fid_attr','type','mean_t_air','energy_sav']}
        for f in l:
            i=f.GetFID();ids[i]=f.GetField(indices['fid_attr']);types[i]=f.GetField(indices['type'])
            area[i]=f.GetGeometryRef().Area();old_t[i]=f.GetField(indices['mean_t_air']);old_energy[i]=f.GetField(indices['energy_sav'])
        assert len(np.unique(ids))==n
        np.savez_compressed(attributes,ids=ids,types=types,area=area,old_t=old_t,old_energy=old_energy,source_hash=sha(vector.with_suffix('.dbf')))
    att=np.load(attributes);assert str(att['source_hash'])==sha(vector.with_suffix('.dbf'))
    # Cached extraction geometry is bound to the completed pilot provenance.
    # A new cache is created from the geometry verified against allocation above.
    if (a.output/'pilot/manifest.json').exists():
        provenance=json.loads((a.output/'pilot/manifest.json').read_text())['input_sha256']
        for p in (vector,vector.with_suffix('.shx'),vector.with_suffix('.dbf')): assert provenance[str(p)]==before[str(p)]
    area=att['area'];types=att['types']
    old_raster=next((production/'baseline').glob('T_air_london*.tif'))
    extract_paths={s:paths[s] for s in scenarios};extract_paths['legacy_baseline_replay']=old_raster
    # Batching shares expensive polygon grouping while retaining overlap handling.
    missing=[s for s in extract_paths if not (cache/f'{s}_temperature.npz').exists()]
    if missing:
        print('Extracting building temperatures',missing,flush=True)
        stats=pygeoprocessing.zonal_statistics([(str(extract_paths[s]),1) for s in missing],str(vector),polygons_might_overlap=True,working_dir=str(cache))
        for s,ss in zip(missing,stats):
            means=np.full(n,np.nan);counts=np.zeros(n,int);nodata=np.zeros(n,int)
            for i,r in ss.items():
                counts[i]=r['count'];nodata[i]=r['nodata_count']
                if r['count']>0: means[i]=r['sum']/r['count']
            np.savez_compressed(cache/f'{s}_temperature.npz',mean=means,count=counts,nodata=nodata,source_hash=sha(extract_paths[s]))
        del stats
    extracts={s:np.load(cache/f'{s}_temperature.npz') for s in extract_paths}
    for s,e in extracts.items(): assert str(e['source_hash'])==sha(extract_paths[s])
    replay=extracts['legacy_baseline_replay']['mean']; replay_error=float(np.nanmax(abs(replay-att['old_t'])))
    assert replay_error<1e-4, replay_error
    valid=inside.copy()
    for s in scenarios:valid &= np.isfinite(extracts[s]['mean'])
    assert valid.sum()>0
    with table.open(encoding='latin1') as f: rows=list(csv.DictReader(f))
    old={int(r['Type']):float(r['Consumption']) for r in rows};tariff={int(r['Type']):float(r['cost']) for r in rows}
    budget=previous['london_kwh_per_degree_30days']; share=previous['london_gb_shares']['2021']['All_consumption_GWh']
    # Recompute denominators on common support, and explicitly report the support
    # change. Otherwise a corrected mask could silently discard assigned budget.
    def pool(groups):
        c={t:0. for t in old}
        for ts,w in groups:
            denom=area[valid & np.isin(types,ts)].sum();assert denom>0
            for t in ts:c[t]=budget*w/denom
        assert np.isclose(sum(c[t]*area[valid & (types==t)].sum() for t in c),budget)
        return c
    nonres=[1,2,3,5,6,8,9,11]
    coefficients={'legacy':old,'scale_only':{t:v*share for t,v in old.items()},'pooled_candidate':pool([(nonres,.99),([7],.01)]),'sector_sensitivity':pool([([1,6],.99*.255/1.001),([2],.99*.016/1.001),([3,5],.99*.542/1.001),([8],.99*.188/1.001),([7],.01)])}
    price=np.array([tariff[int(t)] for t in types]); base=extracts['baseline']['mean']
    bnames,_=boroughs(boundary);city=[];borough=[];max_residual=0
    for s in scenarios:
        cooling=base-extracts[s]['mean']
        legacy=numeric_dbf(old_dbfs[s]);assert np.array_equal(legacy['fid_attr'],att['ids']) and np.array_equal(legacy['type'],types)
        legacy_cooling=att['old_t']-legacy['mean_t_air']
        for variant,c in coefficients.items():
            factor=np.array([c[int(t)] for t in types])*area
            for transfer in (1.,.5):
                diff=np.where(valid,factor*cooling*transfer,0.)
                absolute=np.where(valid,factor*(30-base+cooling*transfer),0.)
                record=dict(scenario=s,variant=variant,cooling_transfer=transfer,period='conditional_30_day_warm_period',energy_change_kwh=float(diff.sum()),energy_change_gbp=float((diff*price).sum()),absolute_counterfactual_kwh=float(absolute.sum()),absolute_counterfactual_gbp=float((absolute*price).sum()),valid_buildings=int(valid.sum()))
                old_change=np.where(valid,factor*legacy_cooling*transfer,0.)
                record['same_support_legacy_temperature_change_gbp']=float((old_change*price).sum())
                record['temperature_correction_effect_gbp']=record['energy_change_gbp']-record['same_support_legacy_temperature_change_gbp']
                city.append(record)
                sums={k:np.bincount(zone,weights=v[fid]*weight,minlength=33) for k,v in [('energy_change_kwh',diff),('energy_change_gbp',diff*price),('absolute_counterfactual_kwh',absolute),('absolute_counterfactual_gbp',absolute*price)]}
                for k,v in sums.items():
                    err=abs(float(v.sum())-record[k]);max_residual=max(max_residual,err);assert np.isclose(v.sum(),record[k],rtol=1e-10,atol=.001)
                for j,(code,name,_) in enumerate(bnames):borough.append(dict(scenario=s,variant=variant,cooling_transfer=transfer,borough_code=code,borough=name,period=record['period'],**{k:float(v[j]) for k,v in sums.items()}))
                if s=='baseline':assert record['energy_change_gbp']==0
                if s in ['allbuilt','treerisk']:assert record['energy_change_gbp']<0
        print('Summarized',s,flush=True)
    for r in city:
        if r['cooling_transfer']==.5:
            full=next(v for v in city if v['scenario']==r['scenario'] and v['variant']==r['variant'] and v['cooling_transfer']==1)
            assert np.isclose(r['energy_change_gbp'],full['energy_change_gbp']/2)
    write_csv(stage/'citywide_energy_candidates.csv',city);write_csv(stage/'borough_energy_candidates.csv',borough)
    coverage={s:dict(valid_buildings=int((inside & np.isfinite(extracts[s]['mean'])).sum()),missing_buildings=int((inside & ~np.isfinite(extracts[s]['mean'])).sum()),missing_footprint_m2=float(area[inside & ~np.isfinite(extracts[s]['mean'])].sum()),partially_nodata_buildings=int((inside & (extracts[s]['count']>0) & (extracts[s]['nodata']>0)).sum()),partially_nodata_footprint_m2=float(area[inside & (extracts[s]['count']>0) & (extracts[s]['nodata']>0)].sum())) for s in scenarios}
    after={str(p):sha(p) for p in source_paths};assert before==after
    manifest=dict(status='candidate only; not adopted production',scenarios=scenarios,common_buildings=int(valid.sum()),london_buildings=int(inside.sum()),excluded_common_buildings=int((inside & ~valid).sum()),excluded_common_area_m2=float(area[inside & ~valid].sum()),coverage=coverage,legacy_temperature_replay_max_error_c=replay_error,max_borough_reconciliation_residual=max_residual,input_sha256=before,inputs_unchanged=True,coefficients=coefficients,coefficient_budget_kwh_per_degree=budget,script_sha256=sha(Path(__file__)),notes=['Same InVEST polygon-centre/bbox-fallback extraction; shared valid building support.','Absolute counterfactual values are model 30C-reference avoided energy, not actual energy consumption.','Half transfer is assumed scenario-minus-baseline cooling transfer, not an empirical bound.','Daily-mean exposure transfer unresolved; conditional 30-day candidate only.'])
    (stage/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k in ['common_buildings','excluded_common_buildings','legacy_temperature_replay_max_error_c','max_borough_reconciliation_residual','coverage']},indent=2),flush=True)


if __name__=='__main__':main()
