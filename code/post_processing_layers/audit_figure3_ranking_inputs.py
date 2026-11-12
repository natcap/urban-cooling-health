"""Read-only input attribution using existing rasters; no UCM executions.

Compare on one common London mask, including the existing Green30 NoData-only
diagnostic. Do not label combined scenario/production effects as pure model effects.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from osgeo import gdal
from valuation_accounting import london_grid, masked_means, sha256


def one(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'Expected unique file, found {paths}')
    return paths[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_root', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    runs = args.data_root/'2_postprocess_intermediate/UCM_official_runs'
    inputs = args.data_root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs'
    prod = runs/'production/equal_area_2026-09'
    paths = {}
    for s, folder in [('baseline','scenario0'),('green10','scenario41'),
        ('green20','scenario42'),('green30','scenario43'),('target10','scenario510'),
        ('target20','scenario520'),('target30','scenario530')]:
        paths['old_'+s] = one(f for f in (runs/'archive/legacy_runs'/folder).rglob('T_air_*.tif')
            if 'work_and_energy_runs' in str(f) and 'nomix' not in f.name
            and ('25deg_5uhi' in f.name or '25.0deg_5.0uhi' in f.name))
        stage = '03_energy_work_capacity' if s=='baseline' else '01_temperature_invest3202'
        paths['new_'+s] = one((prod/stage/s).glob('T_air_*25deg_5uhi*.tif'))
    for name, folder in [('diag_baseline','baseline_health_invest3202'),
        ('diag_green30_original','green30_health_invest3202'),
        ('diag_green30_nodata','green30_health_invest3202_nodata_harmonized')]:
        paths[name] = one((runs/'archive/diagnostic_runs'/folder).glob('T_air_*25deg_5uhi*.tif'))
    rows, zones, counts = london_grid(inputs/'AOIs/London_Borough_aoi.shp', list(paths.values()))
    means = {name: masked_means(path,zones,counts) for name,path in paths.items()}
    result = []
    for name,(city,local) in means.items():
        if name.startswith('old_'): ref = 'old_baseline'
        elif name.startswith('new_'): ref = 'new_baseline'
        else: ref = 'diag_baseline'
        delta = local-means[ref][1]
        result.append(dict(stage=name,city_change=city-means[ref][0],
            borough_change=float(delta.mean()),common_london_cells=int(counts.sum())))
    # Whole-grid canopy transitions: retain original source classes explicitly.
    lulc = inputs/'LULC'
    basepath = lulc/'LCM2023_London_10m_clip2aoi_tcc24.tif'
    base_ds = gdal.Open(str(basepath)); base = base_ds.ReadAsArray()
    canopy_rows = []
    source_paths = {'baseline_lulc':basepath}
    for tier in [10,20,30]:
        for family in ['green','target']:
            if family=='green':
                old = lulc/f'LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_{tier}prc_canopy_increase.tif'
                new = lulc/f'lc_green_scenarios_output/revised_v2_invest_3.20.2_2026-09-10/Green{tier}_equal_area_eligible_v2.tif'
            else:
                old = lulc/f'lc_tree_equity_scenarios_output/LULC_Scenario5{tier}.tif'
                new = lulc/f'lc_tree_equity_scenarios_output/revised_v2_rank_fid_equal_area_2026-09-10/LULC_Target{tier}_equal_area_eligible_v2.tif'
            arrays = {}
            for version,path in [('old',old),('new',new)]:
                d = gdal.Open(str(path)); a = d.ReadAsArray()
                assert d.GetGeoTransform()==base_ds.GetGeoTransform() and a.shape==base.shape
                source_paths[f'{version}_{family}{tier}_lulc']=path
                valid = (base!=base_ds.GetRasterBand(1).GetNoDataValue())
                conversion = valid & (base!=100) & (a==100)
                eligible = conversion & np.isin(base,[4,20,21]); arrays[version]=eligible
                codes, nums = np.unique(base[conversion],return_counts=True)
                r = dict(scenario=f'{family}{tier}',version=version,nodata=d.GetRasterBand(1).GetNoDataValue(),
                    eligible_cells=int(eligible.sum()),other_to_100=int((conversion & ~eligible).sum()))
                r.update({f'from_{c}':int(nums[codes==c][0]) if c in codes else 0 for c in [1,2,4,20,21]})
                canopy_rows.append(r)
            retained = int((arrays['old'] & arrays['new']).sum())
            for r in canopy_rows[-2:]:
                r['eligible_overlap']=retained
                r['old_removed']=int((arrays['old'] & ~arrays['new']).sum())
                r['new_added']=int((arrays['new'] & ~arrays['old']).sum())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Confirm that the NoData diagnostic really changed only missing-value encoding.
    manifests = {}
    for key in ['diag_green30_original','diag_green30_nodata','new_green30']:
        run = paths[key].with_name(paths[key].name.replace('T_air_','run_manifest_')).with_suffix('.json')
        manifests[key] = json.loads(run.read_text())
    omit = {'lulc_raster_path','workspace_dir','results_suffix'}
    settings = [{k:v for k,v in m['model_arguments'].items() if k not in omit}
                for m in manifests.values()]
    assert all(s==settings[0] for s in settings)
    assert all(m['software']==manifests['new_green30']['software'] for m in manifests.values())
    # Companion folders may have moved; historical paths remain compatibility links.
    original = gdal.Open(manifests['diag_green30_original']['model_arguments']['lulc_raster_path'])
    harmonized = gdal.Open(manifests['diag_green30_nodata']['model_arguments']['lulc_raster_path'])
    a,b = original.ReadAsArray(),harmonized.ReadAsArray()
    va = a!=original.GetRasterBand(1).GetNoDataValue()
    vb = b!=harmonized.GetRasterBand(1).GetNoDataValue()
    assert np.array_equal(va,vb) and np.array_equal(a[va],b[vb])
    for m in manifests.values():
        for item in m['inputs']:
            assert sha256(item['path'])==item['sha256'], item['path']
    stem = 'figure3_ranking_input_audit_20261111'
    for suffix,data in [('temperatures',result),('canopy',canopy_rows)]:
        with (args.output_dir/f'{stem}_{suffix}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    manifest = dict(script_sha256=sha256(__file__),common_cells=int(counts.sum()),
        temperature_sources={k:dict(path=str(v),sha256=sha256(v)) for k,v in paths.items()},
        canopy_sources={k:dict(path=str(v),sha256=sha256(v)) for k,v in source_paths.items()},
        grid='All temperature rasters identical grid verified by london_grid; no reprojection or resampling',
        nodata_control='Verified same software and all non-LULC model arguments; identical valid land-cover values and masks; input checksums match run manifests',
        caveat='No new model run. Historical software headers do not establish actual runtime version.')
    (args.output_dir/f'{stem}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
