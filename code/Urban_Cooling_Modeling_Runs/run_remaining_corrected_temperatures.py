"""Isolated seven-scenario temperature correction; no production writes.

Usage: InVEST Python run_remaining_corrected_temperatures.py DATA_ROOT NEW_OUTPUT
Preserves physical parameters; removes the NoData lookup row in private inputs.
"""
import argparse,csv,json,logging,sys
from pathlib import Path
import numpy as np
from osgeo import gdal
import natcap.invest
import natcap.invest.urban_cooling_model as ucm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'post_processing_layers'))
from valuation_accounting import london_grid,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    assert natcap.invest.__version__=='3.20.2'
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    repo=Path(__file__).resolve().parents[2];cfg=json.loads((repo/'code/health_assessment/health-analysis-revised-equal-area.example.json').read_text())
    diag=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    basework=diag/'nodata_correction_2026-11-12/baseline_nodata0'
    baseline_args=json.loads((basework/'arguments.json').read_text())
    a.output.mkdir(parents=True);logging.basicConfig(filename=a.output/'run.log',level=logging.INFO)
    paths={'baseline':basework/'T_air_test.tif','green30':diag/'nodata_correction_2026-11-12/green30_nodata0/T_air_test.tif','target30':diag/'nodata_target30_2026-11-12/target30_nodata0/T_air_test.tif'}
    records=[];hashes={str(v):sha256(v) for v in paths.values()}
    for scenario in ['allbuilt','treerisk','treeopp','green10','green20','target10','target20']:
        original=a.root/cfg['scenarios'][scenario+'_25c']['t_scenario']
        manifests=list(original.parent.glob('run_manifest_*25deg_5uhi*.json'));assert len(manifests)==1
        args=json.loads(manifests[0].read_text())['model_arguments']
        # Compare every physical/shared argument; workspace and optional valuations
        # legitimately differ, but climate/biophysical inputs must match baseline.
        exempt={'workspace_dir','results_suffix','lulc_raster_path','biophysical_table_path','n_workers','do_energy_valuation','do_productivity_valuation','building_vector_path','energy_consumption_table_path'}
        differences={k:[baseline_args.get(k),args.get(k)] for k in set(args)|set(baseline_args) if k not in exempt and baseline_args.get(k)!=args.get(k)}
        assert not differences,differences
        source=Path(args['lulc_raster_path']);table=Path(args['biophysical_table_path'])
        for f in [source,table,original,manifests[0]]:hashes[str(f)]=sha256(f)
        ds=gdal.Open(str(source));nd=ds.GetRasterBand(1).GetNoDataValue();assert nd==0
        work=a.output/f'{scenario}_nodata0';work.mkdir()
        copy=gdal.GetDriverByName('GTiff').CreateCopy(str(work/'input.tif'),ds,options=['COMPRESS=LZW']);copy=None
        with table.open(encoding='utf-8-sig') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
        kept=[r for r in rows if float(r['lucode'])!=nd];assert len(kept)==len(rows)-1
        with (basework/'biophysical.csv').open() as f:assert kept==list(csv.DictReader(f)),'Physical lookup differs from corrected baseline'
        with (work/'biophysical.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(kept)
        run=args.copy();run.update(workspace_dir=str(work),results_suffix='test',lulc_raster_path=str(work/'input.tif'),biophysical_table_path=str(work/'biophysical.csv'),n_workers=-1,do_energy_valuation=False,do_productivity_valuation=False)
        (work/'arguments.json').write_text(json.dumps(run,indent=2)+'\n')
        print('RUN',scenario,flush=True);ucm.execute(run);paths[scenario]=work/'T_air_test.tif'
        records.append(dict(scenario=scenario,output=str(paths[scenario]),production=str(original),physical_parameters_match_baseline=True,removed_lookup_code=nd))
        (a.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n');print('DONE',scenario,flush=True)
    poppath=a.root/cfg['population_2021'];boundary=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    _,zones,counts=london_grid(boundary,list(paths.values())+[poppath]);valid=zones>0
    _,basezones,_=london_grid(boundary,[paths['baseline'],poppath]);assert np.array_equal(valid,basezones>0)
    assert valid.sum()==14258588
    b=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float);pop=gdal.Open(str(poppath)).ReadAsArray()[valid].astype(float)
    rows=[]
    oldbase=a.root/cfg['scenarios']['green30_25c']['t_baseline'];ob=gdal.Open(str(oldbase)).ReadAsArray()[valid].astype(float)
    for scenario,path in paths.items():
        s=gdal.Open(str(path)).ReadAsArray()[valid].astype(float);cool=b-s
        old=oldbase if scenario=='baseline' else a.root/cfg['scenarios'][scenario+'_25c']['t_scenario'];os=gdal.Open(str(old)).ReadAsArray()[valid].astype(float)
        rows.append(dict(scenario=scenario,area_temperature_c=float(s.mean()),area_cooling_c=float(cool.mean()),population_cooling_c=float(cool@pop/pop.sum()),old_area_cooling_c=float((ob-os).mean()),warming_cells=int((cool<0).sum()),cooling_cells=int((cool>0).sum()),max_abs_temperature_change_from_production_c=float(abs(s-os).max())))
    with (a.output/'temperature_comparison.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for path,h in hashes.items():assert sha256(path)==h,path
    manifest=dict(status='seven temperature-only runs completed; not adopted production',invest_version=natcap.invest.__version__,common_cells=int(valid.sum()),population=float(pop.sum()),corrected_baseline_mean_c=float(b.mean()),runs=records,scenario_paths={k:str(v) for k,v in paths.items()},preserved_sources=hashes,outputs={k:sha256(v) for k,v in paths.items()})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
