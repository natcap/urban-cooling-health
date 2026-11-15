"""Check existing nine-scenario coverage; prepare shared corrected-baseline offsets.

Saves a NEW diagnostic folder only. Existing production fields are used as one
consistent family for numerical sample tests, NOT adopted as corrected inputs.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'health_assessment'))
from signed_health_response import signed_sum
from pilot_spatial_reconstructed_health import integrate,london_grid,sha256
from test_temperature_reconstructions import weather

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    repo=Path(__file__).resolve().parents[2];cfgpath=repo/'code/health_assessment/health-analysis-revised-equal-area.example.json';cfg=json.loads(cfgpath.read_text())
    diag=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    coverage=diag/'climate_coverage_consistency_2026-11-12_v1'
    paths={k.replace('_25c',''):a.root/v['t_scenario'] for k,v in cfg['scenarios'].items() if v['temperature_setting_c']==25}
    assert len(paths)==9
    paths={'baseline':a.root/cfg['scenarios']['green30_25c']['t_baseline'],**paths}
    poppath=a.root/cfg['population_2021'];vector=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    mortality={c['name']:a.root/c['mortality_raster'] for c in cfg['causes']}
    corrected=diag/'nodata_correction_2026-11-12/baseline_nodata0/T_air_test.tif'
    sources=list(paths.values())+list(mortality.values())+[poppath,corrected,cfgpath,Path(__file__),repo/'code/health_assessment/signed_health_response.py']
    registry=[]
    for name,p in paths.items():
        manifests=list(p.parent.glob('run_manifest_*25deg_5uhi*.json'));assert len(manifests)==1
        arg=json.loads(manifests[0].read_text())['model_arguments'];table=Path(arg['biophysical_table_path']);lc=Path(arg['lulc_raster_path'])
        ds=gdal.Open(str(lc));nd=ds.GetRasterBand(1).GetNoDataValue()
        with table.open(encoding='utf-8-sig') as f:collision=any(float(r['lucode'])==nd for r in csv.DictReader(f))
        registry.append(dict(scenario=name,path=str(p),nodata_table_collision=collision,corrected_candidate_available=name in ['baseline','green30','target30']))
        sources += [manifests[0],table,lc]
    before={str(p):sha256(p) for p in set(sources)}
    _,zones,_=london_grid(vector,list(paths.values())+list(mortality.values())+[poppath,corrected]);valid=zones>0
    _,z0,_=london_grid(vector,[corrected,poppath]);assert np.array_equal(valid,z0>0),'Nine-scenario footprint loss'
    pop=gdal.Open(str(poppath)).ReadAsArray()[valid].astype(float)
    base=gdal.Open(str(corrected)).ReadAsArray()[valid].astype(float);meanbase=float(base.mean())
    ref=gdal.Open(str(corrected));gt=ref.GetGeoTransform();bounds=[gt[0],gt[3]+ref.RasterYSize*gt[5],gt[0]+ref.RasterXSize*gt[1],gt[3]]
    cl=[]
    for f in sorted((coverage/'buffered_climate').glob('*.tif')):
        name=f.name;model=name.split('_tmax_')[1].split('_ssp')[0];ssp='ssp'+name.split('_ssp')[1].split('_')[0];sources.append(f)
        for month in [5,6,7,8]:
            band=gdal.Translate('',str(f),format='MEM',bandList=[month]);w=gdal.Warp('',band,format='MEM',dstSRS=ref.GetProjection(),outputBounds=bounds,width=ref.RasterXSize,height=ref.RasterYSize,resampleAlg='bilinear',dstNodata=-9999,outputType=gdal.GDT_Float32)
            x=w.ReadAsArray()[valid];assert np.isfinite(x).all() and (x!=-9999).all()
            cl.append(dict(ssp=ssp,gcm=model,month=month,mean_tmax_c=float(x.mean(dtype=float))))
    climate=pd.DataFrame(cl);assert len(climate)==24
    temp=diag/'temperature_reconstructions_2050_2026-11-12_v1';tm=json.loads((temp/'manifest.json').read_text())
    weatherpaths=[Path(x['path']) for x in tm['inputs'] if '/hourly/00697/' in x['path']];sources+=weatherpaths
    daily,qc=weather(weatherpaths);records=[]
    for (ssp,model),clim in climate.groupby(['ssp','gcm']):
        means=clim.set_index('month').mean_tmax_c
        for branch in ['worldclim_aligned','ucm_25_5_anchored']:
            shift=means-(meanbase if branch=='worldclim_aligned' else means.loc[7])
            for year,d in daily.groupby('year'):
                raw=d.anomaly_c+d.month.map(shift);lag=(raw+raw.shift(1))/2;ix=d.month.isin([6,7,8])
                assert len(raw[ix])==92 and lag[ix].notna().all()
                if branch=='worldclim_aligned':
                    for month in [6,7,8]:assert abs(meanbase+raw[d.month==month].mean()-means.loc[month])<1e-10
                for date in raw[ix].index:records.append(dict(reconstruction=branch,ssp=ssp,gcm=model,weather_year=int(year),date=str(date.date()),offset_c=float(raw.loc[date]),lag01_offset_c=float(lag.loc[date])))
    offsets=pd.DataFrame(records);assert len(offsets)==5520 and not offsets.duplicated(['reconstruction','ssp','gcm','weather_year','date']).any()
    # One shared offset series serves every scenario; never centre scenarios
    # separately, which would erase their citywide cooling/warming contrast.
    rng=np.random.default_rng(20261112);sample=rng.choice(len(base),1500,replace=False)
    bprod=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float)
    sampledeaths={n:gdal.Open(str(p)).ReadAsArray()[valid][sample].astype(float) for n,p in mortality.items()}
    cand=repo/'docs/manuscript/manuscript_health_response_candidates.csv';rrs=pd.read_csv(cand).set_index('endpoint');sources.append(cand)
    checks=[];stats=[]
    testoffset=np.r_[-50.,50.,offsets.lag01_offset_c.to_numpy()[::50]-24.7]
    for name,p in paths.items():
        scenario=gdal.Open(str(p)).ReadAsArray()[valid].astype(float);c=bprod-scenario
        stats.append(dict(scenario=name,cells=len(c),warming_cells=int((c<0).sum()),cooling_cells=int((c>0).sum()),area_cooling_c=float(c.mean()),population_cooling_c=float(c@pop/pop.sum())))
        for cause,weights in sampledeaths.items():
            for col in ['rr_per_1c','ci95_lower','ci95_upper']:
                rr=float(rrs.loc[cause,col]);b=bprod[sample];cs=c[sample]
                fast=signed_sum(b,cs,weights,rr,testoffset)
                direct=np.array([np.sum(weights*-np.expm1(np.log(rr)*(np.maximum(b-cs+x,0)-np.maximum(b+x,0)))) for x in testoffset])
                err=float(abs(fast-direct).max());assert err<1e-8
                if name=='baseline':assert (fast==0).all()
                if name in ['allbuilt','treerisk']:assert fast.min()<0,'Warming harm missing'
                checks.append(dict(scenario=name,cause=cause,coefficient=col,max_error=err,min_signed_sample=fast.min(),max_signed_sample=fast.max()))
        print('Signed sample verified',name,flush=True)
    # Synthetic mixed/crossing/equality cases plus regression to cooling helper.
    b=np.r_[rng.uniform(20,35,2000),24.7,24.7,25.7];c=np.r_[rng.uniform(-4,4,2000),0.,-1.,1.];d=rng.uniform(0,3,len(b));o=np.array([-50.,-24.7,-23.7,0.,30.])
    maxerr=0.
    for rr in rrs[['rr_per_1c','ci95_lower','ci95_upper']].to_numpy().ravel():
        got=signed_sum(b,c,d,rr,o);expected=np.array([np.sum(d*-np.expm1(np.log(rr)*(np.maximum(b-c+x,0)-np.maximum(b+x,0)))) for x in o]);maxerr=max(maxerr,float(abs(got-expected).max()))
        pos=abs(c);legacy=integrate(b,pos,d,rr,o,np.argsort(b),np.argsort(b-pos));assert np.allclose(signed_sum(b,pos,d,rr,o),legacy,atol=1e-8,rtol=0)
    assert maxerr<1e-8
    a.output.mkdir(parents=True)
    offsets.to_csv(a.output/'shared_daily_offsets.csv',index=False);climate.to_csv(a.output/'worldclim_monthly_means.csv',index=False)
    pd.DataFrame(registry).to_csv(a.output/'scenario_input_registry.csv',index=False);pd.DataFrame(stats).to_csv(a.output/'production_family_temperature_checks.csv',index=False);pd.DataFrame(checks).to_csv(a.output/'signed_sample_checks.csv',index=False)
    for p,h in before.items():assert sha256(p)==h,p
    manifest=dict(status='diagnostic implementation complete; corrected remaining UCM scenarios required before health adoption',common_cells=int(valid.sum()),population=float(pop.sum()),corrected_baseline_area_mean_c=meanbase,offset_rows=len(offsets),offsets_shared_by=['baseline',*list(paths)[1:]],offset_baseline=str(corrected),sample_family='existing production baseline and production scenarios; never mixed with corrected baseline',sample_pixels=1500,synthetic_max_error=maxerr,scenario_registry=registry,sources=[dict(path=str(p),sha256=sha256(p)) for p in sorted(set(sources))])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print({k:manifest[k] for k in ['common_cells','population','corrected_baseline_area_mean_c','offset_rows','synthetic_max_error']})

if __name__=='__main__':main()
