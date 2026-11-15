"""Read-only input audit; save buffered climate candidates in a NEW diagnostic folder.

Usage: InVEST Python check_health_coverage_consistency.py DATA_ROOT NEW_OUTPUT
No health estimates or production inputs are replaced. Remote native-grid subsets
retain surrounding climate cells so interpolation precedes the London mask.
"""
import argparse
import calendar
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
from test_temperature_reconstructions import weather
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'health_assessment'))
from pilot_spatial_reconstructed_health import integrate,london_grid,sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN','EMPTY_DIR');gdal.SetConfigOption('GDAL_HTTP_TIMEOUT','60')
    repo=Path(__file__).resolve().parents[2]
    diag=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    temp=diag/'temperature_reconstructions_2050_2026-11-12_v1'
    tm=json.loads((temp/'manifest.json').read_text())
    for x in tm['inputs']:assert sha256(x['path'])==x['sha256'],x['path']
    cfgpath=repo/'code/health_assessment/health-analysis-revised-equal-area.example.json'
    cfg=json.loads(cfgpath.read_text())
    rasters=[diag/('nodata_target30_2026-11-12' if s=='target30' else 'nodata_correction_2026-11-12')/f'{s}_nodata0/T_air_test.tif' for s in ['baseline','green30','target30']]
    poppath=a.root/cfg['population_2021'];boundary=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    _,zones,_=london_grid(boundary,rasters+[poppath]);valid=zones>0
    pop=gdal.Open(str(poppath)).ReadAsArray();ref=gdal.Open(str(rasters[0]));gt=ref.GetGeoTransform()
    bounds=[gt[0],gt[3]+ref.RasterYSize*gt[5],gt[0]+ref.RasterXSize*gt[1],gt[3]]
    def warp(path,month):
        band=gdal.Translate('',str(path),format='MEM',bandList=[month])
        w=gdal.Warp('',band,format='MEM',dstSRS=ref.GetProjection(),outputBounds=bounds,width=ref.RasterXSize,height=ref.RasterYSize,resampleAlg='bilinear',dstNodata=-9999,outputType=gdal.GDT_Float32)
        return w.ReadAsArray()
    a.output.mkdir(parents=True);(a.output/'buffered_climate').mkdir()
    rows=[];downloads=[];oldmask=valid.copy();newmask=valid.copy()
    for ssp in ['ssp370','ssp585']:
        for model in ['ACCESS-CM2','IPSL-CM6A-LR','MRI-ESM2-0']:
            name=f'wc2.1_30s_tmax_{model}_{ssp}_2041-2060'
            url=f'https://geodata.ucdavis.edu/cmip6/30s/{model}/{ssp}/{name}.tif'
            p=a.output/'buffered_climate'/f'{name}_london_buffer.tif'
            # Geographic rectangle includes London plus surrounding native cells;
            # no resampling or polygon masking at this download stage.
            ds=gdal.Translate(str(p),'/vsicurl/'+url,projWin=[-.65,51.75,.4,51.2],creationOptions=['COMPRESS=DEFLATE'])
            assert ds.RasterCount==12;ds=None
            old=repo/'data/02_preprocess/GCMs/clipped'/f'{name}_London.tif'
            for month in [5,6,7,8]:
                x=warp(old,month);y=warp(p,month)
                ox=valid&np.isfinite(x)&(x!=-9999);ny=valid&np.isfinite(y)&(y!=-9999)
                oldmask &= ox;newmask &= ny
                overlap=ox&ny
                rows.append(dict(ssp=ssp,gcm=model,month=month,old_cells=int(ox.sum()),new_cells=int(ny.sum()),old_mean_c=float(x[ox].mean(dtype=float)),new_mean_c=float(y[ny].mean(dtype=float)),overlap_mean_change_c=float((y[overlap]-x[overlap]).mean(dtype=float)),overlap_max_abs_change_c=float(abs(y[overlap]-x[overlap]).max())))
            downloads.append(dict(path=str(p),url=url,sha256=sha256(p)))
            print('Buffered climate verified',ssp,model,flush=True)
    assert oldmask.sum()==tm['common_cells'];assert np.array_equal(newmask,valid),'Buffered coverage still incomplete'
    pd.DataFrame(rows).to_csv(a.output/'climate_coverage_comparison.csv',index=False)
    # Reconstruct independent lagged offsets, including May 31 for June 1.
    paths=[Path(i['path']) for i in tm['inputs'] if '/hourly/00697/' in i['path']]
    daily,qc=weather(paths);qc.to_csv(a.output/'weather_qc.csv',index=False)
    oldcl=pd.read_csv(temp/'worldclim_monthly_means.csv')
    saved=pd.read_csv(temp/'daily_temperature_exposure.csv').query("scenario=='baseline'")
    errors=[]
    for (branch,ssp,model,year),g in saved.groupby(['reconstruction','ssp','gcm','weather_year']):
        d=daily[daily.year==year];means=oldcl.query('ssp==@ssp and gcm==@model').set_index('month').mean_tmax_c
        shift=means-(tm['baseline_ucm_area_mean_c'] if branch=='worldclim_aligned' else means.loc[7])
        t=d.anomaly_c+d.month.map(shift);lag=(t+t.shift(1))/2
        expected=lag[d.month.isin([6,7,8])];assert len(expected)==92 and expected.notna().all()
        obs=g.set_index(pd.to_datetime(g.date)).lag01_offset_c
        errors.append(float(abs(expected-obs).max()))
    assert max(errors)<1e-10
    fractions_path=diag/'five_outcome_thresholds_2026-11-12/monthly_cause_fractions.csv'
    fractions=pd.read_csv(fractions_path)
    mort=[];sources=[cfgpath,fractions_path,temp/'manifest.json',Path(__file__)]
    for cause in cfg['causes']:
        name=cause['name'];f=fractions.query('cause==@name').sort_values('month')
        assert len(f)==12 and set(f.month)==set(range(1,13)) and (f['count']>=0).all()
        assert np.allclose(f.fraction,f['count']/f.annual) and np.isclose(f.fraction.sum(),1)
        assert np.allclose(f.annual,f['count'].sum())
        p=a.root/cause['mortality_raster'];sources.append(p);ds=gdal.Open(str(p))
        assert ds.GetGeoTransform()==gt and ds.GetProjection()==ref.GetProjection()
        v=ds.ReadAsArray();nd=ds.GetRasterBand(1).GetNoDataValue()
        assert np.isfinite(v[valid]).all() and (v[valid]>=0).all() and (nd is None or (v[valid]!=nd).all())
        share=f.query('month in [6,7,8]').fraction.sum()
        # Day-level allocation must sum back to the same seasonal fraction.
        summed=sum(row.fraction/calendar.monthrange(2021,int(row.month))[1] for row in f.itertuples() if row.month in [6,7,8] for _ in range(calendar.monthrange(2021,int(row.month))[1]))
        assert abs(summed-share)<1e-12
        mort.append(dict(cause=name,summer_fraction=share,old_annual=float(v[oldmask].sum(dtype=float)),full_annual=float(v[valid].sum(dtype=float)),full_summer=float(v[valid].sum(dtype=float)*share)))
    pd.DataFrame(mort).to_csv(a.output/'mortality_coverage_checks.csv',index=False)
    # Explicit cold, threshold-crossing, hot and zero-cooling edge cases.
    b=np.array([23.,25.,27.,26.]);c=np.array([1.,1.,1.,0.]);d=np.array([1.,2.,3.,4.]);rr=1.021
    exact=np.sum(d*(1-np.exp(np.log(rr)*(np.maximum(b-c-24.7,0)-np.maximum(b-24.7,0)))))
    fast=integrate(b,c,d,rr,np.array([-24.7]),np.argsort(b),np.argsort(b-c))[0]
    assert abs(exact-fast)<1e-12
    out=dict(status='coverage candidate and consistency audit only; no health rerun',old_cells=int(oldmask.sum()),full_cells=int(valid.sum()),old_population=float(pop[oldmask].sum(dtype=float)),full_population=float(pop[valid].sum(dtype=float)),remaining_climate_missing_cells=int((valid&~newmask).sum()),lag_reconstruction_max_error=max(errors),formula_edge_case_error=float(abs(exact-fast)),downloads=downloads,sources=[dict(path=str(p),sha256=sha256(p)) for p in sources])
    for x in tm['inputs']:assert sha256(x['path'])==x['sha256']
    (a.output/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['downloads','sources']},indent=2))

if __name__=='__main__':main()
