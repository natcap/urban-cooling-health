"""Isolated UHI (2/3/5 C) and spatial-response-scale health sensitivity.

Run with InVEST Python: pilot_uhi_threshold_scale.py DATA_ROOT NEW_OUTPUT.
Full cooling transfer throughout: UHI scaling is NOT transfer-fraction scaling.
"""
import argparse
import calendar
import json
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
from pilot_spatial_reconstructed_health import integrate, london_grid, sha256

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')


def common_baseline(c, deaths, rr, q, order):
    """Hybrid: common regional baseline exposure but retain each pixel's cooling."""
    c=c[order];w=deaths[order]
    cw=np.r_[0.,np.cumsum(w)]
    ca=np.r_[0.,np.cumsum(w*-np.expm1(-np.log(rr)*c))]
    k=np.searchsorted(c,q,side='right')
    return ca[k]+(cw[-1]-cw[k])*-np.expm1(-np.log(rr)*q)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    repo=Path(__file__).resolve().parents[2]
    diag=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    temp=diag/'temperature_reconstructions_2050_2026-11-12_v1'
    mf=temp/'manifest.json';tm=json.loads(mf.read_text())
    for x in tm['inputs']:
        assert sha256(x['path'])==x['sha256'],x['path']
    cfgpath=Path(__file__).with_name('health-analysis-revised-equal-area.example.json')
    cfg=json.loads(cfgpath.read_text())
    candidate=repo/'docs/manuscript/manuscript_health_response_candidates.csv'
    rrtable=pd.read_csv(candidate).set_index('endpoint')
    assert (rrtable.threshold_c==24.7).all()
    fractions=diag/'five_outcome_thresholds_2026-11-12/monthly_cause_fractions.csv'
    months=pd.read_csv(fractions).set_index(['cause','month'])
    offsets_path=temp/'daily_temperature_exposure.csv'
    daily=pd.read_csv(offsets_path,parse_dates=['date']).query("scenario=='baseline'").reset_index(drop=True)
    keys=['reconstruction','ssp','gcm','weather_year']
    assert len(daily)==5520
    paths={s:diag/('nodata_target30_2026-11-12' if s=='target30' else 'nodata_correction_2026-11-12')/f'{s}_nodata0/T_air_test.tif' for s in ['baseline','green30','target30']}
    poppath=a.root/cfg['population_2021']
    boundary=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    _,zones,_=london_grid(boundary,[*paths.values(),poppath]);valid=zones>0
    first=gdal.Open(str(paths['baseline']));gt=first.GetGeoTransform()
    bounds=[gt[0],gt[3]+first.RasterYSize*gt[5],gt[0]+first.RasterXSize*gt[1],gt[3]]
    for item in tm['inputs']:
        if 'wc2.1_30s_tmax_' not in item['path']:continue
        for month in [5,6,7,8]:
            band=gdal.Translate('',item['path'],format='MEM',bandList=[month])
            w=gdal.Warp('',band,format='MEM',dstSRS=first.GetProjection(),outputBounds=bounds,
                width=first.RasterXSize,height=first.RasterYSize,resampleAlg='bilinear',dstNodata=-9999,outputType=gdal.GDT_Float32)
            v=w.ReadAsArray();valid &= np.isfinite(v)&(v!=-9999);w=band=None
    assert valid.sum()==tm['common_cells']
    pop=gdal.Open(str(poppath)).ReadAsArray()[valid].astype(float)
    assert np.isclose(pop.sum(),tm['population_on_common_mask'],rtol=0,atol=1e-6)
    weights=pop/pop.sum();del pop,zones
    b5=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float)
    mean5=float(b5.mean());assert abs(mean5-tm['baseline_ucm_area_mean_c'])<1e-10
    ob=np.argsort(b5)
    records=[];checks=[];tempstats=[]
    sources=[mf,offsets_path,cfgpath,candidate,fractions,Path(__file__),
        Path(__file__).with_name('pilot_spatial_reconstructed_health.py'),repo/'code/post_processing_layers/valuation_accounting.py']
    from natcap.invest.urban_cooling_model import urban_cooling_model as ucm
    sources.append(Path(ucm.__file__))
    # Fixed HM and normalized mixing: T_u=25+(u/5)*(T_5-25).
    # Scale corrected 5 C fields, never mix old uncorrected lower-UHI runs.
    rng=np.random.default_rng(20261112);sample=rng.choice(len(b5),1500,replace=False)
    for uhi in [2.,3.,5.]:
        ratio=uhi/5.;b=25+ratio*(b5-25);mean_u=float(b.mean());region_b=float(b@weights)
        off=daily.lag01_offset_c.to_numpy().copy()
        # WorldClim anchor is fixed; anchored branch retains its fixed rural 25 C.
        off[daily.reconstruction.eq('worldclim_aligned')]+=mean5-mean_u
        off-=24.7
        regional_excess=np.maximum(region_b+off,0)
        for scenario,path in paths.items():
            c=ratio*(b5-gdal.Open(str(path)).ReadAsArray()[valid].astype(float))
            assert c.min()>=0
            os=np.argsort(b-c);oc=np.argsort(c)
            region_c=float(c@weights)
            tempstats.append(dict(uhi_c=uhi,scenario=scenario,ucm_area_mean_c=mean_u,
                baseline_spatial_p95_minus_p05_c=float(np.quantile(b,.95)-np.quantile(b,.05)),
                population_cooling_c=region_c,area_cooling_c=float(c.mean())))
            for cause in cfg['causes']:
                name=cause['name'];p=a.root/cause['mortality_raster'];sources.append(p)
                ds=gdal.Open(str(p));assert ds.GetGeoTransform()==gt and ds.GetProjection()==first.GetProjection()
                death=ds.ReadAsArray()[valid].astype(float);nd=ds.GetRasterBand(1).GetNoDataValue()
                assert np.isfinite(death).all() and (death>=0).all() and (nd is None or (death!=nd).all())
                scale=np.array([months.loc[(name,t.month),'fraction']/calendar.monthrange(2021,t.month)[1] for t in daily.date])
                for label,col in [('central','rr_per_1c'),('rr_low','ci95_lower'),('rr_high','ci95_upper')]:
                    rr=float(rrtable.loc[name,col])
                    if scenario=='baseline':
                        pix=hybrid=regional=np.zeros(len(daily))
                    else:
                        pix=integrate(b,c,death,rr,off,ob,os)
                        hybrid=common_baseline(c,death,rr,regional_excess,oc)
                        regional=death.sum()*-np.expm1(-np.log(rr)*np.minimum(region_c,regional_excess))
                    # Direct arithmetic verifies pixel and hybrid algorithms on real cells.
                    bs,cs,ws=b[sample],c[sample],death[sample];xx=np.r_[-50.,0.,50.,off[::500]]
                    direct=np.array([np.sum(ws*-np.expm1(np.log(rr)*(np.maximum(bs-cs+x,0)-np.maximum(bs+x,0)))) for x in xx])
                    fast=integrate(bs,cs,ws,rr,xx,np.argsort(bs),np.argsort(bs-cs))
                    qq=np.maximum(region_b+xx,0)
                    hfast=common_baseline(cs,ws,rr,qq,np.argsort(cs))
                    hdir=np.array([np.sum(ws*-np.expm1(-np.log(rr)*np.minimum(cs,q))) for q in qq])
                    err=max(float(abs(fast-direct).max()),float(abs(hfast-hdir).max()));assert err<1e-8
                    checks.append(dict(uhi_c=uhi,scenario=scenario,cause=name,rr_variant=label,max_direct_error=err))
                    for mode,values in [('pixel',pix),('regional_baseline_pixel_cooling',hybrid),('regional',regional)]:
                        frame=daily[keys].copy();frame['avoided']=values*scale
                        result=frame.groupby(keys,as_index=False).avoided.sum()
                        for r in result.to_dict('records'):
                            records.append(dict(**r,uhi_c=uhi,exposure_mode=mode,scenario=scenario,
                                cause=name,rr_variant=label,rr=rr,cooling_transfer=1.))
                print(uhi,scenario,name,'complete',flush=True)
    result=pd.DataFrame(records)
    k=keys+['uhi_c','exposure_mode','scenario','cause','rr_variant']
    assert len(result)==24300 and not result.duplicated(k).any()
    assert (result.avoided>=0).all() and (result.query("scenario=='baseline'").avoided==0).all()
    oldpath=diag/'spatial_reconstructed_health_2026-11-12_v1/citywide_health_comparison.csv';sources.append(oldpath)
    old=pd.read_csv(oldpath).query('cooling_transfer==1')
    match=result.query("uhi_c==5 and exposure_mode=='pixel'").merge(old,on=keys+['scenario','cause','rr_variant'],suffixes=('_new','_old'),validate='one_to_one')
    regression_error=float(abs(match.avoided_new-match.avoided_old).max());assert regression_error<1e-6
    summary=result.groupby(['reconstruction','ssp','uhi_c','exposure_mode','scenario','cause','rr_variant']).avoided.agg(['mean','min','max']).reset_index()
    # Do not assume a ranking: report every matched pair and any reversals.
    ranking=result.query("scenario!='baseline'").pivot(index=keys+['uhi_c','exposure_mode','cause','rr_variant'],columns='scenario',values='avoided').reset_index()
    ranking['target_minus_green']=ranking.target30-ranking.green30
    a.output.mkdir(parents=True)
    result.to_csv(a.output/'health_sensitivity.csv',index=False)
    summary.to_csv(a.output/'ensemble_summary.csv',index=False)
    ranking.to_csv(a.output/'ranking_checks.csv',index=False)
    pd.DataFrame(checks).to_csv(a.output/'direct_calculation_checks.csv',index=False)
    pd.DataFrame(tempstats).to_csv(a.output/'uhi_temperature_scaling.csv',index=False)
    record=dict(status='isolated diagnostic; no production adoption',common_cells=int(valid.sum()),
        temperature_manifest=str(mf),uhi_values=[2,3,5],cooling_transfer=1,
        uhi_scaling='T_u=25+(u/5)*(T_5-25); cooling scales by u/5; WorldClim monthly mean remains fixed',
        modes=dict(pixel='pixel baseline and scenario temperatures; sum cause-specific pixel effects',
            regional_baseline_pixel_cooling='common population-weighted regional baseline; pixel cooling and mortality retained; hybrid diagnostic',
            regional='population-weight baseline and scenario temperatures first; apply response to total cause-specific mortality'),
        regression_max_error=regression_error,
        limitations=['Regional calculation is a spatial-scale sensitivity, not reproduction of original station network/weights',
                     'UHI scaling assumes fixed heat mitigation and mixing; analytic transformation, no new UCM runs',
                     'Full cooling transfer only to isolate UHI; lower UHI is not a half-transfer correction',
                     'Fixed 2021 population and baseline mortality; seasonal fractions and temperature limitations inherited',
                     'Regional-vs-pixel includes threshold nonlinearity, exposure averaging and mortality/cooling covariance',
                     'No neighbourhood equity inference or spatial health maps'],
        sources=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in sorted(set(sources))])
    (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(summary.query("reconstruction=='worldclim_aligned' and ssp=='ssp370' and rr_variant=='central' and scenario!='baseline'").to_string(index=False))


if __name__=='__main__':main()
