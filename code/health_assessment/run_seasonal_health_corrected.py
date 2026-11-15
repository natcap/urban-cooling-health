"""Versioned nine-scenario seasonal health run; never overwrites existing outputs.

Exact daily risk integration, then equal average over 3 GCMs x 5 weather years.
Primary SSP370 WorldClim: mean raster + RR-endpoint zonal summaries. Other
SSP/reconstruction combinations: central-RR summaries. No annualisation.
"""
import argparse,calendar,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
from osgeo import gdal,ogr,osr
from pilot_spatial_reconstructed_health import london_grid,sha256


def seasonal_fraction(b,s,rr,offsets,weights):
    """Exact time-summed fractional benefit per pixel, allowing signed cooling.

    offsets already include minus threshold; weights include seasonal mortality
    fractions/day count and ensemble weight. No temperature or benefit bins.
    """
    idx=np.argsort(offsets);o=offsets[idx];w=weights[idx];beta=np.log(rr)
    pw=np.r_[0.,np.cumsum(w)];result=np.zeros(len(b),dtype=float)
    for warming in [False,True]:
        m=s>b if warming else s<b
        if not m.any():continue
        lo=np.minimum(b[m],s[m]);hi=np.maximum(b[m],s[m])
        kl=np.searchsorted(o,-lo,side='right');kh=np.searchsorted(o,-hi,side='right')
        if warming:
            pe=np.r_[0.,np.cumsum(w*np.exp(beta*o))]
            result[m]=(pw[-1]-pw[kh])-np.exp(beta*(hi-lo))*(pw[-1]-pw[kl])-np.exp(beta*hi)*(pe[kl]-pe[kh])
        else:
            pe=np.r_[0.,np.cumsum(w*np.exp(-beta*o))]
            result[m]=(pw[-1]-pw[kh])-np.exp(beta*(lo-hi))*(pw[-1]-pw[kl])-np.exp(-beta*hi)*(pe[kl]-pe[kh])
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    repo=Path(__file__).resolve().parents[2];diag=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    tmfile=diag/'remaining_corrected_temperatures_2026-11-12_v1/manifest.json';tm=json.loads(tmfile.read_text())
    paths={k:Path(v) for k,v in tm['scenario_paths'].items()}
    for k,p in paths.items():assert sha256(p)==tm['outputs'][k]
    cfgfile=Path(__file__).with_name('health-analysis-revised-equal-area.example.json');cfg=json.loads(cfgfile.read_text())
    response=repo/'docs/manuscript/manuscript_health_response_candidates.csv';rrs=pd.read_csv(response).set_index('endpoint');assert (rrs.threshold_c==24.7).all()
    ofile=diag/'nine_scenario_health_readiness_2026-11-12_v2/shared_daily_offsets.csv';offsets=pd.read_csv(ofile,parse_dates=['date']);offsets['month']=offsets.date.dt.month
    ffile=diag/'five_outcome_thresholds_2026-11-12/monthly_cause_fractions.csv';fractions=pd.read_csv(ffile).set_index(['cause','month'])
    poppath=a.root/cfg['population_2021'];vector=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp'
    mortality={c['name']:a.root/c['mortality_raster'] for c in cfg['causes']}
    boroughs,zones,counts=london_grid(vector,list(paths.values())+[poppath]+list(mortality.values()));valid=zones>0
    assert valid.sum()==14258588
    base=gdal.Open(str(paths['baseline'])).ReadAsArray()[valid].astype(float)
    oldmanifest=json.loads((ofile.parent/'manifest.json').read_text());assert abs(base.mean()-oldmanifest['corrected_baseline_area_mean_c'])<1e-10
    ds=gdal.Open(str(paths['baseline']));transform=ds.GetGeoTransform();crs=ds.GetProjection();shape=(ds.RasterYSize,ds.RasterXSize)
    vfile=repo/'data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg';vds=ogr.Open(str(vfile));vl=vds.GetLayer();assert vl.GetSpatialRef().IsSame(osr.SpatialReference(wkt=crs))
    v=pd.DataFrame([dict(id=f.GetField('id'),LSOA11CD=f.GetField('LSOA11CD'),geometry=shapely.from_wkb(bytes(f.GetGeometryRef().ExportToWkb()))) for f in vl]).sort_values('id')
    assert len(v)==4835 and v.id.is_unique and v.LSOA11CD.is_unique
    mem=ogr.GetDriverByName('Memory').CreateDataSource('');layer=mem.CreateLayer('lsoa',osr.SpatialReference(wkt=crs),ogr.wkbUnknown);layer.CreateField(ogr.FieldDefn('id',ogr.OFTInteger))
    for row in v.itertuples():
        f=ogr.Feature(layer.GetLayerDefn());f.SetField('id',int(row.id));f.SetGeometry(ogr.CreateGeometryFromWkb(row.geometry.wkb));layer.CreateFeature(f)
    rz=gdal.GetDriverByName('MEM').Create('',shape[1],shape[0],1,gdal.GDT_Int32);rz.SetGeoTransform(transform);rz.SetProjection(crs);gdal.RasterizeLayer(rz,[1],layer,options=['ATTRIBUTE=id']);lsoa=rz.ReadAsArray();rz=None
    rows,cols=np.nonzero(valid&(lsoa==0));maxdist=0.
    if len(rows):
        x=transform[0]+(cols+.5)*transform[1]+(rows+.5)*transform[2];y=transform[3]+(cols+.5)*transform[4]+(rows+.5)*transform[5]
        tree=shapely.STRtree(v.geometry.to_numpy());indices,dist=tree.query_nearest(shapely.points(x,y),all_matches=True,return_distance=True)
        j=pd.DataFrame(dict(point=indices[0],id=v.id.to_numpy()[indices[1]],distance=dist)).sort_values(['point','distance','id']).drop_duplicates('point').sort_values('point')
        assert len(j)==len(rows);lsoa[rows,cols]=j.id.astype(int);maxdist=float(j.distance.max())
    ids=v.id.to_numpy(int);lz=lsoa[valid];bz=zones[valid];assert (lz>0).all()
    pop=gdal.Open(str(poppath)).ReadAsArray()[valid].astype(float)
    ps=np.bincount(lz,weights=pop,minlength=int(ids.max())+1)[ids]
    # Keep the original Figure 6/7 population denominator for matched comparisons;
    # report covered-population differences explicitly instead of silently changing it.
    pfile=repo/'data/derived/lsoa_population_2021_by_lsoa11cd.csv';pp=pd.read_csv(pfile).set_index('LSOA11CD').loc[v.LSOA11CD,'population_2021'].to_numpy()
    a.output.mkdir(parents=True);(a.output/'rasters').mkdir()
    pd.DataFrame(dict(LSOA11CD=v.LSOA11CD,id=ids,population_2021=pp,population_on_health_mask=ps)).to_csv(a.output/'population_coverage.csv',index=False)
    rrcols=[('central','rr_per_1c'),('rr_low','ci95_lower'),('rr_high','ci95_upper')]
    sources=[tmfile,cfgfile,response,ofile,ffile,poppath,vfile,pfile,Path(__file__),*paths.values(),*mortality.values()]
    before={str(p):sha256(p) for p in set(sources)}
    records=[];ls=[];bs=[];tests=[];rng=np.random.default_rng(20261112)
    for scenario,path in paths.items():
        s=gdal.Open(str(path)).ReadAsArray()[valid].astype(float)
        for cause,mpath in mortality.items():
            deaths=gdal.Open(str(mpath)).ReadAsArray()[valid].astype(float)
            for (branch,ssp),d in offsets.groupby(['reconstruction','ssp']):
                assert len(d)==1380 and len(d[['gcm','weather_year']].drop_duplicates())==15
                off=d.lag01_offset_c.to_numpy()-24.7
                weight=np.array([fractions.loc[(cause,int(m)),'fraction']/calendar.monthrange(2021,int(m))[1]/15 for m in d.month])
                primary=branch=='worldclim_aligned' and ssp=='ssp370'
                for variant,col in (rrcols if primary else rrcols[:1]):
                    rr=float(rrs.loc[cause,col]);values=np.empty(len(base))
                    for start in range(0,len(base),500000):
                        sl=slice(start,start+500000);values[sl]=deaths[sl]*seasonal_fraction(base[sl],s[sl],rr,off,weight)
                    # Compare exact time-integrated pixel results against daily arithmetic.
                    ix=rng.choice(len(base),20,replace=False)
                    exact=np.array([deaths[i]*np.sum(weight*-np.expm1(np.log(rr)*(np.maximum(s[i]+off,0)-np.maximum(base[i]+off,0)))) for i in ix])
                    error=float(abs(exact-values[ix]).max());assert error<1e-10
                    if scenario=='baseline':assert (values==0).all()
                    elif scenario in ['allbuilt','treerisk']:assert values.max()<1e-9 and values.sum()<0
                    else:assert values.min()>-1e-9
                    meta=dict(scenario=scenario,cause=cause,reconstruction=branch,ssp=ssp,rr_variant=variant)
                    sums=np.bincount(lz,weights=values,minlength=int(ids.max())+1)[ids];boroughsum=np.bincount(bz,weights=values,minlength=34)[1:]
                    total=float(values.sum());assert abs(sums.sum()-total)<1e-7 and abs(boroughsum.sum()-total)<1e-7
                    records.append(dict(**meta,deaths_averted=total,baseline_annual_deaths=float(deaths.sum()),baseline_summer_deaths=float(deaths.sum()*sum(fractions.loc[(cause,m),'fraction'] for m in [6,7,8]))))
                    tests.append(dict(**meta,direct_max_error=error,lsoa_reconciliation_error=float(sums.sum()-total),borough_reconciliation_error=float(boroughsum.sum()-total)))
                    ls.append(pd.DataFrame(dict(**{k:[x]*len(ids) for k,x in meta.items()},LSOA11CD=v.LSOA11CD.to_numpy(),deaths_averted=sums)))
                    bs.append(pd.DataFrame(dict(**{k:[x]*33 for k,x in meta.items()},borough_id=[r[0] for r in boroughs],borough_name=[r[1] for r in boroughs],deaths_averted=boroughsum)))
                    if primary and variant=='central' and scenario!='baseline':
                        image=np.full(shape,-9999.,dtype='float32');image[valid]=values.astype('float32')
                        dst=gdal.GetDriverByName('GTiff').Create(str(a.output/'rasters'/f'{scenario}_{cause}_summer_deaths_averted.tif'),shape[1],shape[0],1,gdal.GDT_Float32,options=['COMPRESS=DEFLATE','PREDICTOR=3','TILED=YES'])
                        dst.SetGeoTransform(transform);dst.SetProjection(crs);dst.GetRasterBand(1).SetNoDataValue(-9999);dst.GetRasterBand(1).WriteArray(image);dst=None
            print('DONE',scenario,cause,flush=True)
        pd.DataFrame(records).to_csv(a.output/'citywide_results.csv',index=False)
        pd.concat(ls).to_csv(a.output/'lsoa_results.csv',index=False)
        pd.concat(bs).to_csv(a.output/'borough_results.csv',index=False)
    pd.DataFrame(tests).to_csv(a.output/'numerical_checks.csv',index=False)
    for p,h in before.items():assert sha256(p)==h,p
    manifest=dict(status='completed isolated seasonal health rerun; not adopted manuscript outputs',settings=dict(primary='worldclim_aligned SSP370',uhi=5,cooling_transfer=1,population_year=2021,mortality_year=2021,season='June-August',threshold=24.7,lag='0-1 Tmax',ensemble='equal 3 GCM x 5 historical weather patterns; response evaluated before averaging'),common_cells=int(valid.sum()),population=float(pop.sum()),lsoa_count=len(v),edge_cells_assigned=len(rows),edge_max_distance_m=maxdist,scenario_paths={k:str(p) for k,p in paths.items()},sources=before,limitations=['Conditional fixed baseline deaths; not future mortality forecast','RR endpoints are coefficient sensitivity, not total uncertainty','Regional response transferred to pixels','London registrations seasonally allocated with England occurrence fractions','Existing population denominator retained for equity; covered population exported separately','Primary rasters float32; zonal and city sums calculated float64','No new full spatial-GAM sensitivity fits in health runner'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE',a.output,flush=True)

if __name__=='__main__':main()
