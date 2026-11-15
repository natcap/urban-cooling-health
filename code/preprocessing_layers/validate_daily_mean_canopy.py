"""Read-only-input MSOA scale diagnostic; no health or UCM production changes.

Usage: python validate_daily_mean_canopy.py DATA_ROOT NEW_OUTPUT_DIRECTORY
Uses all JJA dates 2018–2023 in already acquired Crowd-Grid Heat Packs.
Cross-sectional canopy coefficients are associations, not planting effects.
"""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'post_processing_layers'))
from valuation_accounting import sha256


def fit(x,y):
    assert np.linalg.matrix_rank(x)==x.shape[1]
    return np.linalg.lstsq(x,y,rcond=None)[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args(); gdal.UseExceptions()
    if a.output.exists(): raise FileExistsError(a.output)
    inp=a.root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs'
    stationroot=inp/'climate/uhi_station_pilot_20261111_v1'
    crowd=stationroot/'crowd_grid_recent_heat_packs_v1'
    diag=a.root/'2_postprocess_intermediate/UCM_official_runs/diagnostics'
    sources={}; frames={'maximum':[],'minimum':[]}
    for f in sorted((crowd/'daily_csvs').glob('*.csv')):
        if f.name.split('_')[0] in ['guildford','crawley','central-bedfordshire','mole-valley']: continue
        kind='maximum' if 'maximum' in f.name else 'minimum'
        t=pd.read_csv(f,skiprows=5,index_col=0,parse_dates=True)
        t=t.loc[(t.index.year>=2018)&(t.index.year<=2023)&t.index.month.isin([6,7,8])]
        t=t[[c for c in t.columns if re.search(r' \d{3}$',c)]]
        frames[kind].append(t); sources[f.name]=f
    mx=pd.concat(frames['maximum'],axis=1); mn=pd.concat(frames['minimum'],axis=1)
    assert mx.columns.is_unique and mx.index.is_unique and len(mx)==552
    mn=mn.loc[mx.index,mx.columns]; mean=(mx+mn)/2
    assert mean.notna().all().all()
    bad=(mx<mn); badrows=bad.stack(); badrows=badrows[badrows].reset_index()
    # Separate interpolations / observation windows can yield inconsistent pairs.
    # Do not swap extrema. Remove affected dates across all areas for comparability.
    excluded_dates=mean.index[bad.any(axis=1)]
    mx=mx.drop(index=excluded_dates); mn=mn.drop(index=excluded_dates); mean=(mx+mn)/2
    names=sorted(mean.columns); mean=mean[names]; mx=mx[names]; mn=mn[names]
    print(f'Loaded {len(names)} MSOAs, {len(mean)} days',flush=True)
    # Rasterize original MSOA polygons at the existing 10 m UCM grid; no warp
    # of temperatures or categorical land cover. Use one common spatial mask.
    paths={'baseline':diag/'nodata_correction_2026-11-12/baseline_nodata0/T_air_test.tif',
       'green30':diag/'nodata_correction_2026-11-12/green30_nodata0/T_air_test.tif',
       'target30':diag/'nodata_target30_2026-11-12/target30_nodata0/T_air_test.tif',
       'lulc':diag/'nodata_correction_2026-11-12/baseline_nodata0/input.tif'}
    ref=gdal.Open(str(paths['baseline'])); shape=(ref.RasterYSize,ref.RasterXSize)
    sr=osr.SpatialReference(wkt=ref.GetProjection())
    vector=crowd/'boundaries/uk-msoa-iz-dez-boundaries.shp'
    ds=ogr.Open(str(vector)); layer=ds.GetLayer()
    trans=osr.CoordinateTransformation(layer.GetSpatialRef(),sr)
    mem=ogr.GetDriverByName('Memory').CreateDataSource(''); ml=mem.CreateLayer('msoa',sr,ogr.wkbMultiPolygon)
    ml.CreateField(ogr.FieldDefn('zone',ogr.OFTInteger)); spatial={}; geoms={}
    index={n:i+1 for i,n in enumerate(names)}
    for f in layer:
        n=f.GetField('NAME')
        if n not in index: continue
        geom=f.GetGeometryRef().Clone(); geom.Transform(trans); cent=geom.Centroid()
        spatial[n]=(f.GetField('CODE'),cent.GetX(),cent.GetY()); geoms[n]=geom.Clone()
        out=ogr.Feature(ml.GetLayerDefn()); out.SetField('zone',index[n]); out.SetGeometry(geom); ml.CreateFeature(out)
    assert set(spatial)==set(names)
    raster=gdal.GetDriverByName('MEM').Create('',shape[1],shape[0],1,gdal.GDT_UInt16)
    raster.SetGeoTransform(ref.GetGeoTransform()); raster.SetProjection(ref.GetProjection())
    gdal.RasterizeLayer(raster,[1],ml,options=['ATTRIBUTE=zone'])
    zones=raster.ReadAsArray(); fullcounts=np.bincount(zones.ravel(),minlength=len(names)+1)[1:]
    valid=zones>0; arrays={}
    for k,path in paths.items():
        d=gdal.Open(str(path)); assert d.GetGeoTransform()==ref.GetGeoTransform() and d.GetProjection()==ref.GetProjection()
        v=d.ReadAsArray(); assert v.shape==shape
        valid &= np.isfinite(v)&(v!=d.GetRasterBand(1).GetNoDataValue()); arrays[k]=v
    z=zones[valid]; counts=np.bincount(z,minlength=len(names)+1)[1:]
    assert (counts>0).all()
    def agg(v): return np.bincount(z,weights=v,minlength=len(names)+1)[1:]/counts
    lc=arrays['lulc'][valid]
    tab=pd.DataFrame({'name':names,'code':[spatial[n][0] for n in names],
       'east':[spatial[n][1] for n in names],'north':[spatial[n][2] for n in names],
       'coverage':counts/fullcounts,'canopy_pct':agg(np.isin(lc,[1,2,100]))*100,
       'urban_suburban_pct':agg(np.isin(lc,[20,21]))*100,
       'ucm_baseline':agg(arrays['baseline'][valid])})
    for s in ['green30','target30']:
        tab[s+'_cooling']=agg(arrays['baseline'][valid]-arrays[s][valid])
    # Require adequate baseline spatial coverage; disclose all exclusions.
    keep=tab.coverage>=.95; good=tab.loc[keep].copy(); goodnames=good.name.tolist()
    e=(good.east.to_numpy()-good.east.mean())/10000; n=(good.north.to_numpy()-good.north.mean())/10000
    c=good.canopy_pct.to_numpy(); built=good.urban_suburban_pct.to_numpy()
    designs={'canopy_only':np.column_stack([np.ones(len(c)),c]),
      'built_adjusted':np.column_stack([np.ones(len(c)),c,built]),
      'built_geography_adjusted':np.column_stack([np.ones(len(c)),c,built,e,n,e*e,e*n,n*n])}
    # Contiguous east-west strips: assess spatial extrapolation, not random folds.
    folds=pd.qcut(good.east,5,labels=False).to_numpy()
    results=[]; cv=[]
    periods=[('all_jja',np.ones(len(mean),dtype=bool))]+[(str(y),mean.index.year==y) for y in range(2018,2024)]
    for period,days in periods:
        for metric,t in [('tmax',mx),('tmin',mn),('midrange',mean)]:
            y=t.loc[days,goodnames].mean().to_numpy()
            for model,x in designs.items():
                beta=fit(x,y); pred=x@beta
                results.append(dict(period=period,metric=metric,model=model,n=len(y),
                    canopy_c_per_10pp=float(beta[1]*10),rmse=float(np.sqrt(np.mean((y-pred)**2)))))
            if period=='all_jja':
                good[metric]=y
                for include in [False,True]:
                    x=designs['built_geography_adjusted']
                    if not include: x=np.delete(x,1,axis=1)
                    pred=np.empty(len(y))
                    for fold in range(5):
                        train=folds!=fold; test=~train; pred[test]=x[test]@fit(x[train],y[train])
                    cv.append(dict(metric=metric,include_canopy=include,rmse=float(np.sqrt(np.mean((y-pred)**2))),
                                   mae=float(np.mean(abs(y-pred)))))
    for model,x in designs.items():
        beta=fit(x,good.ucm_baseline.to_numpy())
        results.append(dict(period='static_25_5',metric='ucm',model=model,n=len(c),canopy_c_per_10pp=float(beta[1]*10)))
    # Station/area comparison is NOT independent validation: Crowd-Grid may
    # assimilate MIDAS and a point represents a different support from an MSOA.
    screen=stationroot/'station_screening_decision.csv'; st=pd.read_csv(screen)
    st=st[pd.to_numeric(st.station_id)==697].iloc[0]
    wgs=osr.SpatialReference(); wgs.ImportFromEPSG(4326); wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    point=ogr.Geometry(ogr.wkbPoint); point.AddPoint(float(st.longitude),float(st.latitude)); point.Transform(osr.CoordinateTransformation(wgs,sr))
    area=[name for name,g in geoms.items() if g.Intersects(point)]; assert len(area)==1
    dailyfile=diag/'london_heat_response_2026-11-12/station_daily_temperature.csv'
    station=pd.read_csv(dailyfile,index_col=0,parse_dates=True).loc['2021-06-01':'2021-08-31']
    paired=station.loc[station.index.intersection(mean.index),['extrema_mean','mean']].rename(columns={'mean':'hourly_mean'})
    paired['crowd_midrange']=mean.loc[paired.index,area[0]]
    paired['london_msoa_average']=mean.loc[paired.index].mean(axis=1)
    stationchecks=[]
    for col in paired:
        diff=paired[col]-paired.extrema_mean
        # Recompute lagged threshold counts with two preceding days from sources.
        if col in ['extrema_mean','hourly_mean']:
            original=pd.read_csv(dailyfile,index_col=0,parse_dates=True)
            series=original['mean' if col=='hourly_mean' else col]
            hot=int((series.rolling(3).mean().loc[paired.index]>18.9).sum())
        else: hot=None  # source subset excludes May; do not fabricate lag history
        stationchecks.append(dict(series=col,bias=float(diff.mean()),rmse=float(np.sqrt(np.mean(diff**2))),
                                  days_daily_above_18_9=int((paired[col]>18.9).sum()),days_lag02_above_18_9=hot))
    summary={'status':'diagnostic associations, not causal cooling calibration',
      'msoa_total':len(tab),'msoa_coverage_ge_95pct':len(good),'excluded':tab.loc[~keep,['name','coverage']].to_dict('records'),
      'days':len(mean),'inverted_extrema_pairs':int(bad.sum().sum()),
      'excluded_dates':[str(d.date()) for d in excluded_dates],'station_paired_days':len(paired),'station_msoa':area[0],
      'station_checks':stationchecks,'spatial_cv':cv,
      'ucm_vs_midrange_spatial_correlation':float(good.ucm_baseline.corr(good.midrange)),
      'ucm_vs_midrange_centered_rmse':float(np.sqrt(np.mean(((good.ucm_baseline-good.ucm_baseline.mean())-(good.midrange-good.midrange.mean()))**2))),
      'limitations':['Crowd-Grid prototype, interpolated and potentially shares MIDAS observations',
        '2023 land cover plus 2024 canopy overlay versus 2018–2023 weather; static canopy proxy',
        'urban/suburban LULC fraction is not building footprint or height; elevation omitted',
        'spatial cross-sectional association, not causal effect of adding trees',
        'UCM is static 25/5 stress test, not observed historical daily temperature',
        'five contiguous easting folds, no independent external validation; no p-values'],
      'sources':{k:{'path':str(v),'sha256':sha256(v)} for k,v in {**sources,**paths,'station_daily':dailyfile,'station_screen':screen,
         **{'boundary_'+ext:vector.with_suffix('.'+ext) for ext in ['shp','shx','dbf','prj']},'script':Path(__file__)}.items()}}
    a.output.mkdir(parents=True)
    badrows.to_csv(a.output/'inconsistent_extrema_pairs.csv',index=False)
    tab.to_csv(a.output/'msoa_input_coverage.csv',index=False); good.to_csv(a.output/'msoa_comparison.csv',index=False)
    pd.DataFrame(results).to_csv(a.output/'canopy_associations.csv',index=False)
    pd.DataFrame(cv).to_csv(a.output/'spatial_holdout.csv',index=False)
    paired.to_csv(a.output/'station_grid_comparison.csv',index_label='date')
    (a.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(pd.DataFrame(results).query("period in ['all_jja','static_25_5']").to_string(index=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))


if __name__=='__main__': main()
