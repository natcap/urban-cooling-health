"""Compare historical references without replacing the current Figure 1.

Match the saved Figure 1 inputs: centre-based London clip and borough
summaries; equal borough/model weights. No climate recalibration. The current
notebook says all_touched=True, but saved CSV counts/means match centres.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from osgeo import gdal,ogr,osr
from valuation_accounting import sha256

def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('repo',type=Path);p.add_argument('data_root',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    inputs=a.data_root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs'
    raw=inputs/'climate/worldclim_v21_historical/wc2.1_30s_tmax_07.tif'
    vector=inputs/'AOIs/London_Borough_aoi.shp'
    ds=gdal.Open(str(raw)); band=ds.GetRasterBand(1)
    src=ogr.Open(str(vector)); lyr=src.GetLayer()
    target=osr.SpatialReference(wkt=ds.GetProjection());target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    origin=lyr.GetSpatialRef();origin.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr=osr.CoordinateTransformation(origin,target)
    feats=[]
    for f in lyr:
        geom=f.GetGeometryRef().Clone();geom.Transform(tr)
        feats.append((f.GetField('NAME'),geom))
    assert len(feats)==33 and len({n for n,g in feats})==33
    boxes=[g.GetEnvelope() for n,g in feats]
    win=[min(b[0] for b in boxes)-.02,max(b[3] for b in boxes)+.02,
         max(b[1] for b in boxes)+.02,min(b[2] for b in boxes)-.02]
    clip=gdal.Translate('',ds,format='MEM',projWin=win)
    values=clip.ReadAsArray().astype(float)
    scale=band.GetScale() or 1;offset=band.GetOffset() or 0
    valid=np.isfinite(values)&(values!=band.GetNoDataValue())
    values=values*scale+offset
    def mask(geoms,touched=False):
        v=ogr.GetDriverByName('Memory').CreateDataSource('');l=v.CreateLayer('mask',target,ogr.wkbMultiPolygon)
        for geom in geoms:
            f=ogr.Feature(l.GetLayerDefn());f.SetGeometry(geom);l.CreateFeature(f)
        m=gdal.GetDriverByName('MEM').Create('',clip.RasterXSize,clip.RasterYSize,1,gdal.GDT_Byte)
        m.SetGeoTransform(clip.GetGeoTransform());m.SetProjection(clip.GetProjection())
        gdal.RasterizeLayer(m,[1],l,burn_values=[1],options=['ALL_TOUCHED=TRUE'] if touched else [])
        return m.ReadAsArray()>0
    london=mask([g for n,g in feats]);valid &= london
    assert 5<np.mean(values[valid])<40 # Celsius sanity check, never rescale by guess.
    # Validate this implementation against a known future raster/CSV pair.
    original_clip=clip
    future_dir=a.repo/'data/02_preprocess/GCMs/clipped'
    diffs=[]
    validation_files=sorted((future_dir/'stats').glob('borough_stats_wc2.1_*tmax_*ssp*.csv'))
    assert len(validation_files)==18
    for csv_path in validation_files:
        clip=gdal.Open(str(future_dir/(csv_path.stem.removeprefix('borough_stats_')+'.tif')))
        arr=clip.GetRasterBand(7).ReadAsArray().astype(float)
        ok=np.isfinite(arr)&(arr!=clip.GetRasterBand(7).GetNoDataValue())
        expected={r['borough']:r for r in csv.DictReader(csv_path.open())}
        for name,geom in feats:
            v=arr[mask([geom],False)&ok]
            assert len(v)==int(expected[name]['count']),(csv_path.name,name,len(v),expected[name]['count'])
            diffs.append(abs(float(v.mean())-float(expected[name]['mean'])))
    assert max(diffs)<1e-4,max(diffs)
    clip=original_clip
    baseline_file=a.repo/'data/02_preprocess/chelsa02/clipped_london_stats/borough_stats_CHELSA_tasmax_07_2021_V.2.1_London.csv'
    chelsa={r['borough']:float(r['mean']) for r in csv.DictReader(baseline_file.open())}
    rows=[]
    for name,geom in feats:
        vals=values[mask([geom],False)&valid]
        assert len(vals)>0
        rows.append(dict(borough=name,chelsa_2021=chelsa[name],worldclim_1970_2000=float(vals.mean()),
            worldclim_count=len(vals),baseline_offset=float(vals.mean())-chelsa[name]))
    lookup={r['borough']:r for r in rows};future=[];sources=[]
    for path in sorted((a.repo/'data/02_preprocess/GCMs/clipped/stats').glob('borough_stats_wc2.1_*tmax_*ssp*.csv')):
        suffix=path.stem.split('tmax_')[1].split('_');model,ssp,period=suffix[:3]
        records=list(csv.DictReader(path.open()));assert len(records)==33
        sources.append(dict(path=str(path),sha256=sha256(path)))
        for r in records:
            old=lookup[r['borough']];t=float(r['mean'])
            future.append(dict(borough=r['borough'],model=model,ssp=ssp,period=period,
                future=t,vs_chelsa=t-old['chelsa_2021'],vs_worldclim=t-old['worldclim_1970_2000']))
    assert len(future)==594
    summaries=[]
    for ssp in ['ssp370','ssp585']:
        for period in ['2021-2040','2041-2060','2061-2080']:
            subset=[r for r in future if r['ssp']==ssp and r['period']==period]
            assert len(subset)==99
            summaries.append(dict(ssp=ssp,period=period,**{k:float(np.mean([r[k] for r in subset])) for k in ['future','vs_chelsa','vs_worldclim']}))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    stem='figure1_baseline_comparison_20261111_v1'
    for suffix,data in [('baselines',rows),('by_model',future),('summary',summaries)]:write(a.output_dir/f'{stem}_{suffix}.csv',data)
    clipped=a.output_dir/'worldclim_v21_july_1970_2000_london.tif'
    out=gdal.GetDriverByName('GTiff').Create(str(clipped),clip.RasterXSize,clip.RasterYSize,1,gdal.GDT_Float32,options=['COMPRESS=LZW'])
    out.SetGeoTransform(clip.GetGeoTransform());out.SetProjection(clip.GetProjection())
    out.GetRasterBand(1).SetNoDataValue(-9999);out.GetRasterBand(1).WriteArray(np.where(valid,values,-9999));out=None
    manifest=dict(script_sha256=sha256(__file__),worldclim_sha256=sha256(raw),boundary_sha256=sha256(vector),
        chelsa_sha256=sha256(baseline_file),future_sources=sources,scale=scale,offset=offset,
        validation_max_future_csv_error=max(diffs),
        validation_records=len(diffs),
        method='London and borough cell-centre masks; native WGS84 grid, no resampling; equal weights',
        baseline_means={k:float(np.mean([r[k] for r in rows])) for k in ['chelsa_2021','worldclim_1970_2000','baseline_offset']})
    (a.output_dir/f'{stem}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['baseline_means'],indent=2));print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
