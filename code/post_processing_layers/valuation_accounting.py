"""Exclusive London-cell accounting and conserved building-area allocation."""
from pathlib import Path
import hashlib
import numpy as np
import shapely
from osgeo import gdal, ogr, osr


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def boroughs(path):
    ds = ogr.Open(str(path)); layer = ds.GetLayer()
    rows = [(f.GetField('GSS_CODE'), f.GetField('NAME'),
             bytes(f.GetGeometryRef().ExportToWkb())) for f in layer]
    rows.sort()
    if len(rows) != 33 or len({r[0] for r in rows}) != 33:
        raise ValueError('Expected 33 unique London boroughs')
    return rows, layer.GetSpatialRef().ExportToWkt()


def london_grid(vector, rasters):
    """One borough per cell centre; one common valid footprint for all scenarios."""
    rows, wkt = boroughs(vector)
    first = gdal.Open(str(rasters[0])); grid = first.GetGeoTransform()
    proj = first.GetProjection(); shape = (first.RasterYSize, first.RasterXSize)
    a = osr.SpatialReference(wkt=wkt); b = osr.SpatialReference(wkt=proj)
    if not a.IsSame(b): raise ValueError('Borough and raster CRS differ')
    mem = ogr.GetDriverByName('Memory').CreateDataSource('')
    layer = mem.CreateLayer('boroughs', a, ogr.wkbMultiPolygon)
    layer.CreateField(ogr.FieldDefn('zone', ogr.OFTInteger))
    for i, (_, _, geom) in enumerate(rows, 1):
        f = ogr.Feature(layer.GetLayerDefn()); f.SetField('zone', i)
        f.SetGeometry(ogr.CreateGeometryFromWkb(geom)); layer.CreateFeature(f)
    mask = gdal.GetDriverByName('MEM').Create('', shape[1], shape[0], 1, gdal.GDT_Byte)
    mask.SetGeoTransform(grid); mask.SetProjection(proj)
    gdal.RasterizeLayer(mask, [1], layer, options=['ATTRIBUTE=zone'])
    zones = mask.ReadAsArray(); common = zones > 0
    for path in rasters:
        ds = gdal.Open(str(path))
        if (ds.RasterYSize, ds.RasterXSize) != shape or ds.GetGeoTransform() != grid or ds.GetProjection() != proj:
            raise ValueError(f'Grid mismatch: {path}')
        values = ds.ReadAsArray(); nodata = ds.GetRasterBand(1).GetNoDataValue()
        valid = np.isfinite(values)
        if nodata is not None: valid &= values != nodata
        common &= valid
    zones[~common] = 0
    counts = np.bincount(zones.ravel(), minlength=34)[1:]
    if np.any(counts == 0): raise ValueError('Empty borough on common valid mask')
    return rows, zones, counts


def masked_means(path, zones, counts):
    values = gdal.Open(str(path)).ReadAsArray()
    sums = np.bincount(zones.ravel(), weights=values.ravel(), minlength=34)[1:]
    means = sums / counts
    return float(sums.sum() / counts.sum()), means


def building_weights(path, vector, cache):
    """Split cross-boundary buildings by intersection area, preserving each total.

    Fractions are normalized over London intersections. Buildings with no positive
    London overlap are excluded, not assigned to a nearest borough; totals and
    counts excluded from the London estimand are reported separately.
    Cache is keyed by exact geometry (.shp/.shx) and boundary content.
    """
    rows, wkt = boroughs(vector)
    key = [sha256(path), sha256(Path(path).with_suffix('.shx')),
           sha256(vector), sha256(Path(vector).with_suffix('.dbf'))]
    cache = Path(cache)
    if cache.exists():
        data = np.load(cache)
        if list(data['key']) == key:
            return data['fid'], data['zone'], data['weight'], int(data['n'])
    polygons = shapely.from_wkb([r[2] for r in rows]); tree = shapely.STRtree(polygons)
    shapely.prepare(polygons)
    ds = ogr.Open(str(path)); layer = ds.GetLayer()
    if not layer.GetSpatialRef().IsSame(osr.SpatialReference(wkt=wkt)):
        raise ValueError('Building and borough CRS differ')
    n = layer.GetFeatureCount(); all_fids=[]; all_zones=[]; all_weights=[]
    batch=[]; ids=[]
    def flush():
        if not batch: return
        geoms=shapely.from_wkb(batch)
        bad=~shapely.is_valid(geoms)
        if bad.any(): geoms[bad]=shapely.make_valid(geoms[bad])
        # Prepared borough predicates avoid millions of expensive full polygon
        # intersections. Only genuinely boundary-crossing buildings need clipping.
        ix, jx = tree.query(geoms)
        hits=shapely.intersects(polygons[jx],geoms[ix]);ix=ix[hits];jx=jx[hits]
        covered=shapely.covers(polygons[jx],geoms[ix])
        areas=np.zeros(len(ix));areas[covered]=shapely.area(geoms[ix[covered]])
        areas[~covered]=shapely.area(shapely.intersection(geoms[ix[~covered]],polygons[jx[~covered]]))
        keep=areas>0;ix=ix[keep];jx=jx[keep];areas=areas[keep]
        totals=np.bincount(ix,weights=areas,minlength=len(batch))
        if np.any(shapely.area(geoms)<=0):
            raise ValueError('Empty or zero-area building geometry')
        all_fids.append(np.asarray(ids,dtype=np.int64)[ix]);all_zones.append(jx)
        all_weights.append(areas/totals[ix]);batch.clear();ids.clear()
    for f in layer:
        geom=f.GetGeometryRef()
        if geom is None: raise ValueError(f'Empty building {f.GetFID()}')
        ids.append(f.GetFID());batch.append(bytes(geom.ExportToWkb()))
        if len(batch)==25000: flush()
        if (f.GetFID()+1)%250000==0: print(f"Building allocation: {f.GetFID()+1:,}/{n:,}",flush=True)
    flush();fid=np.concatenate(all_fids);zone=np.concatenate(all_zones);weight=np.concatenate(all_weights)
    if np.any(fid<0) or np.any(fid>=n):
        raise ValueError('Building FIDs must be sequential')
    if not np.allclose(np.bincount(fid,weights=weight,minlength=n)[np.unique(fid)],1,atol=1e-12):
        raise ValueError('Building weights do not conserve totals')
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,key=key,fid=fid,zone=zone,weight=weight,n=n)
    return fid,zone,weight,n


def allocated_energy(path, vector, cache):
    fid,zone,weight,n=building_weights(path,vector,cache)
    ds=ogr.Open(str(path));layer=ds.GetLayer()
    fields=[layer.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(layer.GetLayerDefn().GetFieldCount())]
    layer.SetIgnoredFields(['OGR_GEOMETRY']+[f for f in fields if f!='energy_sav'])
    energy=np.full(n,np.nan)
    for f in layer:
        value=f.GetField('energy_sav')
        if value is not None:energy[f.GetFID()]=float(value)
    if not np.isfinite(energy).all():raise ValueError('Missing/nonfinite building energy')
    totals=np.bincount(zone,weights=energy[fid]*weight,minlength=33)
    inside=np.unique(fid); london_total=energy[inside].sum()
    if not np.isclose(totals.sum(),london_total,rtol=1e-10,atol=1e-4):
        raise ValueError('Borough energy does not conserve citywide total')
    return float(london_total),totals,len(inside),int(len(fid)-len(inside)),n-len(inside),float(energy.sum())
