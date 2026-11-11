"""Synthetic boundary, pairing and building-total conservation checks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from osgeo import gdal, ogr, osr
from shapely.geometry import box
from code.post_processing_layers.valuation_accounting import london_grid, masked_means, allocated_energy


def vector(path, shapes, energy=False):
    ds=ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(str(path))
    srs=osr.SpatialReference();srs.ImportFromEPSG(27700)
    layer=ds.CreateLayer(path.stem,srs,ogr.wkbPolygon)
    for field in (['energy_sav'] if energy else ['GSS_CODE','NAME']):
        layer.CreateField(ogr.FieldDefn(field,ogr.OFTReal if energy else ogr.OFTString))
    for i,shape in enumerate(shapes):
        f=ogr.Feature(layer.GetLayerDefn());f.SetGeometry(ogr.CreateGeometryFromWkb(shape.wkb))
        if energy:f.SetField('energy_sav',[100.,20.,70.,30.][i])
        else:f.SetField('GSS_CODE',f'B{i:02d}');f.SetField('NAME',f'Borough {i}')
        layer.CreateFeature(f)
    ds=None


class AccountingTest(unittest.TestCase):
    def test_boundary_and_conservation(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);borough=r/'borough.shp'
            vector(borough,[box(i,0,i+1,10) for i in range(33)])
            rasters=[]
            for name,value in [('base',0.5),('scenario',0.6)]:
                p=r/(name+'.tif');ds=gdal.GetDriverByName('GTiff').Create(str(p),36,10,1,gdal.GDT_Float32)
                ds.SetGeoTransform((0,1,0,10,0,-1));srs=osr.SpatialReference();srs.ImportFromEPSG(27700);ds.SetProjection(srs.ExportToWkt())
                ds.GetRasterBand(1).WriteArray(np.full((10,36),value));ds=None;rasters.append(p)
            rows,zones,counts=london_grid(borough,rasters)
            self.assertEqual(int(counts.sum()),330)
            mean,means=masked_means(rasters[1],zones,counts)
            self.assertAlmostEqual(mean,0.6,places=6)
            self.assertAlmostEqual(np.average(means,weights=counts),mean)
            buildings=r/'buildings.shp'
            vector(buildings,[box(.5,1,1.5,2),box(40,1,41,2),box(2.1,1,2.5,2),box(32.5,1,33.5,2)],True)
            total,allocated,n,extra,outside,full=allocated_energy(buildings,borough,r/'cache.npz')
            self.assertEqual((n,extra,outside),(3,1,1))
            self.assertAlmostEqual(total,200);self.assertAlmostEqual(full,220)
            self.assertAlmostEqual(allocated.sum(),total)
            np.testing.assert_allclose(allocated[:3],[50,50,70])
            self.assertAlmostEqual(allocated[-1],30)
            self.assertEqual(allocated_energy(buildings,borough,r/'cache.npz')[0],total)


if __name__=='__main__':unittest.main()
