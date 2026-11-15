"""Read existing Green30 diagnostic stages; print comparisons without writes."""
import argparse
from pathlib import Path
import numpy as np
from osgeo import gdal

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic_root', type=Path)
    args = parser.parse_args()
    gdal.UseExceptions()
    folders = [args.diagnostic_root / name for name in
               ['green30_health_invest3202', 'green30_health_invest3202_nodata_harmonized']]
    for stage in ['lulc', 'ref_eto', 'shade', 'albedo', 'kc', 'green_area',
                  'eti', 'cc', 'cc_masked_green_areas', 'cc_park',
                  'green_area_sum', 'hm', 'T_air_nomix', 'T_air']:
        datasets = []
        for folder in folders:
            paths = list(folder.rglob(stage + '_london_green30_25deg_5uhi_45hum.tif'))
            if len(paths) != 1:
                raise ValueError(f'Expected one {stage} raster in {folder}: {paths}')
            datasets.append(gdal.Open(str(paths[0])))
        a, b = [ds.ReadAsArray() for ds in datasets]
        if (datasets[0].GetGeoTransform() != datasets[1].GetGeoTransform()
                or datasets[0].GetProjection() != datasets[1].GetProjection()
                or a.shape != b.shape):
            raise ValueError(f'Grid mismatch: {stage}')
        na, nb = [ds.GetRasterBand(1).GetNoDataValue() for ds in datasets]
        va, vb = np.isfinite(a) & (a != na), np.isfinite(b) & (b != nb)
        common = va & vb
        difference = b[common].astype(float) - a[common].astype(float)
        print(stage, 'nodata', na, nb, 'valid', int(va.sum()), int(vb.sum()),
              'maskdiff', int(np.count_nonzero(va != vb)),
              'changed', int(np.count_nonzero(difference)),
              'mean', difference.mean(), 'maxabs', np.max(np.abs(difference)), flush=True)

if __name__ == '__main__':
    main()
