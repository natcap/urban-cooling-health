"""Isolated 25/5 C sentinel-invariance diagnostic for selected scenarios.

Never writes to source inputs or production workspaces. Requires a new output
directory; removes only the declared NoData key from a private lookup copy.
"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
from osgeo import gdal
import natcap.invest
import natcap.invest.urban_cooling_model as ucm

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('production', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--scenarios', nargs='+', choices=['baseline','green30','target30'],
                   default=['baseline','green30'])
    a = p.parse_args()
    if natcap.invest.__version__ != '3.20.2':
        raise ValueError('Requires InVEST 3.20.2')
    a.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=a.output/'diagnostic.log', level=logging.INFO)
    records=[]
    for scenario in a.scenarios:
        folder=a.production/scenario
        manifests=list(folder.glob('run_manifest_*25deg_5uhi*.json'))
        assert len(manifests)==1
        manifest=json.loads(manifests[0].read_text())
        args=manifest['model_arguments'].copy()
        source=Path(args['lulc_raster_path'])
        table=Path(args['biophysical_table_path'])
        source_hash,table_hash=sha(source),sha(table)
        ds=gdal.Open(str(source)); arr=ds.ReadAsArray()
        nd=ds.GetRasterBand(1).GetNoDataValue(); valid=arr!=nd
        with table.open(encoding='utf-8-sig') as f:
            reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
        for sentinel in [0,255]:
            assert not np.any(arr[valid]==sentinel), 'Sentinel collides with real class'
            work=a.output/f'{scenario}_nodata{sentinel}'
            work.mkdir()
            lulc=work/'input.tif'
            copy=gdal.GetDriverByName('GTiff').CreateCopy(str(lulc),ds,options=['COMPRESS=LZW'])
            out=arr.copy(); out[~valid]=sentinel
            copy.GetRasterBand(1).WriteArray(out)
            copy.GetRasterBand(1).SetNoDataValue(sentinel); copy=None
            private_table=work/'biophysical.csv'
            with private_table.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                writer.writerows(r for r in rows if float(r['lucode']) not in [nd,sentinel])
            run=args.copy();run.update(workspace_dir=str(work),lulc_raster_path=str(lulc),
                biophysical_table_path=str(private_table),results_suffix='test',n_workers=-1,
                do_energy_valuation=False,do_productivity_valuation=False)
            (work/'arguments.json').write_text(json.dumps(run,indent=2))
            print('RUN',scenario,sentinel,flush=True)
            ucm.execute(run)
            records.append(dict(scenario=scenario,sentinel=sentinel,source=str(source),
                source_sha256=source_hash,table_sha256=table_hash,workspace=str(work)))
            assert sha(source)==source_hash and sha(table)==table_hash
            (a.output/'manifest.json').write_text(json.dumps(records,indent=2))
            print('DONE',scenario,sentinel,flush=True)

if __name__=='__main__': main()
