"""Equal-borough Figure 3, separating executed from reconstructed conditions.

For unchanged HM, masks and mixing kernel, T = t_ref + uhi_max * H.
Use actual 25/5 and 28/5 outputs; validate the +3 translation before deriving
22/5 and baseline 22/2. These derivatives are not new InVEST executions.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from osgeo import gdal
from valuation_accounting import london_grid,masked_means,sha256
from summarize_figure3_temperature import SCENARIOS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production-root',type=Path,required=True)
    p.add_argument('--borough-vector',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    source={};metadata={}
    for s in SCENARIOS:
        for t in [25,28]:
            candidates=[f for stage in ['03_energy_work_capacity','01_temperature_invest3202']
                for f in (a.production_root/stage/s).glob(f'T_air_*{t}deg_5uhi*.tif')]
            assert candidates,(s,t)
            assert len({sha256(f) for f in candidates})==1
            f=candidates[0];source[s,t]=f
            m=f.with_name(f.name.replace('T_air_','run_manifest_')).with_suffix('.json')
            metadata[s,t]=json.loads(m.read_text())['model_arguments']
            assert metadata[s,t]['t_ref']==t and metadata[s,t]['uhi_max']==5
        # No hidden change in land cover, ET or cooling parameters between runs.
        for k in ['lulc_raster_path','ref_eto_raster_path','biophysical_table_path',
                  'cc_method','green_area_cooling_distance','t_air_average_radius']:
            assert metadata[s,25][k]==metadata[s,28][k],(s,k)
    rows,zones,counts=london_grid(a.borough_vector,list(source.values()))
    means={key:masked_means(f,zones,counts)[1] for key,f in source.items()}
    errors={}
    for s in SCENARIOS:
        x=gdal.Open(str(source[s,25])).ReadAsArray()
        y=gdal.Open(str(source[s,28])).ReadAsArray()
        error=float(np.max(np.abs((y[zones>0].astype(float)-x[zones>0])-3)))
        assert error<1e-4,(s,error)
        errors[s]=error
    ref=22+(.4*(means['baseline',25]-25))
    records=[];borough=[]
    records.append(dict(scenario='baseline',condition='22/2',mean_change_c=0.,origin='linear reconstruction'))
    for t in [22,25,28]:
        for s in SCENARIOS:
            vals=means[s,25]-3 if t==22 else means[s,t]
            records.append(dict(scenario=s,condition=f'{t}/5',mean_change_c=float(np.mean(vals-ref)),
                origin='linear reconstruction' if t==22 else 'executed output'))
    for s in SCENARIOS[1:]:
        for row,v in zip(rows,means[s,25]-means['baseline',25]):
            borough.append(dict(scenario=s,borough_code=row[0],borough=row[1],change_c=float(v)))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    stem='figure3_main_v01_20261111'
    for suffix,data in [('conditions',records),('borough',borough)]:
        with (a.output_dir/f'{stem}_{suffix}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    manifest=dict(script=str(Path(__file__).resolve()),script_sha256=sha256(__file__),
        sources={f'{s}_{t}':dict(path=str(f),sha256=sha256(f)) for (s,t),f in source.items()},
        common_cells=int(counts.sum()),boroughs=33,translation_max_error_c=errors,
        reference='baseline land use; reconstructed t_ref=22, uhi_max=2',
        estimand='equal-weight mean of 33 borough means; no uncertainty interval',
        reconstruction='T22_5=T25_5-3; baseline T22_2=22+0.4*(baseline T25_5-25)')
    (a.output_dir/f'{stem}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(common_cells=int(counts.sum()),max_error=max(errors.values())),indent=2))

if __name__=='__main__':main()
