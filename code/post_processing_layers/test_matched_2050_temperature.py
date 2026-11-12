"""Temperature-only mean matching; analytical translation, not a new InVEST run.

Validate the translation against executed 25/5 and 28/5 rasters before using it.
Match Figure 1's equal-borough mean, not its spatial field or rural temperature.
Never change production rasters or calculate downstream valuations.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from osgeo import gdal
from valuation_accounting import london_grid, masked_means, sha256
from summarize_figure3_temperature import SCENARIOS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    # Refuse to overwrite any earlier diagnostic, including partial results.
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    inputs = a.data_root/'1_preprocess/UrbanCoolingModel/OfficialWorkingInputs'
    prod = a.data_root/'2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09'
    climate = inputs/'climate/worldclim_v21_historical/comparison/figure1_baseline_comparison_20261111_v1_by_model.csv'
    with climate.open() as f:
        records = list(csv.DictReader(f))
    targets = {}
    for ssp in ['ssp370', 'ssp585']:
        r = [r for r in records if r['ssp'] == ssp and r['period'] == '2041-2060']
        assert len(r) == 99 and len({(v['borough'], v['model']) for v in r}) == 99
        targets[ssp] = float(np.mean([float(v['future']) for v in r]))
    sources, arguments = {}, {}
    for s in SCENARIOS:
        for t in [25, 28]:
            candidates = [f for stage in ['03_energy_work_capacity', '01_temperature_invest3202']
                          for f in (prod/stage/s).glob(f'T_air_*{t}deg_5uhi*.tif')]
            assert candidates and len({sha256(f) for f in candidates}) == 1
            f = candidates[0]
            sources[s, t] = f
            manifest = f.with_name(f.name.replace('T_air_', 'run_manifest_')).with_suffix('.json')
            arguments[s, t] = json.loads(manifest.read_text())['model_arguments']
            assert arguments[s, t]['t_ref'] == t and arguments[s, t]['uhi_max'] == 5
        for key in ['lulc_raster_path', 'ref_eto_raster_path', 'biophysical_table_path',
                    'cc_method', 'green_area_cooling_distance', 't_air_average_radius']:
            assert arguments[s, 25][key] == arguments[s, 28][key]
    boundary = inputs/'AOIs/London_Borough_aoi.shp'
    rows, zones, counts = london_grid(boundary, list(sources.values()))
    means, errors = {}, {}
    for s in SCENARIOS:
        means[s] = masked_means(sources[s, 25], zones, counts)
        x = gdal.Open(str(sources[s, 25])).ReadAsArray()[zones > 0].astype(float)
        y = gdal.Open(str(sources[s, 28])).ReadAsArray()[zones > 0].astype(float)
        errors[s] = float(np.max(np.abs(y-x-3)))
        assert errors[s] < 1e-4, (s, errors[s])
    base_city, base_borough = means['baseline']
    base = float(base_borough.mean())
    summary, borough = [], []
    for label, target in [('existing_25_5', base), *targets.items()]:
        shift = target-base
        for s in SCENARIOS:
            city, local = means[s]
            matched = local+shift
            assert np.allclose(matched-(base_borough+shift), local-base_borough, atol=1e-12)
            summary.append(dict(configuration=label, scenario=s, t_ref_c=25+shift, uhi_max_c=5,
                shift_from_existing_c=shift, equal_borough_mean_c=float(matched.mean()),
                area_weighted_mean_c=city+shift,
                equal_borough_land_use_change_c=float((local-base_borough).mean()),
                area_weighted_land_use_change_c=city-base_city))
            for row, old, new in zip(rows, local, matched):
                borough.append(dict(configuration=label, scenario=s, borough_code=row[0],
                    borough=row[1], existing_temperature_c=float(old), matched_temperature_c=float(new)))
    a.output_dir.mkdir(parents=True)
    for name, data in [('summary', summary), ('borough', borough)]:
        with (a.output_dir/f'{name}.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=data[0]); w.writeheader(); w.writerows(data)
    evidence = dict(method='Validated analytical translation; no new InVEST execution',
        generator=str(Path(__file__).resolve()), generator_sha256=sha256(__file__),
        targets=targets, existing_equal_borough_baseline_c=base,
        matched_t_ref_c={k:25+v-base for k,v in targets.items()},
        translation_max_error_c=errors, common_london_cells=int(counts.sum()),
        climate_source=dict(path=str(climate), sha256=sha256(climate)),
        sources={f'{s}_{t}':dict(path=str(f),sha256=sha256(f)) for (s,t),f in sources.items()},
        boundary={ext:sha256(boundary.with_suffix(ext)) for ext in ['.shp','.shx','.dbf','.prj']},
        held='Health, work capacity and energy; all existing outputs unchanged')
    (a.output_dir/'manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in ['targets','existing_equal_borough_baseline_c','matched_t_ref_c','translation_max_error_c','common_london_cells']},indent=2))


if __name__ == '__main__':
    main()
