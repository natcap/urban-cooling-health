"""Summarize existing matched-condition temperature outputs; never run InVEST."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from valuation_accounting import london_grid, masked_means, sha256

SCENARIOS = ['baseline', 'allbuilt', 'treerisk', 'treeopp', 'green10',
             'target10', 'green20', 'target20', 'green30', 'target30']
STEM = 'figure3_alternative_area_weighted_v01_20261111'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-root', type=Path, required=True)
    parser.add_argument('--borough-vector', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    signatures = []
    for scenario in SCENARIOS:
        # Prefer the valuation stage; otherwise use the same-version temperature stage.
        candidates = [p for stage in ['03_energy_work_capacity', '01_temperature_invest3202']
                      for p in (args.production_root / stage / scenario).glob('T_air_*25deg_5uhi*.tif')]
        if not candidates:
            raise ValueError(f'Missing primary temperature raster: {scenario}')
        path = candidates[0]
        if len({sha256(p) for p in candidates}) != 1:
            raise ValueError(f'Conflicting duplicate temperature sources: {scenario}')
        run = path.with_name(path.name.replace('T_air_', 'run_manifest_')).with_suffix('.json')
        metadata = json.loads(run.read_text())
        model = metadata['model_arguments']
        assert model['t_ref'] == 25 and model['uhi_max'] == 5
        signature = {k: model[k] for k in ['t_ref', 'uhi_max', 'avg_rel_humidity',
                     'cc_method', 'green_area_cooling_distance', 't_air_average_radius']}
        signatures.append(signature)
        # Record current files alongside the original run manifest; do not rewrite history.
        sources[scenario] = {'path': str(path.resolve()), 'sha256': sha256(path),
                             'run_manifest': str(run.resolve()), 'run_sha256': sha256(run)}
    assert all(s == signatures[0] for s in signatures)
    rasters = [sources[s]['path'] for s in SCENARIOS]
    rows, zones, counts = london_grid(args.borough_vector, rasters)
    assert len(rows) == 33 and len({r[0] for r in rows}) == 33
    assert np.all(counts > 0)
    city_base, borough_base = masked_means(rasters[0], zones, counts)
    city, borough = [], []
    for s in SCENARIOS[1:]:
        mean, local = masked_means(sources[s]['path'], zones, counts)
        delta = local - borough_base
        weighted = float(np.average(delta, weights=counts))
        assert abs(weighted - (mean - city_base)) < 1e-10
        city.append(dict(scenario=s, change_c=mean-city_base,
                         equal_borough_mean_c=float(delta.mean()),
                         min_borough_c=float(delta.min()), max_borough_c=float(delta.max()),
                         valid_london_cells=int(counts.sum())))
        borough.extend(dict(scenario=s, borough_code=r[0], borough=r[1],
                            change_c=float(d), valid_cells=int(n))
                       for r, d, n in zip(rows, delta, counts))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, records in [('citywide', city), ('borough', borough)]:
        path = args.output_dir / f'{STEM}_{name}.csv'
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=records[0]); w.writeheader(); w.writerows(records)
        outputs[name] = {'path': str(path.resolve()), 'sha256': sha256(path)}
    manifest = dict(generator=str(Path(__file__).resolve()), generator_sha256=sha256(__file__),
        inputs=sources, settings=signatures[0],
        boundary={ext: sha256(args.borough_vector.with_suffix(ext)) for ext in ['.shp','.shx','.dbf','.prj']},
        boundary_path=str(args.borough_vector.resolve()),
        common_valid_cells=int(counts.sum()), borough_count=len(rows),
        estimand='Scenario-minus-baseline temperature on shared London cell-centre mask',
        checks='Nine scenarios; 33 unique boroughs; common grid; weighted borough means equal city means',
        outputs=outputs)
    (args.output_dir / f'{STEM}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(city, indent=2))


if __name__ == '__main__':
    main()
