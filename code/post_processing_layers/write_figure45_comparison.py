#!/usr/bin/env python3
"""Export the citywide/equal-borough review table with accounting assertions."""
import argparse
import csv
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('summary',type=Path)
    root=parser.parse_args().summary
    def read(name):return list(csv.DictReader((root/name).open()))
    city={r['scenario']:r for r in read('citywide_energy_productivity_summary.csv')}
    borough=read('borough_energy_productivity_summary.csv')
    base={r['borough_code']:r for r in borough if r['scenario']=='baseline'}
    health={r['scenario']:float(r['estimate']) for r in read('figure4_citywide_cobenefits_data.csv') if r['metric'].startswith('c ')}
    rows=[]
    order=['allbuilt','treerisk','treeopp','green10','target10','green20','target20','green30','target30']
    for s in order:
        c=city[s];b=[r for r in borough if r['scenario']==s]
        assert len(b)==len(base)==33
        energy=[float(r['energy_savings'])-float(base[r['borough_code']]['energy_savings']) for r in b]
        capacity=[100*(float(r['workability_fraction'])-float(base[r['borough_code']]['workability_fraction'])) for r in b]
        weights=[int(r['valid_pixels']) for r in b]
        total=float(c['energy_change_vs_baseline']);cap=float(c['productivity_change_vs_baseline_pp'])
        assert abs(sum(energy)-total)<0.01
        assert abs(sum(v*w for v,w in zip(capacity,weights))/sum(weights)-cap)<1e-9
        rows.append(dict(scenario=s,energy_citywide_million_gbp_month=total/1e6,
            energy_mean_borough_million_gbp_month=sum(energy)/33/1e6,
            capacity_full_model_domain_pp=float(c['productivity_full_domain_change_pp']),
            capacity_london_area_weighted_pp=cap,capacity_equal_borough_pp=sum(capacity)/33,
            annual_deaths_averted=health[s]))
    with (root/'figure45_aggregation_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    text=['# Figure 4–5 aggregation comparison','',
      'Primary: London-boundary area-weighted capacity and citywide energy/health totals. Equal-borough summaries are supplementary. Full-model-domain capacity is superseded, retained here only to explain the correction. All capacity changes are percentage points, not relative percent. Energy is £ million/month; borough means are per borough. Health uses fixed 2021 population/mortality.','',
      '| Scenario | Energy: city | Energy: mean borough | Capacity: old full domain | Capacity: London | Capacity: mean borough | Annual deaths averted |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append('| '+r['scenario']+' | '+' | '.join(f'{v:.3f}' for k,v in r.items() if k!='scenario')+' |')
    text+=['','Checks: all nine energy changes sum from boroughs to the London total (within £0.01); cell-count-weighted borough capacity changes reproduce the London mean (within 1e-9 percentage points). Neither capacity summary is worker-weighted.','',
      'Reproduce with `code/post_processing_layers/write_figure45_comparison.py <summary-folder>` after regenerating both figures.']
    (root/'figure45_aggregation_comparison.md').write_text('\n'.join(text)+'\n')
    print('Comparison CSV/Markdown written; conservation checks passed')


if __name__=='__main__':main()
