"""Join verified candidate energy/capacity for the existing Figure 4/5 scripts.

Preserve absolute model reference fields so plots compute paired differences;
energy 'savings' is a 30 C counterfactual value, not measured consumption.
Writes a fresh summary directory only; does not install production results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(p):
    with p.open() as f: return list(csv.DictReader(f))


def write(p, rows):
    with p.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    for name in ['energy', 'capacity', 'output']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--capacity-method', choices=['corrected_fixed', 'daily_tmax_sensitivity'],
                   default='corrected_fixed')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    ec = a.energy/'citywide_energy_candidates.csv'; eb = a.energy/'borough_energy_candidates.csv'
    cc = a.capacity/'citywide_capacity.csv'; cb = a.capacity/'borough_capacity.csv'
    def ekeep(r): return r['variant']=='pooled_candidate' and float(r['cooling_transfer'])==1
    def ckeep(r): return r['method']==a.capacity_method and float(r['rh_percent'])==45
    erows = [r for r in read(ec) if ekeep(r)]
    crows = [r for r in read(cc) if ckeep(r)]
    e = {r['scenario']:r for r in erows}
    c = {r['scenario']:r for r in crows}
    assert len(erows)==len(e) and len(crows)==len(c), 'Duplicate city records'
    for folder in [a.energy,a.capacity]:
        assert json.loads((folder/'manifest.json').read_text())['inputs_unchanged']
    expected = {'baseline','allbuilt','treerisk','treeopp',*[f'{s}{n}' for s in ['green','target'] for n in [10,20,30]]}
    assert set(e)==set(c)==expected
    city = [dict(scenario=s, temperature_c=25,
                 capacity_method=a.capacity_method,
                 energy_savings_unique_buildings=float(e[s]['absolute_counterfactual_gbp']),
                 energy_change_vs_baseline=float(e[s]['energy_change_gbp']),
                 energy_change_kwh=float(e[s]['energy_change_kwh']),
                 workability_fraction=float(c[s]['capacity_fraction']),
                 productivity_change_vs_baseline_pp=float(c[s]['change_pp'])) for s in sorted(expected)]
    berows = [r for r in read(eb) if ekeep(r)]
    bcrows = [r for r in read(cb) if ckeep(r)]
    be = {(r['scenario'],r['borough_code']):r for r in berows}
    bc = {(r['scenario'],r['borough_code']):r for r in bcrows}
    assert len(berows)==len(be) and len(bcrows)==len(bc), 'Duplicate borough records'
    assert set(be)==set(bc) and len(be)==330
    borough_codes = {b for s,b in be if s=='baseline'}
    assert len(borough_codes)==33
    for s in expected:
        assert {b for sc,b in be if sc==s}==borough_codes
        assert abs(float(e[s]['absolute_counterfactual_gbp'])-float(e['baseline']['absolute_counterfactual_gbp'])-float(e[s]['energy_change_gbp']))<.001
        assert abs((float(c[s]['capacity_fraction'])-float(c['baseline']['capacity_fraction']))*100-float(c[s]['change_pp']))<1e-9
        for b in borough_codes:
            assert abs(float(be[s,b]['absolute_counterfactual_gbp'])-float(be['baseline',b]['absolute_counterfactual_gbp'])-float(be[s,b]['energy_change_gbp']))<.001
            assert abs((float(bc[s,b]['capacity_fraction'])-float(bc['baseline',b]['capacity_fraction']))*100-float(bc[s,b]['change_pp']))<1e-9
    borough = [dict(scenario=s, temperature_c=25, capacity_method=a.capacity_method,
                    borough_code=b, borough=be[(s,b)]['borough'],
                    energy_savings=float(be[(s,b)]['absolute_counterfactual_gbp']),
                    energy_change_gbp=float(be[(s,b)]['energy_change_gbp']),
                    workability_fraction=float(bc[(s,b)]['capacity_fraction']),
                    valid_cells=int(bc[(s,b)]['valid_cells'])) for s,b in sorted(be)]
    for row in city:
        bs = [r for r in borough if r['scenario']==row['scenario']]
        assert abs(sum(r['energy_change_gbp'] for r in bs)-row['energy_change_vs_baseline'])<.001
        avg = sum(r['workability_fraction']*r['valid_cells'] for r in bs)/sum(r['valid_cells'] for r in bs)
        assert abs(avg-row['workability_fraction'])<1e-10
    sources = [ec,eb,cc,cb,a.energy/'manifest.json',a.capacity/'manifest.json',Path(__file__)]
    manifest = {'status':'candidate only; not adopted production',
                'energy':'London-scaled pooled, full assumed cooling transfer, conditional30day proxy',
                'capacity':('JJA daily-Tmax-based RH45, equal 92 days x 15 members; '
                            'not average working-day capacity' if a.capacity_method=='daily_tmax_sensitivity'
                            else 'corrected fixed25/5C RH45; not average working-day capacity'),
                'capacity_method':a.capacity_method,
                'sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}}
    a.output.mkdir(parents=True)
    write(a.output/'citywide_energy_productivity_summary.csv',city)
    write(a.output/'borough_energy_productivity_summary.csv',borough)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(a.output)


if __name__=='__main__': main()
