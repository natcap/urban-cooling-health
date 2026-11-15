"""Check candidate joins, paired differences, and refusal to overwrite."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CandidateSummaryTest(unittest.TestCase):
    def test_join_and_preservation(self):
        script = Path(__file__).resolve().parents[1] / 'prepare_candidate_valuation_summary.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            energy, capacity = root/'energy', root/'capacity'
            for folder in (energy, capacity):
                folder.mkdir()
                (folder/'manifest.json').write_text(json.dumps({'inputs_unchanged': True}))
            scenarios = ['baseline','allbuilt','treerisk','treeopp'] + [f'{s}{n}' for s in ['green','target'] for n in [10,20,30]]
            ec, eb, cc, cb = [], [], [], []
            for i, scenario in enumerate(scenarios):
                change = -i if i < 3 else i
                erow = dict(scenario=scenario, variant='pooled_candidate', cooling_transfer=1,
                    absolute_counterfactual_gbp=3300+change*33, energy_change_gbp=change*33, energy_change_kwh=change*66)
                crow = dict(scenario=scenario, method='corrected_fixed', rh_percent=45,
                    capacity_fraction=.5+change/100, change_pp=change)
                daily = dict(crow, method='daily_tmax_sensitivity',
                    capacity_fraction=.7+change/200, change_pp=change/2)
                ec.append(erow); cc.extend([crow,daily])
                for b in range(33):
                    eb.append(dict(erow, borough_code=str(b), borough=f'B{b}',
                        absolute_counterfactual_gbp=100+change, energy_change_gbp=change))
                    cb.extend([dict(crow, borough_code=str(b), valid_cells=100),
                               dict(daily, borough_code=str(b), valid_cells=100)])
            for path, rows in [(energy/'citywide_energy_candidates.csv',ec),(energy/'borough_energy_candidates.csv',eb),
                               (capacity/'citywide_capacity.csv',cc),(capacity/'borough_capacity.csv',cb)]:
                with path.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            command=[sys.executable,str(script),'--energy',str(energy),'--capacity',str(capacity),'--output',str(root/'summary')]
            subprocess.run(command,check=True,capture_output=True)
            output=root/'summary'/'citywide_energy_productivity_summary.csv'
            before=output.read_bytes()
            with output.open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),10)
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,output.read_bytes())
            daily_command=command[:-1]+[str(root/'daily_summary'),'--capacity-method','daily_tmax_sensitivity']
            subprocess.run(daily_command,check=True,capture_output=True)
            with (root/'daily_summary/citywide_energy_productivity_summary.csv').open() as f:
                daily_rows=list(csv.DictReader(f))
            self.assertTrue(all(r['capacity_method']=='daily_tmax_sensitivity' for r in daily_rows))
            self.assertAlmostEqual(float(next(r for r in daily_rows if r['scenario']=='target30')['productivity_change_vs_baseline_pp']),4.5)


if __name__=='__main__': unittest.main()
