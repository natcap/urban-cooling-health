"""Read ONS ad hoc 2155; print England-only 2021 seasonal cause fractions.

Usage: python inspect_ons_monthly_causes.py monthlycausefinal.xlsx
No model inputs are written. These occurrence fractions are NOT London counts.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
import openpyxl

p = Path(sys.argv[1])
w = openpyxl.load_workbook(p, read_only=True, data_only=True)
totals = collections.defaultdict(lambda: [0, 0, 0])
seen = set()
for row in w['1'].iter_rows(min_row=6, values_only=True):
    year, month, code, *ages = row
    if year != 2021:
        continue
    assert code == 'Neonate' or re.fullmatch(r'[A-Z][0-9]{2}', code), code
    assert (month, code) not in seen
    seen.add((month, code))
    assert all(isinstance(x, (int, float)) and x >= 0 for x in ages)
    count = sum(ages)
    groups = []
    # Neonates have a separate non-ICD category; include in overall only.
    if code == 'Neonate' or 'A00' <= code <= 'R99' or 'U00' <= code <= 'Y89': groups.append('all_cause')
    for name, lo, hi in [('mental_disorder','F00','F99'),('cardiovascular','I00','I99'),
                         ('respiratory','J00','J99'),('self_harm','X60','X84')]:
        if lo <= code <= hi: groups.append(name)
    for group in groups:
        totals[group][0] += count
        if month in ['June', 'July', 'August']: totals[group][1] += count
        if month == 'July': totals[group][2] += count
assert len({m for m,c in seen}) == 12
print(json.dumps({'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'geography':'England only; occurrence month; all persons, all ages',
    'results':{k:dict(annual=v[0],summer=v[1],july=v[2],summer_share=v[1]/v[0],july_share=v[2]/v[0])
               for k,v in totals.items()}}, indent=2))
