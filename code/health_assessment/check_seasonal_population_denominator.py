"""Sensitivity to retaining old LSOA denominator versus matched health-mask population."""
import argparse
from pathlib import Path
import pandas as pd
from osgeo import ogr

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    out=a.run/'population_denominator_sensitivity.csv';assert not out.exists()
    metrics=['pIncomeDeprivation','p5under','p75over','pBAME','overall_score']
    ds=ogr.Open('data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg')
    v=pd.DataFrame([dict(LSOA11CD=f.GetField('LSOA11CD'),**{m:f.GetField(m) for m in metrics}) for f in ds.GetLayer()])
    x=pd.read_csv(a.run/'lsoa_results.csv').query("reconstruction=='worldclim_aligned' and ssp=='ssp370' and rr_variant=='central' and cause=='all_cause' and scenario in ['green30','target30']").pivot(index='LSOA11CD',columns='scenario',values='deaths_averted').reset_index()
    x=x.merge(pd.read_csv(a.run/'population_coverage.csv'),on='LSOA11CD',validate='one_to_one').merge(v,on='LSOA11CD',validate='one_to_one');assert len(x)==4835
    rows=[]
    for denominator in ['population_2021','population_on_health_mask']:
        assert (x[denominator]>0).all()
        g=1e6*x.green30/x[denominator];t=1e6*x.target30/x[denominator];cut=g.quantile(1/3)
        for metric in metrics:
            high=x[metric]>x[metric].quantile(2/3)
            rows.append(dict(denominator=denominator,metric=metric,low_green=int(((g<=cut)&high).sum()),low_target=int(((t<=cut)&high).sum()),pct_green=100*(g[high]<=cut).mean(),pct_target=100*(t[high]<=cut).mean()))
    pd.DataFrame(rows).to_csv(out,index=False);print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
