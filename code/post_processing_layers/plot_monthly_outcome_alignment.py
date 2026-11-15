"""Visual diagnostic of monthly representativeness; no manuscript replacement.

Show daily-normalized health, mean daily-Tmax capacity and climate separately.
JJA is a 92-day weighted mean, not a fourth month. Energy has no month-specific
exposure calculation, so an invented flat monthly energy curve is not plotted.
"""
import argparse,json,hashlib,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);a=ap.parse_args()
    png=a.run/'monthly_outcome_alignment_20261113_v1.png'
    if png.exists():raise FileExistsError(png)
    cp=a.run/'capacity/monthly_capacity_summary.csv';hp=a.run/'health/monthly_health_summary.csv'
    c=pd.read_csv(cp);h=pd.read_csv(hp).query("cause=='all_cause'")
    periods=['June','July','August','JJA'];scenarios=['green30','target30']
    assert not c.duplicated(['period','scenario']).any() and not h.duplicated(['period','scenario']).any()
    palette=Path(__file__).resolve().parents[1]/'func_colors.R'
    text=palette.read_text();colors={s:re.search(r'"'+s+r'"\s*=\s*"(#[A-Fa-f0-9]+)"',text).group(1) for s in scenarios}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    def panel(ax,values,color,marker,label,low=None,high=None):
        ax.plot(range(3),values[:3],color=color,marker=marker,label=label,lw=1.8)
        ax.scatter([3],[values[3]],color=color,marker=marker,zorder=3)
        if low is not None:
            ax.fill_between(range(3),low[:3],high[:3],color=color,alpha=.10,linewidth=0)
            ax.vlines(3,low[3],high[3],color=color,alpha=.35,lw=1)
    bc=c.query("scenario=='baseline'").set_index('period').loc[periods]
    panel(axes[0],bc.baseline_tmax_c.to_numpy(),'#444444','o','Baseline',bc.baseline_tmax_c_min.to_numpy(),bc.baseline_tmax_c_max.to_numpy())
    for s,marker in zip(scenarios,['o','^']):
        v=c.query('scenario==@s').set_index('period').loc[periods]
        panel(axes[1],v.change_pp.to_numpy(),colors[s],marker,s.title(),v.change_pp_min.to_numpy(),v.change_pp_max.to_numpy())
        v=h.query('scenario==@s').set_index('period').loc[periods]
        panel(axes[2],v.deaths_averted_per_day.to_numpy(),colors[s],marker,s.title(),v.member_min_per_day.to_numpy(),v.member_max_per_day.to_numpy())
    titles=['a  Reconstructed baseline climate','b  Daily-Tmax-based capacity gain','c  All-cause deaths averted per day']
    ys=['Mean daily Tmax (°C)','Percentage points (RH45)','Deaths per day']
    for ax,title,y in zip(axes,titles,ys):
        ax.set_title(title,loc='left',fontweight='bold',fontsize=11,pad=12);ax.set_ylabel(y)
        ax.set_xticks(range(4),['June','July','August','JJA\nmean']);ax.set_xlim(-.2,3.3)
        ax.axvspan(2.65,3.3,color='grey',alpha=.05);ax.grid(axis='y',alpha=.15)
    for ax in axes[1:]:ax.set_ylim(bottom=0)
    axes[1].legend(frameon=False,fontsize=9,loc='upper left')
    fig.text(.05,.065,'Means across 3 GCMs × 5 weather patterns; shading/ranges show member minima–maxima, not confidence intervals.',fontsize=9)
    fig.text(.05,.025,'Energy is not plotted: the conditional 30-day calculation does not estimate monthly exposure. Fixed spatial cooling is retained throughout.',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(png,dpi=250,facecolor='white');plt.close(fig)
    joined=[]
    for s in scenarios:
        cv=c.query('scenario==@s').set_index('period');hv=h.query('scenario==@s').set_index('period')
        for p in periods:
            joined.append(dict(scenario=s,period=p,days=int(hv.loc[p,'days']),baseline_tmax_c=bc.loc[p,'baseline_tmax_c'],capacity_gain_pp=cv.loc[p,'change_pp'],deaths_averted=hv.loc[p,'deaths_averted'],deaths_averted_per_day=hv.loc[p,'deaths_averted_per_day'],health_daily_vs_jja_pct=100*(hv.loc[p,'deaths_averted_per_day']/hv.loc['JJA','deaths_averted_per_day']-1),capacity_vs_jja_pct=100*(cv.loc[p,'change_pp']/cv.loc['JJA','change_pp']-1)))
    pd.DataFrame(joined).to_csv(a.run/'monthly_outcome_comparison.csv',index=False)
    sources=[cp,hp,palette,Path(__file__)]
    (a.run/'plot_manifest.json').write_text(json.dumps({'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'png':str(png),'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'uncertainty':'member range, not CI','support':'capacity15,733,893 cells; health14,258,588; constant within each outcome'},indent=2))
    print(png)


if __name__=='__main__':main()
