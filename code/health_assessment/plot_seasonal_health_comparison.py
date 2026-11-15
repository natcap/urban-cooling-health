"""Diagnostic PNGs only, in a new health-run subfolder; no manuscript figures changed."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path)
    ap.add_argument('--refresh',action='store_true',help='Regenerate the two diagnostic PNGs only.')
    a=ap.parse_args()
    out=a.run/'comparison_figures';out.mkdir(exist_ok=a.refresh)
    eq=a.run/'equity_comparison';x=pd.read_csv(eq/'gam_curves.csv')
    metrics=['pIncomeDeprivation','p5under','p75over','pBAME','SVI'];labels=['Income deprivation','Age <5','Age 75+','Ethnic-minority share','Overall SVI']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,5,figsize=(14,6.4),sharex=True)
    for row,scenario in enumerate(['green30','target30']):
        for col,(metric,label) in enumerate(zip(metrics,labels)):
            ax=axes[row,col]
            for version,color,ls,title in [('previous_annual','#777777','--','Previous annual'),('revised_summer','#0072B2','-','Revised summer')]:
                z=x.query('scenario==@scenario and metric==@metric and version==@version')
                ax.plot(z.x,z.relative_to_mean,color=color,linestyle=ls,label=title)
            ax.axhline(1,color='.75',lw=.6);ax.axvline(0,color='.8',lw=.6)
            if row==0:ax.set_title(label)
            if col==0:ax.set_ylabel(scenario.title()+'\nPrediction / mean LSOA rate')
            if row==1:ax.set_xlabel('Vulnerability (SD)')
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Curve-shape comparison: magnitude removed within each version  |  v1',fontsize=13)
    fig.text(.5,.015,'Full GAM predictions including intercept; all 4,835 LSOAs fitted; P1–P99 displayed. No spatial adjustment.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(out/'figure6_shape_comparison_v1.png',dpi=180);plt.close(fig)
    z=pd.read_csv(eq/'figure7_summary.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharex=True,sharey=True)
    for ax,version,title in zip(axes,['previous_annual','revised_summer'],['Previous annual','Revised summer']):
        for i,metric in enumerate(metrics):
            v=z.query('version==@version and metric==@metric').iloc[0]
            ax.plot([v.pct_green,v.pct_target],[i,i],color='.6',lw=1)
            ax.scatter(v.pct_green,i,color='#0072B2',marker='o',label='Green30' if i==0 else None)
            ax.scatter(v.pct_target,i,color='#D55E00',marker='^',label='Target30' if i==0 else None)
        ax.set_title(title);ax.set_xlabel('High-vulnerability LSOAs with low benefit (%)');ax.set_xlim(0,70);ax.set_yticks(range(5),labels);ax.legend(frameon=False)
    axes[0].invert_yaxis()
    fig.suptitle('Low-benefit classification under Green30 and Target30  |  v1')
    fig.text(.5,.015,'Green30 tertiles fixed across scenarios within each version; annual and seasonal numerical thresholds differ.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'figure7_classification_comparison_v1.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
