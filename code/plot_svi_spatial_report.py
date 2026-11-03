"""Rebuild the two aggregate illustrations in docs/manuscript (no model refits)."""
from pathlib import Path
from io import StringIO
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / "docs/manuscript"
# Reuse the exact R ranking and tie handling of the scoring audit.
cmd = 'invisible(capture.output(source("code/compare_figure2_svi.R"))); write.table(unclass(table(ntile(old,5),ntile(canonical,5))),row.names=FALSE,col.names=FALSE)'
matrix = np.loadtxt(StringIO(subprocess.check_output(["Rscript", "-e", cmd], cwd=root, text=True)))
assert matrix.shape == (5, 5) and matrix.sum() == 4835
fig, ax = plt.subplots(figsize=(5.7, 4.5), layout="constrained")
ax.imshow(matrix, cmap="Blues", vmin=0)
for (i, j), n in np.ndenumerate(matrix):
    ax.text(j, i, str(int(n)) if n else "–", ha="center", va="center",
            color="white" if n > matrix.max()/2 else "#263b48")
labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
ax.set(xticks=range(5), xticklabels=labels, yticks=range(5), yticklabels=labels,
       xlabel="Author’s four-theme SVI", ylabel="Former Figure 2 reconstruction",
       title="Most neighbourhoods stayed in the same quintile")
fig.savefig(out / "svi_quintile_transitions.png", dpi=170)
plt.close(fig)

stats = pd.read_csv(root / "figures/figure6_statistical_reporting/figure6_spatial_stats.csv")
metrics = ["pIncomeDeprivation", "p5under", "p75over", "pBAME", "overall_score"]
assert (stats["rate_denominator"] == 1_000_000).all()
contrasts = stats.pivot(index="metric", columns="k", values="contrast").loc[metrics,[0,100]]
assert not contrasts.isna().any().any()
fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
ys = np.arange(len(metrics))
ax.axvline(0, color="#63717a", linewidth=1, linestyle="--")
ax.hlines(ys, contrasts.min(axis=1), contrasts.max(axis=1), color="#abbac1", linewidth=2)
for k, marker, colour, label in [(0,"o","#3e5263","Citywide"),(100,"D","#b54b16","Spatially adjusted (k = 100)")]:
    ax.scatter(contrasts[k], ys, marker=marker, color=colour, label=label, s=56, zorder=3)
ax.set(yticks=ys, yticklabels=["Income deprivation","Age <5 share","Age ≥75 share",
                               "Ethnic-minority share","Four-theme SVI"],
       xlabel="P90 minus P10: modelled deaths averted per million residents",
       title="Accounting for location brings all five contrasts closer to zero")
ax.invert_yaxis()
ax.spines[["top","right","left"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(.5,-.18), ncol=2, frameon=False)
fig.savefig(out / "figure6_spatial_contrasts.png", dpi=170)
