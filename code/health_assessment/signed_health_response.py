"""Exact signed sum of D*(1-RR_scenario/RR_baseline), with thresholded exposure.

Offsets include minus the threshold. Positive cooling means baseline-scenario.
Negative results are warming-related harm and MUST NOT be clipped to zero.
Baseline and scenario must have identical exposure offsets and matched pixels.
"""
import numpy as np


def signed_sum(baseline, cooling, deaths, rr, offsets):
    b=np.asarray(baseline,dtype=float);c=np.asarray(cooling,dtype=float)
    d=np.asarray(deaths,dtype=float);o=np.asarray(offsets,dtype=float)
    if b.shape!=c.shape or b.shape!=d.shape or b.ndim!=1:
        raise ValueError('Expected matched one-dimensional pixel arrays')
    if not all(np.isfinite(v).all() for v in [b,c,d,o]) or (d<0).any() or not np.isfinite(rr) or rr<=0:
        raise ValueError('Invalid exposure, mortality or risk input')
    beta=np.log(rr);answer=np.zeros_like(o)
    for warming in [False,True]:
        mask=c<0 if warming else c>0
        if not mask.any():continue
        lo=np.minimum(b[mask],b[mask]-c[mask]);hi=np.maximum(b[mask],b[mask]-c[mask]);w=d[mask]
        il=np.argsort(lo);ih=np.argsort(hi)
        kl=np.searchsorted(lo[il],-o,side='right');kh=np.searchsorted(hi[ih],-o,side='right')
        def tail(v,order,k):
            acc=np.r_[0.,np.cumsum(v[order])];return acc[-1]-acc[k]
        if warming:
            # Baseline-cold/scenario-hot crossings and both-hot cells.
            cross=w*np.exp(beta*hi)
            answer+=tail(w,ih,kh)-tail(w*np.exp(beta*(hi-lo)),il,kl)-np.exp(beta*o)*(tail(cross,ih,kh)-tail(cross,il,kl))
        else:
            cross=w*np.exp(-beta*hi)
            answer+=tail(w,ih,kh)-tail(w*np.exp(-beta*(hi-lo)),il,kl)-np.exp(-beta*o)*(tail(cross,ih,kh)-tail(cross,il,kl))
    return answer
