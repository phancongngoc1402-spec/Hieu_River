#!/usr/bin/env python3
"""
Reproduce CR2/Satterthwaite and exact restricted wild-cluster-bootstrap-t
inference for the Hieu River repeated-monitoring rainfall models.

Input: Stats_Ready_robustness_input.csv in the same directory.
Dependencies: Python 3, numpy, scipy.
No random number generator is used: Rademacher sign patterns are enumerated exactly.
"""
from __future__ import annotations
import csv, itertools, math
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
INPUT = HERE / "Stats_Ready_robustness_input.csv"
OUTPUT = HERE / "robustness_results_reproduced.csv"

RESPONSES = {"TSS":"log10_TSS","Turbidity":"log10_Turbidity","Fe":"log10_Fe"}
WINDOWS = ["P1_mm","P3_mm","P7_mm","P14_mm","Rmax7_mm","P30_mm"]

def read_records(path):
    with open(path,newline="",encoding="utf-8") as f:
        recs=list(csv.DictReader(f))
    numeric=["round_order","rain_proxy_flag"]+WINDOWS+list(RESPONSES.values())
    for r in recs:
        for k in numeric:
            r[k]=float(r[k])
    return recs

def mat_power_psd(A,power,tol=1e-10):
    vals,vecs=np.linalg.eigh((A+A.T)/2)
    vmax=max(1.0,float(np.max(np.abs(vals))))
    keep=vals>tol*vmax
    vp=np.zeros_like(vals)
    vp[keep]=vals[keep]**power
    return (vecs*vp)@vecs.T

def ols_fit(y,X):
    M=np.linalg.pinv(X.T@X)
    b=M@X.T@y
    e=y-X@b
    return b,e,M

def cluster_cr1s(y,X,clusters):
    b,e,M=ols_fit(y,X)
    uniq=list(dict.fromkeys(clusters))
    meat=np.zeros((X.shape[1],X.shape[1]))
    for g in uniq:
        idx=np.array([c==g for c in clusters])
        s=X[idx,:].T@e[idx]
        meat += np.outer(s,s)
    G=len(uniq); N=len(y); K=np.linalg.matrix_rank(X)
    correction=(G/(G-1))*((N-1)/(N-K))
    V=correction*M@meat@M
    return b,e,V

def cr2_satt(y,X,clusters,coef_idx=1):
    b,e,M=ols_fit(y,X)
    unique=list(dict.fromkeys(clusters))
    # Square root of X'X inverse for Satterthwaite calculations.
    vals,vecs=np.linalg.eigh((M+M.T)/2)
    L=vecs@np.diag(np.sqrt(np.clip(vals,0,None)))@vecs.T
    scores=[]; G_list=[]; H_rows=[]
    for g in unique:
        idx=np.array([c==g for c in clusters])
        Xg=X[idx,:]; eg=e[idx]
        Hgg=Xg@M@Xg.T
        A=mat_power_psd(np.eye(Xg.shape[0])-Hgg,-0.5)
        score=Xg.T@(A@eg)
        scores.append(score)
        ME=M@(Xg.T@A)
        G_list.append(ME)
        Hmat=ME@Xg@L
        H_rows.append(Hmat[coef_idx,:])
    meat=sum(np.outer(s,s) for s in scores)
    V=M@meat@M
    se=math.sqrt(max(float(V[coef_idx,coef_idx]),0.0))

    # Bell-McCaffrey / Pustejovsky-Tipton Satterthwaite df.
    H_i=np.column_stack(H_rows)
    P=-(H_i.T@H_i)
    pdiag=np.array([np.sum(Gi[coef_idx,:]**2) for Gi in G_list])
    P=P+np.diag(pdiag)
    tr=float(np.trace(P)); ss=float(np.sum(P**2))
    df=(tr**2/ss) if ss>0 else float("nan")
    t=float(b[coef_idx]/se)
    p=float(2*stats.t.sf(abs(t),df))
    crit=float(stats.t.ppf(0.975,df))
    ci=(float(b[coef_idx]-crit*se),float(b[coef_idx]+crit*se))
    return {"beta":float(b[coef_idx]),"se":se,"df":df,"t":t,"p":p,"ci":ci}

def wild_cluster_bootstrap_exact(y,X,clusters,coef_idx=1):
    clusters=np.asarray(clusters)
    uniq=np.array(list(dict.fromkeys(clusters.tolist())))
    G=len(uniq)
    b,e,V=cluster_cr1s(y,X,clusters.tolist())
    t_obs=float(b[coef_idx]/math.sqrt(V[coef_idx,coef_idx]))

    # Restricted model imposes H0: beta_x = 0.
    Xr=np.delete(X,coef_idx,axis=1)
    br,er,_=ols_fit(y,Xr)
    fitted0=Xr@br
    gmap={g:i for i,g in enumerate(uniq)}
    gi=np.array([gmap[g] for g in clusters])
    exceed=0; B=0
    for signs in itertools.product([-1.0,1.0],repeat=G):
        s=np.asarray(signs)
        ystar=fitted0+er*s[gi]
        bb,_,Vb=cluster_cr1s(ystar,X,clusters.tolist())
        se=math.sqrt(max(float(Vb[coef_idx,coef_idx]),0.0))
        if se==0: continue
        tstar=float(bb[coef_idx]/se)
        B+=1
        if abs(tstar)>=abs(t_obs)-1e-12:
            exceed+=1
    return {"t_obs":t_obs,"p":exceed/B,"B":B}

def make_X(recs,xvar="P3_mm",campaign=False):
    stations=sorted({r["station"] for r in recs})
    rounds=sorted({int(r["round_order"]) for r in recs})
    X=[]
    for r in recs:
        row=[1.0,float(r[xvar])/10.0]
        row += [1.0 if r["station"]==s else 0.0 for s in stations[1:]]
        if campaign:
            row += [1.0 if int(r["round_order"])==rd else 0.0 for rd in rounds[1:]]
        X.append(row)
    return np.asarray(X,float)

def fit_spec(recs,xvar="P3_mm",campaign=False):
    X=make_X(recs,xvar,campaign)
    cl=[r["station"] for r in recs]
    out=[]
    for response,ycol in RESPONSES.items():
        y=np.asarray([float(r[ycol]) for r in recs])
        cr=cr2_satt(y,X,cl,1)
        wc=wild_cluster_bootstrap_exact(y,X,cl,1)
        out.append({
            "response":response,"n":len(recs),"clusters":len(set(cl)),
            "xvar":xvar,"campaign":int(campaign),
            "effect_pct":(10**cr["beta"]-1)*100,
            "ci_lo_pct":(10**cr["ci"][0]-1)*100,
            "ci_hi_pct":(10**cr["ci"][1]-1)*100,
            "cr2_df":cr["df"],"cr2_p":cr["p"],
            "wcb_p":wc["p"],"wcb_patterns":wc["B"]
        })
    return out

def main():
    recs=read_records(INPUT)
    all_rows=[]
    def add(section,vals,omitted_station=""):
        for v in vals:
            row={"section":section,"omitted_station":omitted_station}
            row.update(v); all_rows.append(row)

    add("primary_station_FE",fit_spec(recs))
    add("campaign_adjusted",fit_spec(recs,campaign=True))

    no_proxy=[r for r in recs if int(r["rain_proxy_flag"])==0]
    add("proxy_excluded_station_FE",fit_spec(no_proxy))
    add("proxy_excluded_campaign_adjusted",fit_spec(no_proxy,campaign=True))

    for st in sorted({r["station"] for r in recs}):
        sub=[r for r in recs if r["station"]!=st]
        add("leave_one_station_out_station_FE",fit_spec(sub),st)
        add("leave_one_station_out_campaign_adjusted",fit_spec(sub,campaign=True),st)

    for window in WINDOWS:
        add("alternative_rainfall_windows_station_FE",fit_spec(recs,xvar=window))

    fields=["section","omitted_station","response","n","clusters","xvar","campaign",
            "effect_pct","ci_lo_pct","ci_hi_pct","cr2_df","cr2_p","wcb_p","wcb_patterns"]
    with open(OUTPUT,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {OUTPUT}")

if __name__=="__main__":
    main()
