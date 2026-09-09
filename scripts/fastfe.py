# -*- coding: utf-8 -*-
"""Fixed-effects estimation by absorption, with cluster-robust inference.

Estimating supplier effects with ~11,000 dummy variables is infeasible at
this sample size, so the supplier effect is absorbed by within-transforming
every variable around its supplier mean.  Standard errors are clustered on
the supplier, with the finite-sample correction that accounts for the
absorbed parameters.
"""
import numpy as np
import pandas as pd


def demean(df, cols, group):
    g = df[group].values
    _, gidx = np.unique(g, return_inverse=True)
    cnt = np.bincount(gidx)
    out = {}
    for c in cols:
        v = df[c].astype(float).values
        m = np.bincount(gidx, weights=v) / cnt
        out[c] = v - m[gidx]
    return pd.DataFrame(out, index=df.index), gidx, len(cnt)


def fe_ols(df, y, X, group):
    """Within estimator with cluster-robust (CR1) standard errors."""
    W, gidx, G = demean(df, [y] + list(X), group)
    yv = W[y].values
    Xv = W[list(X)].values
    XtX = Xv.T @ Xv
    XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ (Xv.T @ yv)
    resid = yv - Xv @ beta

    N, K = Xv.shape
    meat = np.zeros((K, K))
    order = np.argsort(gidx)
    gs = gidx[order]
    Xs, rs = Xv[order], resid[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    for sl in np.split(np.arange(N), bounds):
        u = Xs[sl].T @ rs[sl]
        meat += np.outer(u, u)
    dof = (G / (G - 1)) * ((N - 1) / (N - K - G))
    V = XtXi @ meat @ XtXi * dof
    se = np.sqrt(np.diag(V))
    from scipy import stats as st
    t = beta / se
    p = 2 * st.t.sf(np.abs(t), G - 1)
    ss_res = (resid ** 2).sum()
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    return pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p},
                        index=list(X)), {
        "N": N, "groups": G, "within_r2": 1 - ss_res / ss_tot}
