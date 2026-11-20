"""Evaluation shared by every model. Inputs: y, p of shape (N, A) with scores in {-2..2}.

T1  polar-aspect detection   : binary (score != 0) per slot
T2  polarity-intensity       : on gold-polar slots, classes {-2,-1,+1,+2}; predicting 0 counts as an error
T3  end-to-end 5-level (MAIN): per-aspect macro-F1 over the 5 levels present in gold, averaged over aspects
    + joint (aspect, polarity) tuple F1 (VLSP-2018 style), QWK, accuracy, exact match
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

from .common import ASPECTS, SCORE_NAMES, SCORES

POLAR = [-2, -1, 1, 2]
PRIMARY = 't3_macro_f1'


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def evaluate(y, p, per_aspect=True):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    N, A = y.shape
    r = {'n_reviews': N}
    # ---------------- T1
    t, q = y != 0, p != 0
    pr, rc, f, _ = precision_recall_fscore_support(t.ravel(), q.ravel(), average='binary', zero_division=0)
    r.update(t1_p=pr, t1_r=rc, t1_f1_micro=f)
    t1a = [f1_score(t[:, a], q[:, a], zero_division=0) for a in range(A)]
    r['t1_f1_macro'] = float(np.mean(t1a))
    # ---------------- T2 (gold-polar slots)
    m = y != 0
    t2a = []
    for a in range(A):
        g, h = y[m[:, a], a], p[m[:, a], a]
        labs = [c for c in POLAR if (g == c).any()]
        t2a.append(f1_score(g, h, labels=labs, average='macro', zero_division=0) if len(g) else np.nan)
    r['t2_macro_f1'] = float(np.nanmean(t2a))
    gy, gp = y[m], p[m]
    r['t2_acc'] = float((gy == gp).mean())
    r['t2_sign_acc'] = float((np.sign(gy) == np.sign(gp)).mean())
    r['t2_mae'] = float(np.abs(gy - gp).mean())
    # ---------------- T3
    t3a, f1mat = [], np.full((A, 5), np.nan)
    for a in range(A):
        labs = [c for c in SCORES if (y[:, a] == c).any()]
        t3a.append(f1_score(y[:, a], p[:, a], labels=labs, average='macro', zero_division=0))
        pc = f1_score(y[:, a], p[:, a], labels=SCORES, average=None, zero_division=0)
        for k, c in enumerate(SCORES):
            if (y[:, a] == c).any():
                f1mat[a, k] = pc[k]
    r['t3_macro_f1'] = float(np.mean(t3a))
    r['t3_macro_f1_pooled'] = float(f1_score(y.ravel(), p.ravel(), labels=SCORES, average='macro', zero_division=0))
    r['t3_acc'] = float((y == p).mean())
    r['t3_qwk'] = float(cohen_kappa_score(y.ravel(), p.ravel(), weights='quadratic', labels=SCORES))
    r['t3_mae'] = float(np.abs(y - p).mean())
    r['exact_match'] = float((y == p).all(1).mean())
    tp = ((p == y) & m).sum(0)
    fp = ((p != 0) & (p != y)).sum(0)
    fn = (m & (p != y)).sum(0)
    r['joint_f1_micro'] = float(_f1(tp.sum(), fp.sum(), fn.sum()))
    r['joint_f1_macro'] = float(_f1(tp, fp, fn).mean())
    for k, c in enumerate(SCORES):
        sel = y.ravel() == c
        r[f'f1_class_{SCORE_NAMES[k]}'] = float(f1_score(y.ravel(), p.ravel(), labels=[c], average='macro',
                                                       zero_division=0)) if sel.any() else np.nan
    # ---------------- error types (share of all slots)
    err = y != p
    r['err_intensity'] = float((err & (y != 0) & (p != 0) & (np.sign(y) == np.sign(p))).mean())
    r['err_polarity_flip'] = float((err & (np.sign(y) * np.sign(p) == -1)).mean())
    r['err_missed'] = float((err & (y != 0) & (p == 0)).mean())
    r['err_spurious'] = float((err & (y == 0) & (p != 0)).mean())
    if per_aspect:
        r['per_aspect'] = {a: {'t1_f1': float(t1a[i]), 't2_macro_f1': float(t2a[i]), 't3_macro_f1': float(t3a[i]),
                               'joint_f1': float(_f1(tp[i], fp[i], fn[i])), 'n_polar': int(m[:, i].sum())}
                           for i, a in enumerate(ASPECTS)}
        r['f1_matrix'] = f1mat.tolist()
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}


# =========================================================================== significance
def _features(y, p, metric):
    """Per-review count features (N,F) and fn(summed counts)->metric, for a fast vectorised bootstrap."""
    N, A = y.shape
    if metric == 't1_f1_micro':
        t, q = y != 0, p != 0
        F = np.stack([(t & q).sum(1), (~t & q).sum(1), (t & ~q).sum(1)], 1)
        return F, lambda c: _f1(c[..., 0], c[..., 1], c[..., 2])
    if metric == 'joint_f1_macro':
        m = y != 0
        F = np.concatenate([(p == y) & m, (p != 0) & (p != y), m & (p != y)], 1).astype(np.int32)
        return F, lambda c: _f1(c[..., :A], c[..., A:2 * A], c[..., 2 * A:]).mean(-1)
    if metric in ('t3_macro_f1', 't2_macro_f1'):
        feats, groups = [], []
        for a in range(A):
            if metric == 't3_macro_f1':
                scope, labs = np.ones(N, bool), [c for c in SCORES if (y[:, a] == c).any()]
            else:
                scope, labs = y[:, a] != 0, [c for c in POLAR if (y[:, a] == c).any()]
            for c in labs:
                g, h = y[:, a], p[:, a]
                feats += [scope & (g == c) & (h == c), scope & (g != c) & (h == c), scope & (g == c) & (h != c)]
                groups.append(a)
        F = np.stack(feats, 1).astype(np.int32)
        groups = np.array(groups)
        ua = np.unique(groups)

        def fn(c):
            f1 = _f1(c[..., 0::3], c[..., 1::3], c[..., 2::3])
            return np.stack([f1[..., groups == a].mean(-1) for a in ua], -1).mean(-1)
        return F, fn
    raise ValueError(metric)


def paired_bootstrap(y, pa, pb, metric=PRIMARY, n_boot=1000, seed=0):
    """Same resamples for A and B. Returns delta (A-B), 95% CI and one-sided p = P(delta* <= 0)."""
    y, pa, pb = (np.asarray(x, dtype=int) for x in (y, pa, pb))
    Fa, fa = _features(y, pa, metric)
    Fb, fb = _features(y, pb, metric)
    N = len(y)
    delta = float(fa(Fa.sum(0)) - fb(Fb.sum(0)))
    rng = np.random.RandomState(seed)
    ds = []
    for s in range(0, n_boot, 100):
        k = min(100, n_boot - s)
        W = np.stack([np.bincount(rng.randint(0, N, N), minlength=N) for _ in range(k)]).astype(np.float64)
        ds.append(fa(W @ Fa) - fb(W @ Fb))
    ds = np.concatenate(ds)
    return {'delta': delta, 'ci_low': float(np.percentile(ds, 2.5)), 'ci_high': float(np.percentile(ds, 97.5)),
            'p_value': float((ds <= 0).mean())}


def bootstrap_ci(y, p, metric=PRIMARY, n_boot=1000, seed=0):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    F, f = _features(y, p, metric)
    N = len(y)
    rng = np.random.RandomState(seed)
    vals = []
    for s in range(0, n_boot, 100):
        k = min(100, n_boot - s)
        W = np.stack([np.bincount(rng.randint(0, N, N), minlength=N) for _ in range(k)]).astype(np.float64)
        vals.append(f(W @ F))
    v = np.concatenate(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, run, m = {}, 0.0, len(items)
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out