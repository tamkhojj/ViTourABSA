"""Aggregation over seeds, significance, result tables and analysis figures."""
import glob
import os
import re

import numpy as np
import pandas as pd

from ..common import ASPECTS, SCORE_NAMES, labels_of, load_json, read_jsonl
from .eda import _close_if_script, _plt, to_latex
from ..metrics import PRIMARY, evaluate, holm, paired_bootstrap

OUT = 'reports/results'
MAIN_COLS = [('t1_f1_micro', 'T1 F1', 100), ('t2_macro_f1', 'T2 Macro-F1', 100), ('t3_macro_f1', 'T3 Macro-F1', 100),
             ('joint_f1_macro', 'Joint F1', 100), ('t3_qwk', 'QWK', 1), ('exact_match', 'Exact', 100)]


def _save(df, name, caption='', fmt='{:.2f}'):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name + '.csv'), encoding='utf-8-sig')
    to_latex(df, os.path.join(OUT, name + '.tex'), caption, 'tab:' + name, fmt)
    return df


def collect(root='results'):
    recs = []
    for f in glob.glob(os.path.join(root, '*', '*', 'seed*', 'metrics.json')):
        parts = f.split(os.sep)
        split, run, seed = parts[-4], parts[-3], int(re.search(r'seed(\d+)', parts[-2]).group(1))
        if '__lr' in run:
            continue
        m = load_json(f)
        r = {'split': split, 'run': run, 'seed': seed, 'dir': os.path.dirname(f),
             'params': m.get('params'), 'sec_per_epoch': m.get('sec_per_epoch'), 'peak_mem_gb': m.get('peak_mem_gb'),
             'reviews_per_sec': m.get('test_reviews_per_sec'), 'invalid_rate': m.get('invalid_rate',
                                                                                   m.get('test', {}).get('invalid_rate'))}
        for k, v in m.get('test', {}).items():
            if not isinstance(v, (dict, list)):
                r[k] = v
        r['val_primary'] = m.get('val', {}).get(PRIMARY)
        recs.append(r)
    return pd.DataFrame(recs)


def meta_of(run, meta):
    if run in meta:
        return meta[run]
    return ('Other', run)


def summary(df, split, meta, cols=MAIN_COLS):
    d = df[df.split == split]
    rows = []
    for run, g in d.groupby('run'):
        grp, disp = meta_of(run, meta)
        r = {'group': grp, 'model': disp, 'run': run, 'n_seeds': len(g), 'params': g.params.iloc[0]}
        for k, _, s in cols:
            if k in g:
                r[k + '_mean'] = g[k].mean() * s
                r[k + '_std'] = g[k].std() * s if len(g) > 1 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {g: i for i, g in enumerate(dict.fromkeys(v[0] for v in meta.values()))}
    out['_o'] = out.group.map(lambda g: order.get(g, 99))
    out['_r'] = out.run.map(lambda r: list(meta).index(r) if r in meta else 999)
    return out.sort_values(['_o', '_r']).drop(columns=['_o', '_r']).reset_index(drop=True)


def main_table(df, meta, split='S1_main', human=None, name='R01_main_results', caption=None):
    s = summary(df, split, meta)
    if s.empty:
        return s
    tab = pd.DataFrame(index=pd.MultiIndex.from_arrays([s.group, s.model], names=['Group', 'Model']))
    tab['#Params'] = [f'{p / 1e6:.0f}M' if isinstance(p, (int, float)) and not np.isnan(p) else '--' for p in s.params]
    tab['Seeds'] = s.n_seeds.values
    for k, lab, _ in MAIN_COLS:
        if k + '_mean' not in s:
            continue
        m, sd = s[k + '_mean'].values, s[k + '_std'].values
        best = np.nanmax(m)
        f = '{:.3f}' if k == 't3_qwk' else '{:.2f}'
        tab[lab] = [(f.format(a) + ('' if np.isnan(b) else ' ± ' + f.format(b)) + (' *' if np.isclose(a, best) else ''))
                    for a, b in zip(m, sd)]
    if human is not None:
        tab.loc[('Human', 'Second annotator (subset)'), :] = ['--', '--'] + [
            ('{:.3f}' if k == 't3_qwk' else '{:.2f}').format(human[k] * s_) for k, _, s_ in MAIN_COLS]
    _save(tab, name, caption or f'Results on {split} (mean ± std over seeds; * best).')
    return tab


def best_run(df, split='S1_main', exclude=('classic_majority', 'classic_prior')):
    d = df[(df.split == split) & (~df.run.isin(exclude))]
    return d.groupby('run')[PRIMARY].mean().idxmax()


def significance(df, split='S1_main', ref=None, metrics=(PRIMARY, 'joint_f1_macro'), n_boot=1000):
    d = df[df.split == split]
    ref = ref or best_run(df, split)
    R = d[d.run == ref].set_index('seed')['dir'].to_dict()
    rows = []
    for metric in metrics:
        pv = {}
        for run, g in d[d.run != ref].groupby('run'):
            O = g.set_index('seed')['dir'].to_dict()
            pairs = [(s, s) for s in R if s in O] or [(s, next(iter(O))) for s in R]
            deltas, ps = [], []
            for a, b in pairs:
                A_, B_ = np.load(os.path.join(R[a], 'preds_test.npz')), np.load(os.path.join(O[b], 'preds_test.npz'))
                if A_['y'].shape != B_['y'].shape or not (A_['y'] == B_['y']).all():
                    continue
                r = paired_bootstrap(A_['y'], A_['pred'], B_['pred'], metric, n_boot)
                deltas.append(r['delta']); ps.append(r['p_value'])
            if deltas:
                pv[run] = max(ps)
                rows.append({'metric': metric, 'reference': ref, 'run': run, 'delta (ref − run) ×100': 100 * np.mean(deltas),
                             'p (max over seeds)': max(ps), 'pairs': len(deltas)})
        adj = holm(pv)
        for r in rows:
            if r['metric'] == metric and r['run'] in adj:
                r['p Holm'] = adj[r['run']]
    t = pd.DataFrame(rows)
    if len(t):
        t['significant (p<0.05)'] = t['p Holm'] < 0.05
        _save(t.set_index(['metric', 'run']), 'R02_significance', 'Paired bootstrap vs. the best model.', '{:.4f}')
    return t


def _runs_dirs(df, split, run):
    return df[(df.split == split) & (df.run == run)].sort_values('seed')['dir'].tolist()


def per_aspect_table(df, runs, meta, split='S1_main'):
    rows = {}
    for run in runs:
        ms = [load_json(os.path.join(d, 'metrics.json'))['test']['per_aspect'] for d in _runs_dirs(df, split, run)]
        if not ms:
            continue
        rows[meta_of(run, meta)[1]] = {a: 100 * np.mean([m[a]['t3_macro_f1'] for m in ms]) for a in ASPECTS}
    t = pd.DataFrame(rows).T
    t['Mean'] = t.mean(1)
    t.index.name = 'Model'
    return _save(t, 'R03_per_aspect', 'T3 Macro-F1 per aspect (mean over seeds).')


def f1_heatmap(df, run, meta, split='S1_main'):
    plt = _plt()
    mats = [np.array(load_json(os.path.join(d, 'metrics.json'))['test']['f1_matrix'], dtype=float)
            for d in _runs_dirs(df, split, run)]
    M = np.nanmean(mats, 0) * 100
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M, cmap='viridis', vmin=0, vmax=100)
    ax.set_xticks(range(5)); ax.set_xticklabels(SCORE_NAMES); ax.set_yticks(range(len(ASPECTS))); ax.set_yticklabels(ASPECTS)
    for i in range(M.shape[0]):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', color='w' if M[i, j] < 60 else 'k', fontsize=7)
    ax.set_title(f'F1 per aspect × level: {meta_of(run, meta)[1]}'); fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); save_fig_r(fig, 'R_F01_f1_heatmap')
    return fig


def save_fig_r(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', dpi=200)
    _close_if_script(fig)


def confusion(df, run, meta, split='S1_main'):
    plt = _plt()
    C = np.zeros((5, 5))
    for d in _runs_dirs(df, split, run):
        z = np.load(os.path.join(d, 'preds_test.npz'))
        for i, g in enumerate(range(-2, 3)):
            for j, p in enumerate(range(-2, 3)):
                C[i, j] += ((z['y'] == g) & (z['pred'] == p)).sum()
    Cn = C / C.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(Cn, cmap='Blues', vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{Cn[i, j]:.1f}', ha='center', va='center', fontsize=7, color='w' if Cn[i, j] > 60 else 'k')
    ax.set_xticks(range(5)); ax.set_xticklabels(SCORE_NAMES); ax.set_yticks(range(5)); ax.set_yticklabels(SCORE_NAMES)
    ax.set_xlabel('predicted'); ax.set_ylabel('gold'); ax.set_title(f'Row-normalised confusion (%)\n{meta_of(run, meta)[1]}')
    plt.tight_layout(); save_fig_r(fig, 'R_F02_confusion')
    return fig


def error_types(df, runs, meta, split='S1_main'):
    keys = ['err_intensity', 'err_polarity_flip', 'err_missed', 'err_spurious']
    d = df[(df.split == split) & df.run.isin(runs)]
    t = d.groupby('run')[keys].mean() * 100
    t['total error'] = t.sum(1)
    t.index = [meta_of(r, meta)[1] for r in t.index]
    t.index.name = 'Model'
    t.columns = ['Intensity (±1↔±2)', 'Polarity flip', 'Missed (≠0→0)', 'Spurious (0→≠0)', 'Total']
    return _save(t, 'R04_error_types', 'Error types (% of all aspect slots).', '{:.2f}')


def breakdown(df, runs, meta, split_dir, split='S1_main'):
    plt = _plt()
    te = read_jsonl(os.path.join(split_dir, 'test.jsonl'))
    Y = labels_of(te)
    nz = (Y != 0).sum(1)
    L = np.array([len(r['text'].split()) for r in te])
    mixed = (Y > 0).any(1) & (Y < 0).any(1)
    groups = {
        '#polar aspects': {'≤1': nz <= 1, '2': nz == 2, '3': nz == 3, '4': nz == 4, '≥5': nz >= 5},
        'length': {'≤10': L <= 10, '11–25': (L > 10) & (L <= 25), '26–50': (L > 25) & (L <= 50),
                   '51–100': (L > 50) & (L <= 100), '>100': L > 100},
        'polarity': {'single-polarity': ~mixed, 'mixed': mixed},
        'province': {p: np.array([r['province'] == p for r in te]) for p in sorted({r['province'] for r in te})},
        'facility': {f: np.array([r['facility'] == f for r in te]) for f in sorted({r['facility'] for r in te})},
    }
    rows = []
    for run in runs:
        preds = [np.load(os.path.join(d, 'preds_test.npz'))['pred'] for d in _runs_dirs(df, split, run)]
        preds = [P for P in preds if len(P) == len(Y)]          # skip runs evaluated on a different test set
        if not preds:
            continue
        for gname, bins in groups.items():
            for b, m in bins.items():
                if m.sum() < 20:
                    continue
                v = [evaluate(Y[m], P[m], per_aspect=False) for P in preds]
                rows.append({'model': meta_of(run, meta)[1], 'by': gname, 'bucket': b, 'n': int(m.sum()),
                             'T3 Macro-F1': 100 * np.mean([x[PRIMARY] for x in v]),
                             'Joint F1': 100 * np.mean([x['joint_f1_macro'] for x in v])})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, None
    _save(t.set_index(['by', 'bucket', 'model']), 'R05_breakdown', 'Performance by review characteristics.')
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, by in zip(axes, ['#polar aspects', 'length', 'province']):
        sub = t[t.by == by]
        for mdl, g in sub.groupby('model', sort=False):
            ax.plot(g.bucket, g['Joint F1'], marker='o', label=mdl)
        ax.set_title(f'Joint F1 by {by}'); ax.tick_params(axis='x', rotation=45)
    axes[0].legend(fontsize=6)
    plt.tight_layout(); save_fig_r(fig, 'R_F03_breakdown')
    return t, fig


def generalization(df, runs, meta):
    rows = []
    for run in runs:
        r = {'Model': meta_of(run, meta)[1]}
        for split, lab in [('S1_main', 'In-domain (S1)'), ('S2_out_of_region', 'Out-of-region (S2)'),
                           ('S4_hotel_to_resort', 'Hotel→Resort'), ('S4_resort_to_hotel', 'Resort→Hotel')]:
            v = df[(df.split == split) & (df.run == run)][PRIMARY]
            r[lab] = 100 * v.mean() if len(v) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).set_index('Model')
    return _save(t, 'R06_generalization', 'T3 Macro-F1 under distribution shift.')


def lopo(df, runs, meta, s1_dir):
    te = read_jsonl(os.path.join(s1_dir, 'test.jsonl'))
    prov = np.array([r['province'] for r in te])
    rows = []
    for run in runs:
        for split in sorted(df.split.unique()):
            if not split.startswith('S3_lopo_'):
                continue
            p = split.replace('S3_lopo_', '')
            v = df[(df.split == split) & (df.run == run)][PRIMARY]
            ins = []
            for d in _runs_dirs(df, 'S1_main', run):
                z = np.load(os.path.join(d, 'preds_test.npz'))
                if len(z['y']) != len(prov):
                    continue
                m = prov == p
                ins.append(evaluate(z['y'][m], z['pred'][m], per_aspect=False)[PRIMARY])
            rows.append({'Model': meta_of(run, meta)[1], 'Held-out province': p,
                         'In-distribution (S1 test, same province)': 100 * np.mean(ins) if ins else np.nan,
                         'Leave-province-out': 100 * v.mean() if len(v) else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t['Drop'] = t.iloc[:, 2] - t.iloc[:, 3]
    return _save(t.set_index(['Model', 'Held-out province']), 'R07_lopo', 'Leave-one-province-out (T3 Macro-F1).')


def learning_curve(df, runs, meta):
    plt = _plt()
    fr = {f'S5_frac{f:g}': f for f in [0.01, 0.05, 0.1, 0.25, 0.5]}
    fr['S1_main'] = 1.0
    rows = []
    for run in runs:
        for split, f in fr.items():
            v = df[(df.split == split) & (df.run == run)][PRIMARY]
            if len(v):
                rows.append({'model': meta_of(run, meta)[1], 'fraction': f, 'mean': 100 * v.mean(),
                             'std': 100 * v.std() if len(v) > 1 else 0})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, None
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m, g in t.groupby('model', sort=False):
        g = g.sort_values('fraction')
        ax.errorbar(g.fraction, g['mean'], yerr=g['std'], marker='o', capsize=3, label=m)
    ax.set_xscale('log'); ax.set_xlabel('fraction of training data'); ax.set_ylabel('T3 Macro-F1'); ax.legend(fontsize=7)
    plt.tight_layout(); save_fig_r(fig, 'R_F04_learning_curve')
    _save(t.set_index(['model', 'fraction']), 'R08_learning_curve', 'Learning curve.')
    return t, fig


def leakage(df, runs, meta, s6_dir):
    te = read_jsonl(os.path.join(s6_dir, 'test.jsonl'))
    seen = np.array([bool(r.get('seen_in_train')) for r in te])
    rows = []
    for run in runs:
        r = {'Model': meta_of(run, meta)[1]}
        v1 = df[(df.split == 'S1_main') & (df.run == run)][PRIMARY]
        r['S1 deduplicated'] = 100 * v1.mean() if len(v1) else np.nan
        a, s, u = [], [], []
        for d in _runs_dirs(df, 'S6_with_duplicates', run):
            z = np.load(os.path.join(d, 'preds_test.npz'))
            if len(z['y']) != len(seen):
                continue
            a.append(evaluate(z['y'], z['pred'], per_aspect=False)[PRIMARY])
            s.append(evaluate(z['y'][seen], z['pred'][seen], per_aspect=False)[PRIMARY])
            u.append(evaluate(z['y'][~seen], z['pred'][~seen], per_aspect=False)[PRIMARY])
        r['S6 all test'] = 100 * np.mean(a) if a else np.nan
        r[f'S6 test seen in train ({100 * seen.mean():.1f}%)'] = 100 * np.mean(s) if s else np.nan
        r['S6 test unseen'] = 100 * np.mean(u) if u else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).set_index('Model')
    return _save(t, 'R09_duplicate_leakage', 'Effect of duplicate leakage (T3 Macro-F1).')


def llm_table(df, llm_models, split='S1_main'):
    rows = []
    for short, disp in llm_models.items():
        for mode in ['zero', 'few', 'qlora']:
            g = df[(df.split == split) & (df.run == f'llm_{short}_{mode}')]
            if g.empty:
                continue
            rows.append({'Model': disp, 'Setting': {'zero': 'Zero-shot', 'few': '10-shot', 'qlora': 'QLoRA'}[mode],
                         'Invalid %': 100 * g.invalid_rate.mean(), 'T1 F1': 100 * g.t1_f1_micro.mean(),
                         'T2 Macro-F1': 100 * g.t2_macro_f1.mean(), 'T3 Macro-F1': 100 * g[PRIMARY].mean(),
                         'Joint F1': 100 * g.joint_f1_macro.mean(), 'Reviews/s': g.reviews_per_sec.mean()})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return _save(t.set_index(['Model', 'Setting']), 'R10_llm', 'LLM results by setting.')


def efficiency(df, meta, split='S1_main'):
    d = df[df.split == split]
    t = d.groupby('run').agg(params=('params', 'first'), sec_per_epoch=('sec_per_epoch', 'mean'),
                             peak_mem_gb=('peak_mem_gb', 'mean'), reviews_per_sec=('reviews_per_sec', 'mean'),
                             t3=(PRIMARY, 'mean'))
    t = t[t.reviews_per_sec.notna()]
    t.index = [meta_of(r, meta)[1] for r in t.index]
    t.index.name = 'Model'
    t['params'] = t['params'] / 1e6
    t['t3'] *= 100
    t.columns = ['Params (M)', 'Train s/epoch', 'Peak GPU mem (GB)', 'Inference reviews/s', 'T3 Macro-F1']
    return _save(t.sort_values('T3 Macro-F1', ascending=False), 'R11_efficiency', 'Computational cost (RTX 4090).')


def subset_comparison(df, runs, meta, ids, human_metrics, split_dir, split='S1_main'):
    """Models vs. the human annotator on exactly the human-annotated subset."""
    te = read_jsonl(os.path.join(split_dir, 'test.jsonl'))
    pos = {r['id']: i for i, r in enumerate(te)}
    idx = np.array([pos[i] for i in ids if i in pos])
    rows = [{'Model': 'Human (second annotator)', **{lab: human_metrics[k] * s for k, lab, s in MAIN_COLS}}]
    for run in runs:
        vals = []
        for d in _runs_dirs(df, split, run):
            z = np.load(os.path.join(d, 'preds_test.npz'))
            if len(z['y']) == len(te):
                vals.append(evaluate(z['y'][idx], z['pred'][idx], per_aspect=False))
        if vals:
            rows.append({'Model': meta_of(run, meta)[1], **{lab: np.mean([v[k] for v in vals]) * s for k, lab, s in MAIN_COLS}})
    t = pd.DataFrame(rows).set_index('Model')
    return _save(t, 'R12_human_vs_models', 'Human upper bound vs. models on the re-annotated subset.')


def error_sample(df, run, split_dir, split='S1_main', n=200, seed=0):
    d = _runs_dirs(df, split, run)[0]
    z = np.load(os.path.join(d, 'preds_test.npz'))
    te = read_jsonl(os.path.join(split_dir, 'test.jsonl'))
    err = np.argwhere(z['y'] != z['pred'])
    pick = err[np.random.RandomState(seed).choice(len(err), min(n, len(err)), replace=False)]
    rows = []
    for i, a in pick:
        g, p = int(z['y'][i, a]), int(z['pred'][i, a])
        kind = 'missed' if p == 0 else 'spurious' if g == 0 else 'polarity flip' if g * p < 0 else 'intensity'
        rows.append({'id': te[i]['id'], 'text': te[i]['text'], 'aspect': ASPECTS[a], 'gold': g, 'pred': p,
                     'auto type': kind, 'manual error type': '', 'gold label wrong? (y/n)': ''})
    t = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, 'R13_error_sample_for_manual_coding.csv'), index=False, encoding='utf-8-sig')
    return t