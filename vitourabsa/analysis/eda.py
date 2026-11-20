"""Exploratory data analysis + dataset statistics. Every function saves its table (CSV + LaTeX) and/or
figure (PDF + PNG) under OUT and returns the objects so the notebook can display them."""
import math
import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from ..common import ASPECTS, SCORE_NAMES, SCORES

OUT = 'reports/eda'
COLORS = {-2: '#b2182b', -1: '#ef8a62', 0: '#d9d9d9', 1: '#67a9cf', 2: '#2166ac'}
POL_COLORS = {'positive only': '#4393c3', 'negative only': '#d6604d', 'mixed': '#9970ab', 'no polar opinion': '#bababa'}


def _plt():
    import matplotlib
    try:
        get_ipython()
    except NameError:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 110, 'font.family': 'DejaVu Sans'})
    return plt


def _close_if_script(fig):
    """Free figure memory when running as a script (figures stay open in notebooks for display)."""
    try:
        get_ipython()
    except NameError:
        import matplotlib.pyplot as plt
        plt.close(fig)


def save_fig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', dpi=200)
    _close_if_script(fig)


def to_latex(df, path, caption='', label='', float_fmt='{:.2f}'):
    """Minimal LaTeX booktabs writer (no jinja2 dependency)."""
    def f(v):
        if isinstance(v, (float, np.floating)):
            return '--' if np.isnan(v) else float_fmt.format(v)
        return str(v).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    cols = [str(c) for c in df.columns]
    idx_name = df.index.name or ''
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\begin{tabular}{l' + 'r' * len(cols) + '}',
             r'\toprule', ' & '.join([f(idx_name)] + [f(c) for c in cols]) + r' \\', r'\midrule']
    for i, row in df.iterrows():
        lines.append(' & '.join([f(i)] + [f(v) for v in row.values]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if caption:
        lines.append(r'\caption{' + caption + '}')
    if label:
        lines.append(r'\label{' + label + '}')
    lines.append(r'\end{table}')
    with open(path, 'w', encoding='utf-8') as fo:
        fo.write('\n'.join(lines))


def save_table(df, name, caption='', fmt='{:.2f}'):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name + '.csv'), encoding='utf-8-sig')
    to_latex(df, os.path.join(OUT, name + '.tex'), caption=caption, label='tab:' + name, float_fmt=fmt)
    return df


# ====================================================================== helpers
def n_words(s):
    return s.str.split().str.len()


def review_polarity(df):
    pos = (df[ASPECTS] > 0).any(axis=1)
    neg = (df[ASPECTS] < 0).any(axis=1)
    return np.select([pos & neg, pos, neg], ['mixed', 'positive only', 'negative only'], 'no polar opinion')


# ====================================================================== 1. pipeline & overview
def table_pipeline(clean_sum, dedup_sum, near_sum):
    rows = [('Raw records', clean_sum['n_raw'], '')]
    for k, v in clean_sum['removed_by_reason'].items():
        rows.append((f'  − removed: {k.replace("_", " ")}', -v, ''))
    rows.append(('After structural cleaning', clean_sum['n_clean'], ''))
    rows.append(('  exact-duplicate groups (≥2 identical texts)', dedup_sum['dup_groups'],
                 f"{dedup_sum['rows_in_dup_groups']:,} rows"))
    rows.append(('    of which with conflicting labels', dedup_sum['dup_groups_conflicting_labels'], 'majority vote'))
    rows.append(('    of which dropped (tie on some aspect)', -dedup_sum['dup_groups_dropped_tie'], ''))
    rows.append(('After exact deduplication', dedup_sum['n_out'], ''))
    rows.append((f"  − near-duplicates (MinHash Jaccard ≥ {near_sum['threshold']})", -near_sum['removed'], ''))
    rows.append(('Final dataset', near_sum['n_out'], ''))
    df = pd.DataFrame(rows, columns=['Step', 'Count', 'Note']).set_index('Step')
    return save_table(df, 'T01_cleaning_pipeline', 'Data cleaning pipeline.', fmt='{:,.0f}')


def table_overview(df):
    w = n_words(df.text)
    toks = [t for s in df.text for t in s.split()]
    vocab = Counter(toks)
    nz = (df[ASPECTS] != 0)
    pol = pd.Series(review_polarity(df))
    stats = {
        'Reviews': len(df),
        'Provinces': df.province.nunique(),
        'Facility types': df.facility.nunique(),
        'Aspect categories': len(ASPECTS),
        'Sentiment levels': 5,
        'Tokens (syllables)': len(toks),
        'Vocabulary size': len(vocab),
        'Hapax legomena (%)': 100 * sum(1 for c in vocab.values() if c == 1) / len(vocab),
        'Mean length (syllables)': w.mean(),
        'Median length': w.median(),
        'Std length': w.std(),
        'Min / Max length': f'{w.min()} / {w.max()}',
        'Reviews ≤ 3 syllables (%)': 100 * (w <= 3).mean(),
        'Polar aspect opinions (≠0)': int(nz.values.sum()),
        'Mean polar aspects / review': nz.sum(axis=1).mean(),
        'Reviews with ≥2 polar aspects (%)': 100 * (nz.sum(axis=1) >= 2).mean(),
        'Reviews with no polar aspect (%)': 100 * (nz.sum(axis=1) == 0).mean(),
        'Reviews with a negative aspect (%)': 100 * (df[ASPECTS] < 0).any(axis=1).mean(),
        'Mixed-polarity reviews (%)': 100 * (pol == 'mixed').mean(),
        'Slots labelled 0 (%)': 100 * (~nz).values.mean(),
    }
    out = pd.DataFrame({'Value': [f'{v:,.2f}' if isinstance(v, float) else (f'{v:,}' if isinstance(v, (int, np.integer)) else v)
                                  for v in stats.values()]}, index=list(stats.keys()))
    out.index.name = 'Statistic'
    return save_table(out, 'T02_overview', 'Overall dataset statistics.')


def table_by_group(df, col):
    w = n_words(df.text)
    nz = (df[ASPECTS] != 0).sum(axis=1)
    pol = pd.Series(review_polarity(df), index=df.index)
    g = pd.DataFrame({
        'Reviews': df.groupby(col).size(),
        '%': 100 * df.groupby(col).size() / len(df),
        'Mean len': w.groupby(df[col]).mean(),
        'Polar asp./rev.': nz.groupby(df[col]).mean(),
        'Pos-only %': 100 * (pol == 'positive only').groupby(df[col]).mean(),
        'Neg-only %': 100 * (pol == 'negative only').groupby(df[col]).mean(),
        'Mixed %': 100 * (pol == 'mixed').groupby(df[col]).mean(),
    }).sort_values('Reviews', ascending=False)
    g.loc['Total'] = [len(df), 100.0, w.mean(), nz.mean(), 100 * (pol == 'positive only').mean(),
                      100 * (pol == 'negative only').mean(), 100 * (pol == 'mixed').mean()]
    g['Reviews'] = g['Reviews'].astype(int)
    g.index.name = col
    return save_table(g, f'T03_by_{col}', f'Statistics by {col}.')


def table_province_facility(df):
    ct = pd.crosstab(df.province, df.facility, margins=True, margins_name='Total')
    return save_table(ct, 'T04_province_x_facility', 'Reviews per province and facility type.', fmt='{:,.0f}')


# ====================================================================== 2. length
def fig_length(df):
    plt = _plt()
    w = n_words(df.text)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    ax[0].hist(w.clip(upper=300), bins=60, color='#4393c3')
    for q, ls in [(0.5, '-'), (0.9, '--'), (0.99, ':')]:
        ax[0].axvline(w.quantile(q), color='k', ls=ls, lw=0.8, label=f'p{int(q * 100)}={w.quantile(q):.0f}')
    ax[0].set_xlabel('review length (syllables, clipped at 300)'); ax[0].set_ylabel('#reviews'); ax[0].legend()
    order = df.province.value_counts().index
    ax[1].boxplot([w[df.province == p] for p in order], showfliers=False)
    ax[1].set_xticks(range(1, len(order) + 1)); ax[1].set_xticklabels(order, rotation=45, ha='right')
    ax[1].set_ylabel('length'); ax[1].set_title('by province')
    nz = (df[ASPECTS] != 0).sum(axis=1)
    m = w.groupby(nz).median()
    ax[2].bar(m.index, m.values, color='#9970ab'); ax[2].set_xlabel('#polar aspects in review')
    ax[2].set_ylabel('median length'); ax[2].set_title('length vs. #aspects')
    plt.tight_layout(); save_fig(fig, 'F01_length')
    q = w.quantile([.25, .5, .75, .9, .95, .99]).to_frame('syllables')
    return fig, q


# ====================================================================== 3. labels
def table_label_distribution(df):
    rows = {}
    for a in ASPECTS:
        vc = df[a].value_counts().reindex(SCORES).fillna(0).astype(int)
        r = {SCORE_NAMES[i]: vc[s] for i, s in enumerate(SCORES)}
        nz = vc.drop(0).sum()
        r['Polar total'] = nz
        r['Polar %'] = 100 * nz / len(df)
        r['Pos:Neg'] = (vc[1] + vc[2]) / max(1, vc[-1] + vc[-2])
        r['Strong share %'] = 100 * (vc[2] + vc[-2]) / max(1, nz)
        rows[a] = r
    t = pd.DataFrame(rows).T
    tot = t[SCORE_NAMES + ['Polar total']].sum()
    t.loc['All aspects'] = list(tot.values) + [100 * tot['Polar total'] / (len(df) * len(ASPECTS)),
                                               (tot['+1'] + tot['+2']) / (tot['-1'] + tot['-2']),
                                               100 * (tot['+2'] + tot['-2']) / tot['Polar total']]
    t.index.name = 'Aspect'
    for c in SCORE_NAMES + ['Polar total']:
        t[c] = t[c].astype(int)
    return save_table(t, 'T05_label_distribution', 'Label distribution per aspect.')


def fig_label_distribution(df):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [1.3, 1]})
    # diverging bar among polar labels
    order = sorted(ASPECTS, key=lambda a: (df[a] != 0).mean())
    for i, a in enumerate(order):
        vc = df[a].value_counts(normalize=True).reindex(SCORES).fillna(0)
        left = -(vc[-1] + vc[-2])
        for s in [-2, -1, 1, 2]:
            ax[0].barh(i, vc[s], left=left, color=COLORS[s], label=SCORE_NAMES[s + 2] if i == 0 else None)
            left += vc[s]
    ax[0].axvline(0, color='k', lw=0.6)
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order)
    ax[0].set_xlabel('share of all reviews (negative ← | → positive)')
    ax[0].set_title('Polar labels per aspect (0 omitted)'); ax[0].legend(ncol=4, fontsize=7, loc='lower right')
    nz = (df[ASPECTS] != 0).sum(axis=1).value_counts().sort_index()
    ax[1].bar(nz.index, nz.values, color='#4393c3')
    for x, y in zip(nz.index, nz.values):
        ax[1].text(x, y, f'{100 * y / len(df):.1f}%', ha='center', va='bottom', fontsize=7)
    ax[1].set_xlabel('#aspects with a polar opinion'); ax[1].set_ylabel('#reviews')
    ax[1].set_title('Aspects per review')
    plt.tight_layout(); save_fig(fig, 'F02_label_distribution')
    return fig


def fig_cooccurrence(df):
    plt = _plt()
    M = (df[ASPECTS] != 0).values.astype(float)
    n_a = M.sum(0)
    co = M.T @ M
    cond = co / n_a[:, None]                       # P(col | row)
    p = n_a / len(M)
    lift = cond / p[None, :]
    np.fill_diagonal(cond, np.nan)
    np.fill_diagonal(lift, np.nan)
    S = np.full((len(ASPECTS), len(ASPECTS)), np.nan)
    N = np.zeros_like(S)
    from scipy.stats import spearmanr
    X = df[ASPECTS].values
    for i in range(len(ASPECTS)):
        for j in range(len(ASPECTS)):
            m = (X[:, i] != 0) & (X[:, j] != 0)
            N[i, j] = m.sum()
            if i != j and m.sum() > 30:
                S[i, j] = spearmanr(X[m, i], X[m, j]).correlation
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for k, (Mx, title, cmap, vmin, vmax) in enumerate([
            (cond, 'P(column polar | row polar)', 'Blues', 0, 1),
            (lift, 'Lift of co-occurrence', 'RdBu_r', 0, 2),
            (S, 'Spearman ρ of scores (both polar)', 'RdBu_r', -1, 1)]):
        im = ax[k].imshow(Mx, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[k].set_xticks(range(len(ASPECTS))); ax[k].set_xticklabels(ASPECTS, rotation=60, ha='right', fontsize=7)
        ax[k].set_yticks(range(len(ASPECTS))); ax[k].set_yticklabels(ASPECTS, fontsize=7)
        ax[k].set_title(title)
        for i in range(len(ASPECTS)):
            for j in range(len(ASPECTS)):
                if not np.isnan(Mx[i, j]):
                    ax[k].text(j, i, f'{Mx[i, j]:.2f}', ha='center', va='center', fontsize=5.5)
        fig.colorbar(im, ax=ax[k], shrink=0.75)
    plt.tight_layout(); save_fig(fig, 'F03_cooccurrence_correlation')
    t = pd.DataFrame(S, index=ASPECTS, columns=ASPECTS)
    save_table(t, 'T06_score_correlation', 'Spearman correlation between aspect scores (reviews where both are polar).')
    return fig


# ====================================================================== 4. heterogeneity across provinces / facilities
def province_aspect(df, col='province'):
    plt = _plt()
    from scipy.stats import chi2_contingency
    groups = df[col].value_counts().index.tolist()
    mention = pd.DataFrame({a: (df[a] != 0).groupby(df[col]).mean() * 100 for a in ASPECTS}).loc[groups]
    meanpol = pd.DataFrame({a: df[a][df[a] != 0].groupby(df[col][df[a] != 0]).mean() for a in ASPECTS}).loc[groups]
    neg = pd.DataFrame({a: (df[a] < 0).groupby(df[col]).sum() / (df[a] != 0).groupby(df[col]).sum().clip(lower=1) * 100
                        for a in ASPECTS}).loc[groups]
    tests = []
    for a in ASPECTS:
        ct = pd.crosstab(df[col], df[a])
        chi2, p, dof, _ = chi2_contingency(ct)
        v = math.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        tests.append({'aspect': a, 'chi2': chi2, 'dof': dof, 'p_value': p, "Cramér's V": v})
    tests = pd.DataFrame(tests).set_index('aspect')
    save_table(tests, f'T07_chi2_{col}', f'Chi-square test of label distribution across {col}s.', fmt='{:.3g}')
    fig, ax = plt.subplots(1, 3, figsize=(18, 0.45 * len(groups) + 2.5))
    for k, (M, title, cmap, vmin, vmax, fmt) in enumerate([
            (mention, 'Polar-mention rate (%)', 'Blues', 0, 100, '{:.0f}'),
            (meanpol, 'Mean score among polar labels', 'RdBu', -2, 2, '{:.2f}'),
            (neg, 'Negative share among polar labels (%)', 'Reds', 0, 60, '{:.0f}')]):
        im = ax[k].imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax[k].set_xticks(range(len(ASPECTS))); ax[k].set_xticklabels(ASPECTS, rotation=60, ha='right', fontsize=7)
        ax[k].set_yticks(range(len(groups))); ax[k].set_yticklabels(groups, fontsize=7); ax[k].set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M.values[i, j]):
                    ax[k].text(j, i, fmt.format(M.values[i, j]), ha='center', va='center', fontsize=6)
        fig.colorbar(im, ax=ax[k], shrink=0.8)
    plt.tight_layout(); save_fig(fig, f'F04_{col}_x_aspect')
    return fig, tests


def fig_review_polarity(df):
    plt = _plt()
    pol = pd.Series(review_polarity(df), index=df.index)
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    for k, col in enumerate(['province', 'facility']):
        ct = pd.crosstab(df[col], pol, normalize='index')[list(POL_COLORS)] * 100
        ct = ct.loc[df[col].value_counts().index]
        left = np.zeros(len(ct))
        for c in POL_COLORS:
            ax[k].barh(range(len(ct)), ct[c], left=left, color=POL_COLORS[c], label=c)
            left += ct[c].values
        ax[k].set_yticks(range(len(ct))); ax[k].set_yticklabels(ct.index); ax[k].set_xlabel('% of reviews')
        ax[k].set_title(f'Review-level polarity by {col}')
    ax[1].legend(fontsize=7, loc='lower right')
    plt.tight_layout(); save_fig(fig, 'F05_review_polarity')
    return fig


# ====================================================================== 5. lexical analysis
def _log_odds(c1, c2, prior, a0=None):
    """Monroe et al. (2008) log-odds ratio with informative Dirichlet prior; returns z-scores."""
    n1, n2 = sum(c1.values()), sum(c2.values())
    a0 = a0 or sum(prior.values())
    z = {}
    for w, aw in prior.items():
        y1, y2 = c1.get(w, 0), c2.get(w, 0)
        d = math.log((y1 + aw) / (n1 + a0 - y1 - aw)) - math.log((y2 + aw) / (n2 + a0 - y2 - aw))
        var = 1 / (y1 + aw) + 1 / (y2 + aw)
        z[w] = d / math.sqrt(var)
    return z


def lexical_markers(df, k=8, min_count=20):
    toks = (df.text_seg if 'text_seg' in df else df.text).str.split()   # word-segmented: 'nhân_viên'
    prior = Counter(t for s in toks for t in s)
    prior = {w: c * 0.01 for w, c in prior.items() if c >= min_count and any(ch.isalpha() for ch in w)}
    rows = []
    for a in ASPECTS:
        pos = Counter(t for s in toks[df[a] > 0] for t in s)
        neg = Counter(t for s in toks[df[a] < 0] for t in s)
        men = Counter(t for s in toks[df[a] != 0] for t in s)
        non = Counter(t for s in toks[df[a] == 0] for t in s)
        z_m = _log_odds(men, non, prior)
        z_p = _log_odds(pos, neg, prior)
        top = lambda z, rev=False: ', '.join(w for w, _ in sorted(z.items(), key=lambda x: x[1], reverse=not rev)[:k])
        rows.append({'aspect': a, 'aspect-indicative (polar vs 0)': top(z_m),
                     'positive markers': top(z_p), 'negative markers': top(z_p, rev=True)})
    t = pd.DataFrame(rows).set_index('aspect')
    return save_table(t, 'T08_lexical_markers', 'Most distinctive words per aspect (log-odds ratio with informative Dirichlet prior, Monroe et al. 2008).')


# ====================================================================== 6. duplicates / annotation consistency
def duplicates_report(with_dups, dedup_sum):
    plt = _plt()
    vc = with_dups.text.value_counts()
    top = vc.head(15).to_frame('copies')
    top.index.name = 'text'
    save_table(top, 'T09_top_duplicates', 'Most frequent duplicated reviews (before deduplication).', fmt='{:,.0f}')
    ag = pd.Series(dedup_sum['pairwise_agreement_in_dup_groups']).astype(float) * 100
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    sizes = vc[vc > 1]
    ax[0].hist(sizes.clip(upper=30), bins=29, color='#9970ab'); ax[0].set_yscale('log')
    ax[0].set_xlabel('copies per duplicated text (clipped at 30)'); ax[0].set_ylabel('#groups (log)')
    ax[1].bar(ag.index, ag.values, color='#4393c3'); ax[1].set_ylim(min(80, ag.min() - 2), 100)
    ax[1].set_xticks(range(len(ag))); ax[1].set_xticklabels(ag.index, rotation=45, ha='right')
    ax[1].set_ylabel('% agreeing label pairs'); ax[1].set_title('Label agreement among identical texts')
    for i, v in enumerate(ag.values):
        ax[1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout(); save_fig(fig, 'F06_duplicates')
    agt = ag.to_frame('pairwise agreement (%)')
    agt.index.name = 'aspect'
    save_table(agt, 'T10_duplicate_agreement', 'Pairwise label agreement among identical review texts.')
    return fig, top, agt


# ====================================================================== 7. text characteristics
EN_WORDS = re.compile(r'\b(?:the|and|very|good|nice|room|staff|hotel|great|clean|location|view|ok|check|service)\b')


def text_characteristics(df, train_texts=None, test_texts=None):
    def no_diac(s):
        letters = [c for c in s if c.isalpha()]
        if not letters:
            return False
        d = sum(1 for c in letters if unicodedata.normalize('NFD', c) != c or c in 'đĐ')
        return d / len(letters) < 0.02
    w = n_words(df.text)
    r = {
        'Reviews containing English words (%)': 100 * df.text.str.contains(EN_WORDS).mean(),
        'Reviews (almost) without diacritics (%)': 100 * df.text.map(no_diac).mean(),
        'Reviews containing digits (%)': 100 * df.text.str.contains(r'\d').mean(),
        'Reviews ≤ 3 syllables (%)': 100 * (w <= 3).mean(),
        'Reviews > 128 syllables (%)': 100 * (w > 128).mean(),
    }
    if train_texts is not None:
        tv = Counter(t for s in train_texts for t in s.split())
        tt = [t for s in test_texts for t in s.split()]
        r['Test OOV token rate vs. train (%)'] = 100 * sum(1 for t in tt if t not in tv) / max(1, len(tt))
        r['Test OOV type rate vs. train (%)'] = 100 * len({t for t in tt if t not in tv}) / max(1, len(set(tt)))
    t = pd.DataFrame({'Value': r})
    t.index.name = 'Characteristic'
    return save_table(t, 'T11_text_characteristics', 'Text characteristics.')


# ====================================================================== 8. split statistics
def _js(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))


def split_statistics(split_root, names):
    from ..common import labels_of, read_jsonl
    rows, rare = [], []
    for name in names:
        d = os.path.join(split_root, name)
        parts = {p: labels_of(read_jsonl(os.path.join(d, f'{p}.jsonl'))) for p in ('train', 'val', 'test')}
        r = {'split': name}
        for p, Y in parts.items():
            r[f'n_{p}'] = len(Y)
        js = []
        for a in range(len(ASPECTS)):
            ptr = np.bincount(parts['train'][:, a] + 2, minlength=5)
            pte = np.bincount(parts['test'][:, a] + 2, minlength=5)
            js.append(_js(ptr, pte))
        r['mean JS(train,test)'] = float(np.mean(js))
        r['max JS(train,test)'] = float(np.max(js))
        r['max-JS aspect'] = ASPECTS[int(np.argmax(js))]
        rows.append(r)
        if name == 'S1_main':
            for a_i, a in enumerate(ASPECTS):
                for s in SCORES:
                    rare.append({'aspect': a, 'label': SCORE_NAMES[s + 2],
                                 **{p: int((Y[:, a_i] == s).sum()) for p, Y in parts.items()}})
    t = pd.DataFrame(rows).set_index('split')
    save_table(t, 'T12_splits', 'Split sizes and label-distribution shift (Jensen-Shannon divergence, bits).',
               fmt='{:.4f}')
    rt = pd.DataFrame(rare)
    rt = rt[rt.test < 60].set_index(['aspect', 'label'])
    save_table(rt, 'T13_rare_cells', 'Rare (aspect, label) cells in the main split (test support < 60).',
               fmt='{:,.0f}')
    return t, rt