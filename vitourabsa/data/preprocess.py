"""Cleaning, deduplication (majority vote), near-duplicate removal, segmentation and all splits."""
import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from ..common import ASPECTS, BIG_PROVINCES, PROVINCES, SMALL_PROVINCES, save_json, write_jsonl

FACILITIES = ['hotel', 'resort', 'hostel']


def norm_text(s):
    s = unicodedata.normalize('NFC', str(s))
    return re.sub(r'\s+', ' ', s).strip()


# ------------------------------------------------------------------ 1. structural cleaning
def clean(df):
    """Remove rows with corrupted / missing labels, invalid metadata or empty text.
    Returns (clean_df, removed_df with a `reason` column, summary dict)."""
    df = df.copy()
    df['text'] = df['data'].map(norm_text)
    reasons = pd.Series('', index=df.index)
    num = df[ASPECTS].apply(pd.to_numeric, errors='coerce')

    def mark(mask, why):
        m = mask & (reasons == '')
        reasons[m] = why

    mark(df['text'].str.len() == 0, 'empty_text')
    mark(df[ASPECTS].apply(lambda c: c.notna() & pd.to_numeric(c, errors='coerce').isna()).any(axis=1),
         'non_numeric_label')
    mark((num.abs() > 2).any(axis=1) | ((num != num.round()) & num.notna()).any(axis=1), 'out_of_range_label')
    mark(num.isna().any(axis=1), 'missing_label')
    mark(~df['province'].isin(PROVINCES), 'invalid_province')
    mark(~df['facility_type'].isin(FACILITIES), 'invalid_facility')
    removed = df[reasons != ''].assign(reason=reasons[reasons != ''])
    out = df[reasons == ''].copy()
    out[ASPECTS] = num.loc[out.index].astype(int)
    out = out.rename(columns={'facility_type': 'facility'})
    summary = {'n_raw': int(len(df)), 'n_removed': int(len(removed)),
               'removed_by_reason': removed['reason'].value_counts().to_dict(), 'n_clean': int(len(out))}
    return out.reset_index(drop=True), removed, summary


# ------------------------------------------------------------------ 2. exact-duplicate consolidation
def dedup_majority(df):
    """Group identical texts. Label of each aspect = majority vote inside the group.
    A group with a tie on ANY aspect is dropped. Also measures within-group label agreement
    (a natural estimate of annotation consistency)."""
    g = df.groupby('text', sort=False)
    sizes = g.size()
    multi = sizes[sizes > 1].index
    keep_rows, dropped_tie, conflict_groups = [], 0, 0
    agree = {a: [0, 0] for a in ASPECTS}            # pairwise agreeing pairs / all pairs, per aspect
    exact_agree_groups = 0
    multi_prov = 0
    for text, grp in g:
        if len(grp) == 1:
            r = grp.iloc[0].to_dict()
            r['dup_count'] = 1
            keep_rows.append(r)
            continue
        labs = grp[ASPECTS].values
        for j, a in enumerate(ASPECTS):
            c = Counter(labs[:, j].tolist())
            n = len(labs)
            agree[a][0] += sum(v * (v - 1) // 2 for v in c.values())
            agree[a][1] += n * (n - 1) // 2
        conflict = (labs != labs[0]).any()
        conflict_groups += int(conflict)
        exact_agree_groups += int(not conflict)
        multi_prov += int(grp['province'].nunique() > 1)
        maj, tie = [], False
        for j in range(len(ASPECTS)):
            mc = Counter(labs[:, j].tolist()).most_common()
            if len(mc) > 1 and mc[0][1] == mc[1][1]:
                tie = True
                break
            maj.append(mc[0][0])
        if tie:
            dropped_tie += 1
            continue
        r = grp.iloc[0].to_dict()
        r.update(dict(zip(ASPECTS, maj)))
        r['province'] = grp['province'].mode().iloc[0]
        r['facility'] = grp['facility'].mode().iloc[0]
        r['dup_count'] = int(len(grp))
        keep_rows.append(r)
    out = pd.DataFrame(keep_rows)
    summary = {
        'n_in': int(len(df)), 'n_unique_texts': int(len(sizes)),
        'dup_groups': int(len(multi)), 'rows_in_dup_groups': int(sizes[sizes > 1].sum()),
        'dup_groups_conflicting_labels': int(conflict_groups),
        'dup_groups_fully_consistent': int(exact_agree_groups),
        'dup_groups_spanning_provinces': int(multi_prov),
        'dup_groups_dropped_tie': int(dropped_tie),
        'n_out': int(len(out)),
        'pairwise_agreement_in_dup_groups': {a: (v[0] / v[1] if v[1] else None) for a, v in agree.items()},
    }
    return out.reset_index(drop=True), summary


# ------------------------------------------------------------------ 3. near-duplicates
def near_dedup(df, threshold=0.85, min_words=6, num_perm=64):
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = np.ones(len(df), dtype=bool)
    examples = []
    texts = df['text'].tolist()
    for i, t in enumerate(texts):
        toks = t.split()
        if len(toks) < min_words:
            continue
        m = MinHash(num_perm=num_perm)
        for j in range(len(toks) - 2):
            m.update(' '.join(toks[j:j + 3]).encode('utf-8'))
        hit = lsh.query(m)
        if hit:
            keep[i] = False
            if len(examples) < 10:
                examples.append({'removed': t, 'kept': texts[int(hit[0])]})
        else:
            lsh.insert(str(i), m)
    out = df[keep].reset_index(drop=True)
    return out, {'threshold': threshold, 'removed': int((~keep).sum()), 'n_out': int(len(out)),
                 'examples': examples}


# ------------------------------------------------------------------ 4. segmentation (PhoBERT)
def segment_all(texts):
    from pyvi import ViTokenizer
    uniq = list(dict.fromkeys(texts))
    seg = {t: ViTokenizer.tokenize(t) for t in uniq}
    return [seg[t] for t in texts]


# ------------------------------------------------------------------ 5. splits
def _onehot(df):
    Y = df[ASPECTS].values + 2
    oh = np.zeros((len(df), len(ASPECTS) * 5), dtype=np.int8)
    for a in range(len(ASPECTS)):
        oh[np.arange(len(df)), a * 5 + Y[:, a]] = 1
    return oh[:, oh.sum(0) > 0]


def strat_split(df, test_size, seed):
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    s = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(s.split(np.zeros(len(df)), _onehot(df)))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)


def to_rows(df, extra=()):
    rows = []
    for r in df.itertuples(index=False):
        d = {'id': int(r.review_id), 'text': r.text, 'text_seg': r.text_seg,
             'labels': [int(getattr(r, a)) for a in ASPECTS], 'province': r.province, 'facility': r.facility,
             'dup_count': int(getattr(r, 'dup_count', 1))}
        for e in extra:
            d[e] = getattr(r, e)
        rows.append(d)
    return rows


def write_split(root, name, tr, va, te, extra=()):
    d = os.path.join(root, name)
    for part, x in [('train', tr), ('val', va), ('test', te)]:
        write_jsonl(os.path.join(d, f'{part}.jsonl'), to_rows(x, extra))
    return {'train': len(tr), 'val': len(va), 'test': len(te)}


def build_all_splits(final, with_dups, root, seed=42):
    """final: deduplicated clean data; with_dups: clean data BEFORE dedup (for S6)."""
    info = {}
    # S1 main 80/10/10
    tr, rest = strat_split(final, 0.2, seed)
    va, te = strat_split(rest, 0.5, seed)
    info['S1_main'] = write_split(root, 'S1_main', tr, va, te)
    # S5 learning-curve: nested subsets of S1 train
    order = np.random.RandomState(seed).permutation(len(tr))
    for f in [0.01, 0.05, 0.1, 0.25, 0.5]:
        sub = tr.iloc[np.sort(order[:max(50, int(f * len(tr)))])]
        info[f'S5_frac{f:g}'] = write_split(root, f'S5_frac{f:g}', sub, va, te)
    # S2 out-of-region: train on big provinces, test on small provinces
    big = final[final.province.isin(BIG_PROVINCES)]
    small = final[final.province.isin(SMALL_PROVINCES)]
    btr, bva = strat_split(big, 0.1, seed)
    info['S2_out_of_region'] = write_split(root, 'S2_out_of_region', btr, bva, small)
    # S3 leave-one-province-out (big provinces)
    for p in BIG_PROVINCES:
        rest_p = final[final.province.isin([q for q in BIG_PROVINCES if q != p])]
        ptr, pva = strat_split(rest_p, 0.1, seed)
        info[f'S3_lopo_{p}'] = write_split(root, f'S3_lopo_{p}', ptr, pva, final[final.province == p])
    # S4 cross-facility
    for src, tgt in [('hotel', 'resort'), ('resort', 'hotel')]:
        s = final[final.facility == src]
        str_, sva = strat_split(s, 0.1, seed)
        info[f'S4_{src}_to_{tgt}'] = write_split(root, f'S4_{src}_to_{tgt}', str_, sva, final[final.facility == tgt])
    # S6 duplicate leakage: random stratified split of data WITH duplicates
    w = with_dups.copy()
    w['dup_count'] = 1
    wtr, wrest = strat_split(w, 0.2, seed)
    wva, wte = strat_split(wrest, 0.5, seed)
    seen = set(wtr.text)
    wte = wte.assign(seen_in_train=wte.text.isin(seen))
    wva = wva.assign(seen_in_train=wva.text.isin(seen))
    wtr = wtr.assign(seen_in_train=True)
    info['S6_with_duplicates'] = write_split(root, 'S6_with_duplicates', wtr, wva, wte, extra=('seen_in_train',))
    info['S6_test_seen_rate'] = float(wte.seen_in_train.mean())
    save_json(info, os.path.join(root, 'split_info.json'))
    return info