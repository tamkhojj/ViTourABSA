"""Additional analyses used in the paper. They only read existing results (no training).

X01  generalisation measured on identical S1 test reviews (separates distribution shift from test composition)
X02  learning-curve table
X03  hyper-parameters actually used (chosen learning rate, epochs, time per epoch), QLoRA and seq2seq settings
X04  LLM detection precision / recall and rate of predicted polar slots
X05  95% bootstrap confidence intervals (seed 1)
X06  per-level F1 pooled over aspects
"""
import glob
import os

import numpy as np
import pandas as pd

from .. import paths
from ..common import BIG_PROVINCES, PROVINCE_VI, SCORE_NAMES, SMALL_PROVINCES, load_json, read_jsonl
from ..metrics import PRIMARY, bootstrap_ci, evaluate
from .eda import to_latex

NAMES = {'classic_tfidf_lr': 'TF-IDF + LR', 'phobert_base': 'PhoBERT-base-v2', 'xlmr_large': 'XLM-R-large',
         'phobert_large': 'PhoBERT-large', 'llm_qwen25_7b_zero': 'Qwen2.5-7B (zero-shot)',
         'phobert_base_pair': 'PhoBERT-base · pair', 'phobert_base_weighted': 'PhoBERT-base · weighted CE',
         'phobert_base_coral': 'PhoBERT-base · CORAL', 'phobert_base_logit_adjust': 'PhoBERT-base · logit adjustment',
         'llm_vistral_7b_qlora': 'Vistral-7B (QLoRA)', 'llm_qwen25_7b_qlora': 'Qwen2.5-7B (QLoRA)',
         'llm_llama31_8b_few': 'Llama-3.1-8B (10-shot)', 'vit5_base': 'ViT5-base'}


def save(df, name, caption, out=None):
    out = out or paths.EXTRA_OUT
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, name + '.csv'), encoding='utf-8-sig')
    to_latex(df, os.path.join(out, name + '.tex'), caption, 'tab:' + name)
    return df


def preds(split, run):
    """[(ids, y, pred)] for every seed of a run on a split."""
    rows = read_jsonl(os.path.join(paths.split_dir(split), 'test.jsonl'))
    ids = np.array([r['id'] for r in rows])
    out = []
    for d in sorted(glob.glob(os.path.join(paths.RESULTS, split, run, 'seed*'))):
        f = os.path.join(d, 'preds_test.npz')
        if os.path.exists(f):
            z = np.load(f)
            if len(z['y']) == len(ids):
                out.append((ids, z['y'], z['pred']))
    return out


def subset_scores(split, run, keep_ids):
    vals, n = [], 0
    for ids, y, p in preds(split, run):
        m = np.isin(ids, keep_ids)
        if m.sum():
            vals.append(evaluate(y[m], p[m], per_aspect=False))
            n = int(m.sum())
    if not vals:
        return None
    return {k: 100 * np.mean([v[k] for v in vals]) for k in (PRIMARY, 'joint_f1_macro', 't1_f1_micro')} | \
        {'n': n, 'seeds': len(vals)}


def shift_same_reviews(runs=('classic_tfidf_lr', 'phobert_base', 'xlmr_large', 'llm_qwen25_7b_zero')):
    """X01. For each shift split, score the S1 model and the shift model on the S1 test reviews of the held-out group.
    Zero-shot LLMs receive no training data, so their two scores should coincide (sanity check)."""
    s1 = read_jsonl(os.path.join(paths.split_dir('S1_main'), 'test.jsonl'))
    cases = [('S2_out_of_region', 'Cross-region (4 lower-volume provinces)', lambda r: r['province'] in SMALL_PROVINCES),
             ('S4_hotel_to_resort', 'Hotel to resort', lambda r: r['facility'] == 'resort'),
             ('S4_resort_to_hotel', 'Resort to hotel', lambda r: r['facility'] == 'hotel')]
    cases += [(f'S3_lopo_{p}', f'Held out: {PROVINCE_VI[p]}', (lambda p: lambda r: r['province'] == p)(p))
              for p in BIG_PROVINCES]
    rows = []
    for split, label, cond in cases:
        if not os.path.isdir(paths.split_dir(split)):
            continue
        keep = np.array([r['id'] for r in s1 if cond(r)])
        for run in runs:
            a, b = subset_scores('S1_main', run, keep), subset_scores(split, run, keep)
            if a is None or b is None:
                continue
            rows.append({'Shift': label, 'Model': NAMES.get(run, run), 'n reviews': a['n'],
                         'Seen (trained on S1)': a[PRIMARY], 'Unseen (shift split)': b[PRIMARY],
                         'Δ T3': b[PRIMARY] - a[PRIMARY], 'Joint seen': a['joint_f1_macro'],
                         'Joint unseen': b['joint_f1_macro'], 'Δ Joint': b['joint_f1_macro'] - a['joint_f1_macro']})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index(['Shift', 'Model']).round(2)
    return save(df, 'X01_shift_same_reviews', 'Generalisation measured on identical test reviews (T3 macro-F1 / joint F1).')


def learning_curve(runs=('classic_tfidf_lr', 'phobert_base', 'xlmr_large'),
                   fractions=(0.01, 0.05, 0.1, 0.25, 0.5)):
    """X02."""
    rows = []
    for run in runs:
        for f, split in [(f, f'S5_frac{f:g}') for f in fractions] + [(1.0, 'S1_main')]:
            v = [evaluate(y, p, per_aspect=False) for _, y, p in preds(split, run)]
            if not v:
                continue
            n_tr = sum(1 for _ in open(os.path.join(paths.split_dir(split), 'train.jsonl'), encoding='utf-8'))
            rows.append({'Model': NAMES.get(run, run), 'Fraction': f, 'Train reviews': n_tr, 'Seeds': len(v),
                         'T3 mean': 100 * np.mean([x[PRIMARY] for x in v]),
                         'T3 std': 100 * np.std([x[PRIMARY] for x in v], ddof=1) if len(v) > 1 else np.nan,
                         'Joint mean': 100 * np.mean([x['joint_f1_macro'] for x in v])})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index(['Model', 'Fraction']).round(2)
    return save(df, 'X02_learning_curve', 'Learning curve (T3 macro-F1 and joint F1).')


def hyperparameters():
    """X03. Settings actually used, read from the saved configs and metrics."""
    rows = []
    for f in sorted(glob.glob(os.path.join(paths.RESULTS, 'S1_main', '*', 'chosen_lr.json'))):
        run = f.split(os.sep)[-2]
        c = load_json(f)
        ms = [load_json(m) for m in glob.glob(os.path.join(paths.RESULTS, 'S1_main', run, 'seed*', 'metrics.json'))]
        cfg = load_json(glob.glob(os.path.join(paths.RESULTS, 'S1_main', run, 'seed*', 'config.json'))[0])
        rows.append({'run': run, 'model': cfg.get('model_name'), 'chosen lr': c['chosen_lr'],
                     'val score per lr': {k: round(v, 4) for k, v in c['val_scores'].items()},
                     'batch': cfg.get('batch_size'), 'grad_accum': cfg.get('grad_accum'), 'max_len': cfg.get('max_len'),
                     'epochs run (mean)': np.mean([m['epochs_run'] for m in ms]),
                     'sec/epoch': np.mean([m['sec_per_epoch'] for m in ms])})
    for run in ['phobert_base_pair', 'phobert_base_coral', 'phobert_base_weighted', 'phobert_base_logit_adjust',
                'xlmr_large_pair']:
        cf = glob.glob(os.path.join(paths.RESULTS, 'S1_main', run, 'seed*', 'config.json'))
        ms = [load_json(m) for m in glob.glob(os.path.join(paths.RESULTS, 'S1_main', run, 'seed*', 'metrics.json'))]
        if cf:
            cfg = load_json(cf[0])
            rows.append({'run': run, 'model': cfg['model_name'], 'chosen lr': cfg['lr'], 'batch': cfg.get('batch_size'),
                         'grad_accum': cfg.get('grad_accum'), 'max_len': cfg.get('max_len'),
                         'epochs run (mean)': np.mean([m['epochs_run'] for m in ms]) if ms else None,
                         'sec/epoch': np.mean([m['sec_per_epoch'] for m in ms]) if ms else None})
    hp = pd.DataFrame(rows).set_index('run') if rows else pd.DataFrame()
    os.makedirs(paths.EXTRA_OUT, exist_ok=True)
    hp.to_csv(os.path.join(paths.EXTRA_OUT, 'X03_hyperparameters.csv'), encoding='utf-8-sig')
    other = {}
    for f in glob.glob(os.path.join(paths.CKPT, '*', 'qlora_config.json')):
        other[f] = load_json(f)
    for f in glob.glob(os.path.join(paths.RESULTS, 'S1_main', 'vit5_base', 'seed*', 'config.json')):
        other[f] = load_json(f)
    return hp, other


def llm_detection():
    """X04."""
    rows = []
    for f in sorted(glob.glob(os.path.join(paths.RESULTS, 'S1_main', 'llm_*', 'seed1', 'metrics.json'))):
        m = load_json(f)
        t = m['test']
        run = f.split(os.sep)[-3]
        z = np.load(os.path.join(os.path.dirname(f), 'preds_test.npz'))
        rows.append({'run': run, 'T1 P': 100 * t['t1_p'], 'T1 R': 100 * t['t1_r'], 'T1 F1': 100 * t['t1_f1_micro'],
                     'pred polar slots %': 100 * (z['pred'] != 0).mean(), 'gold polar slots %': 100 * (z['y'] != 0).mean(),
                     'T2 Macro-F1': 100 * t['t2_macro_f1'], 'Joint F1': 100 * t['joint_f1_macro'],
                     'Invalid %': 100 * m.get('invalid_rate', 0)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index('run').round(2)
    return save(df, 'X04_llm_detection', 'LLM detection precision/recall and polar-prediction rate.')


def bootstrap_intervals(runs=('phobert_base_pair', 'phobert_base_weighted', 'xlmr_large', 'phobert_large',
                              'phobert_base', 'llm_vistral_7b_qlora', 'llm_qwen25_7b_qlora', 'vit5_base',
                              'classic_tfidf_lr', 'llm_llama31_8b_few')):
    """X05. 95% percentile bootstrap intervals (seed 1, 1,000 resamples)."""
    rows = []
    for run in runs:
        p = preds('S1_main', run)
        if not p:
            continue
        _, y, pr = p[0]
        r = {'Model': NAMES.get(run, run)}
        for met in [PRIMARY, 'joint_f1_macro']:
            lo, hi = bootstrap_ci(y, pr, met)
            r[f'{met} (seed1)'] = 100 * evaluate(y, pr, per_aspect=False)[met]
            r[f'{met} 95% CI'] = f'[{100 * lo:.2f}, {100 * hi:.2f}]'
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return save(pd.DataFrame(rows).set_index('Model'), 'X05_bootstrap_ci', 'Bootstrap 95% confidence intervals.')


def per_class_f1(runs=('phobert_base_pair', 'phobert_base_weighted', 'xlmr_large', 'phobert_base_coral',
                       'phobert_base_logit_adjust', 'llm_vistral_7b_qlora', 'llm_llama31_8b_few', 'classic_tfidf_lr')):
    """X06. Per-level F1 pooled over aspects, averaged over seeds."""
    rows = []
    for run in runs:
        ms = [load_json(f)['test'] for f in glob.glob(os.path.join(paths.RESULTS, 'S1_main', run, 'seed*', 'metrics.json'))]
        if ms:
            rows.append({'Model': NAMES.get(run, run),
                         **{c: 100 * np.mean([m[f'f1_class_{c}'] for m in ms]) for c in SCORE_NAMES}})
    if not rows:
        return pd.DataFrame()
    return save(pd.DataFrame(rows).set_index('Model').round(2), 'X06_per_class_f1', 'Per-level F1 (pooled over aspects).')
