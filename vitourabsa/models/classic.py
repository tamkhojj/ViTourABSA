"""Floor baselines and TF-IDF linear models (one 5-way classifier per aspect)."""
import os
import time

import numpy as np

from ..common import ASPECTS, labels_of, read_jsonl, save_json
from ..metrics import evaluate


def _save(out_dir, name, seed, y, p, extra=None, yv=None, pv=None):
    os.makedirs(out_dir, exist_ok=True)
    res = {'run': name, 'seed': seed, 'test': evaluate(y, p), **(extra or {})}
    if yv is not None:
        res['val'] = evaluate(yv, pv)
        np.savez_compressed(os.path.join(out_dir, 'preds_val.npz'), y=yv, pred=pv)
    np.savez_compressed(os.path.join(out_dir, 'preds_test.npz'), y=y, pred=p)
    save_json(res, os.path.join(out_dir, 'metrics.json'))
    return res


def majority(Ytr, n):
    return np.tile([np.bincount(Ytr[:, a] + 2, minlength=5).argmax() - 2 for a in range(len(ASPECTS))], (n, 1))


def prior_random(Ytr, n, seed):
    rng = np.random.RandomState(seed)
    out = np.zeros((n, len(ASPECTS)), dtype=int)
    for a in range(len(ASPECTS)):
        pr = np.bincount(Ytr[:, a] + 2, minlength=5) / len(Ytr)
        out[:, a] = rng.choice(5, n, p=pr) - 2
    return out


def tfidf_features(tr, others):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True, token_pattern=r'\S+')
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=300000, sublinear_tf=True)
    Xtr = hstack([w.fit_transform([r['text_seg'] for r in tr]), c.fit_transform([r['text'] for r in tr])]).tocsr()
    Xo = [hstack([w.transform([r['text_seg'] for r in o]), c.transform([r['text'] for r in o])]).tocsr() for o in others]
    return Xtr, Xo


def tfidf_model(kind, Xtr, Ytr, Xs, seed, C=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    preds = [np.zeros((X.shape[0], len(ASPECTS)), dtype=int) for X in Xs]
    for a in range(len(ASPECTS)):
        clf = (LogisticRegression(C=C or 4.0, max_iter=3000, random_state=seed) if kind == 'lr'
               else LinearSVC(C=C or 0.5, random_state=seed))
        clf.fit(Xtr, Ytr[:, a])
        for k, X in enumerate(Xs):
            preds[k][:, a] = clf.predict(X)
    return preds


def run_classic(split_dir, results_root, seeds=(1,), which=('majority', 'prior', 'tfidf_lr', 'tfidf_svm'), log=print,
                max_train=0):
    tr = read_jsonl(os.path.join(split_dir, 'train.jsonl'))
    if max_train:
        tr = tr[:max_train]
    va = read_jsonl(os.path.join(split_dir, 'val.jsonl'))
    te = read_jsonl(os.path.join(split_dir, 'test.jsonl'))
    Ytr, Yva, Yte = labels_of(tr), labels_of(va), labels_of(te)
    feats = None
    out = {}
    for w in which:
        for s in (seeds if w == 'prior' else [1]):         # deterministic models: one run
            d = os.path.join(results_root, f'classic_{w}', f'seed{s}')
            if os.path.exists(os.path.join(d, 'metrics.json')):
                continue
            t0 = time.time()
            if w == 'majority':
                pv, pt = majority(Ytr, len(va)), majority(Ytr, len(te))
            elif w == 'prior':
                pv, pt = prior_random(Ytr, len(va), s), prior_random(Ytr, len(te), s + 1000)
            else:
                if feats is None:
                    feats = tfidf_features(tr, [va, te])
                Xtr, (Xva, Xte) = feats
                pv, pt = tfidf_model(w.split('_')[1], Xtr, Ytr, [Xva, Xte], s)
            r = _save(d, f'classic_{w}', s, Yte, pt, {'fit_sec': time.time() - t0}, Yva, pv)
            out[w] = r
            log(f"[classic_{w}] seed{s} test t3_macro_f1={r['test']['t3_macro_f1']:.4f}")
    return out