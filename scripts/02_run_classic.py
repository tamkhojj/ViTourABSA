"""Run the G0 baselines (majority, prior-random, TF-IDF + logistic regression, TF-IDF + linear SVM) on one split.

    python scripts/02_run_classic.py                                   # all four baselines on S1
    python scripts/02_run_classic.py --split S2_out_of_region --which tfidf_lr
"""
import argparse

import _root  # noqa: F401

from vitourabsa import paths
from vitourabsa.pipeline import load_config, run_classic

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='S1_main')
    ap.add_argument('--which', nargs='+', default=['majority', 'prior', 'tfidf_lr', 'tfidf_svm'],
                    choices=['majority', 'prior', 'tfidf_lr', 'tfidf_svm'])
    ap.add_argument('--seeds', nargs='+', type=int, default=None, help='seeds of the prior-random baseline')
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    paths.ensure()
    cfg = load_config(a.config)
    run_classic(a.split, cfg, which=a.which, seeds=a.seeds or cfg['seeds']['base'])
