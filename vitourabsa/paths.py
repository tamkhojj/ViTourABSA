"""Folder layout. Everything the code generates goes under work/ (git-ignored); the released data are read-only.

    <repo>/data, <repo>/splits, <repo>/stats   released dataset (input)
    work/splits/<split>/{train,val,test}.jsonl  experiment-ready splits (built by scripts/00_prepare_splits.py)
    work/results/<split>/<run>/seed<k>/         metrics.json, predictions, training logs
    work/ckpt/                                  QLoRA adapters
    work/reports/{eda,results,extra}/           tables (CSV + LaTeX) and figures (PDF + PNG)
    work/human_annotation/                      sheets for a second annotator
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = ROOT                                   # the released dataset sits at the repository root
WORK = os.environ.get('VITOURABSA_WORK', os.path.join(ROOT, 'work'))
SPLITS = os.path.join(WORK, 'splits')
RESULTS = os.path.join(WORK, 'results')
CKPT = os.path.join(WORK, 'ckpt')
REPORTS = os.path.join(WORK, 'reports')
EDA_OUT = os.path.join(REPORTS, 'eda')
RESULTS_OUT = os.path.join(REPORTS, 'results')
EXTRA_OUT = os.path.join(REPORTS, 'extra')
HUMAN = os.path.join(WORK, 'human_annotation')
FAIL_LOG = os.path.join(REPORTS, 'failures.log')
PREPARED = os.path.join(WORK, 'prepared.pkl')    # only exists when splits were rebuilt from the raw file


def split_dir(name):
    return os.path.join(SPLITS, name)


def res_dir(split, run, seed=1):
    return os.path.join(RESULTS, split, run, f'seed{seed}')


def ensure():
    for d in (SPLITS, RESULTS, CKPT, EDA_OUT, RESULTS_OUT, EXTRA_OUT, HUMAN):
        os.makedirs(d, exist_ok=True)
