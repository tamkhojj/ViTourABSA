"""Run the whole benchmark of the paper, step by step, in the order used for the paper.

Steps: classic, encoders, llm, variants, seq2seq, qlora, shift, learning_curve, leakage
Finished runs are skipped, failed runs are logged in work/reports/failures.log and the next run starts, so the
command can simply be started again after an interruption. The final tables are made by 08_make_report.py.

    python scripts/07_run_benchmark.py                          # everything (several days on one RTX 4090)
    python scripts/07_run_benchmark.py --steps classic encoders  # only some steps
    nohup python scripts/07_run_benchmark.py > work/benchmark.log 2>&1 &     # in the background
"""
import argparse
import os

import _root  # noqa: F401

from vitourabsa import paths
from vitourabsa.pipeline import STEPS, load_config, log

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', nargs='+', default=list(STEPS), choices=list(STEPS))
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    paths.ensure()
    if not os.path.exists(os.path.join(paths.SPLITS, 'split_info.json')):
        raise SystemExit('work/splits not found: run  python scripts/00_prepare_splits.py  first')
    cfg = load_config(a.config)
    for step in a.steps:
        log(f'===== step: {step}')
        STEPS[step](cfg)
    if os.path.exists(paths.FAIL_LOG):
        log(f'some runs failed, see {paths.FAIL_LOG}')
    log('done. Next: python scripts/08_make_report.py')
