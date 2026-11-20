"""Fine-tune the G2 sequence-to-sequence model (ViT5-base) on S1.

    python scripts/04_run_seq2seq.py
    python scripts/04_run_seq2seq.py --model vit5_base --seeds 1 2 3
"""
import argparse

import _root  # noqa: F401

from vitourabsa import paths
from vitourabsa.pipeline import load_config, run_seq2seq

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='vit5_base')
    ap.add_argument('--split', default='S1_main')
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    paths.ensure()
    run_seq2seq(load_config(a.config), a.model, split=a.split, seeds=a.seeds)
