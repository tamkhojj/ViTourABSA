"""Fine-tune a G1 encoder (or one of its variants) on one split.

On S1 the learning rate is chosen on the validation set with the first seed (grid in configs/benchmark.yaml);
on the other splits and for the variants, the learning rate chosen on S1 is reused.

    python scripts/03_run_encoder.py --model phobert_base                     # 5 seeds on S1
    python scripts/03_run_encoder.py --variant phobert_base_pair              # review-aspect pair variant
    python scripts/03_run_encoder.py --model xlmr_large --split S2_out_of_region --seeds 1 2 3
"""
import argparse

import _root  # noqa: F401

from vitourabsa import paths
from vitourabsa.pipeline import chosen_lr, load_config, run_encoder, run_variant

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--model', help='encoder name from configs/benchmark.yaml (e.g. phobert_base, xlmr_large)')
    g.add_argument('--variant', help='variant name from configs/benchmark.yaml (e.g. phobert_base_pair)')
    ap.add_argument('--split', default='S1_main')
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    paths.ensure()
    cfg = load_config(a.config)
    if a.variant:
        run_variant(cfg, a.variant, split=a.split, seeds=a.seeds)
    elif a.split == 'S1_main':
        run_encoder(cfg, a.model, seeds=a.seeds)
    else:
        run_encoder(cfg, a.model, split=a.split, seeds=a.seeds or cfg['seeds']['aux'], lr_grid=chosen_lr(cfg, a.model))
