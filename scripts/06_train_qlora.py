"""Train a QLoRA adapter for an LLM on 20,000 S1 training reviews (one epoch), then evaluate it on the S1 test set.

    python scripts/06_train_qlora.py --model vistral_7b
    python scripts/06_train_qlora.py --model qwen25_7b --no-eval
"""
import argparse

import _root  # noqa: F401

from vitourabsa import paths
from vitourabsa.pipeline import load_config, run_llm, train_qlora

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='short name from configs/benchmark.yaml (llms)')
    ap.add_argument('--no-eval', action='store_true', help='only train the adapter')
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    paths.ensure()
    cfg = load_config(a.config)
    ok = train_qlora(cfg, a.model)
    if ok and not a.no_eval:
        ok = run_llm(cfg, a.model, 'qlora')
    raise SystemExit(0 if ok else 1)
