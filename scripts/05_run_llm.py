"""Evaluate a G3 LLM with zero-shot, 10-shot or QLoRA (adapter from 06_train_qlora.py) on a split.

Gated models (Llama, Vistral) need a Hugging Face token in the environment variable HF_TOKEN and an accepted
licence on the model page. Inference runs in a subprocess so that the GPU is released afterwards.

    python scripts/05_run_llm.py --model qwen25_7b --mode zero
    python scripts/05_run_llm.py --model llama31_8b --mode few
    python scripts/05_run_llm.py --model vistral_7b --mode qlora
    python scripts/05_run_llm.py --model qwen25_7b --mode zero --split S2_out_of_region
"""
import argparse

import _root  # noqa: F401

from vitourabsa import paths
from vitourabsa.pipeline import load_config, run_llm

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='short name from configs/benchmark.yaml (llms)')
    ap.add_argument('--mode', choices=['zero', 'few', 'qlora'], default='zero')
    ap.add_argument('--split', default='S1_main')
    ap.add_argument('--adapter', default=None, help='QLoRA adapter folder (default: work/ckpt/<model>_qlora)')
    ap.add_argument('--limit', type=int, default=None, help='evaluate only the first N test reviews (quick check)')
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    paths.ensure()
    ok = run_llm(load_config(a.config), a.model, a.mode, split=a.split, adapter=a.adapter, limit=a.limit)
    raise SystemExit(0 if ok else 1)
