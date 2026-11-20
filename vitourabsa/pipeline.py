"""Experiment runner shared by the scripts in scripts/.

Every run writes work/results/<split>/<run>/seed<k>/metrics.json. A run whose metrics.json exists is skipped,
so any script can be stopped and started again. Failed runs are written to work/reports/failures.log and the
pipeline continues with the next run.
"""
import gc
import json
import os
import subprocess
import sys
import time
import traceback

import yaml

from . import paths
from .common import BIG_PROVINCES
from .models import classic, encoders, seq2seq

DEFAULT_CONFIG = os.path.join(paths.ROOT, 'configs', 'benchmark.yaml')


# --------------------------------------------------------------------------- configuration
def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    seeds = cfg['seeds']
    cfg['encoders'] = {k: (v[0], v[1] or {}, [float(x) for x in v[2]], seeds[v[3]]) for k, v in cfg['encoders'].items()}
    cfg['seq2seq'] = {k: (v[0], float(v[1])) for k, v in cfg.get('seq2seq', {}).items()}
    return cfg


# --------------------------------------------------------------------------- logging and robustness
def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def _fail(tag, text):
    os.makedirs(os.path.dirname(paths.FAIL_LOG), exist_ok=True)
    with open(paths.FAIL_LOG, 'a', encoding='utf-8') as f:
        f.write(f'==== {time.ctime()} {tag}\n{text}\n')


def free_gpu():
    """Release the GPU memory cached by this process; returns the memory still reserved (GB)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return torch.cuda.memory_reserved() / 1e9
    except Exception:
        pass
    return 0.0


def safe(tag, fn, *a, **k):
    """Run fn; on error, log it and continue (returns None)."""
    try:
        return fn(*a, **k)
    except Exception as e:
        log(f'!! FAILED {tag}: {e!r}')
        _fail(tag, traceback.format_exc())
        free_gpu()
        return None


def sh(args, tag):
    """Run `python -m <args>` in a subprocess from the repository root; True on success."""
    held = free_gpu()
    log(f'$ python -m {" ".join(args)}   [GPU memory held by this process: {held:.2f} GB]')
    p = subprocess.Popen([sys.executable, '-m'] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         cwd=paths.ROOT)
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if any(k in line for k in ('t3_macro_f1', 'SFT examples', '/', 'Error', 'error', 'trainable')):
            print('   ', line.rstrip()[:200], flush=True)
    ok = p.wait() == 0
    if not ok:
        _fail(tag, ''.join(tail))
        log(f'!! FAILED {tag} (see {paths.FAIL_LOG})')
    return ok


def done(split, run, seed=1):
    return os.path.exists(os.path.join(paths.res_dir(split, run, seed), 'metrics.json'))


# --------------------------------------------------------------------------- runners
def run_classic(split, cfg, which=('tfidf_lr',), seeds=(1,)):
    return safe(f'{split}/classic', classic.run_classic, paths.split_dir(split), os.path.join(paths.RESULTS, split),
                seeds=list(seeds), which=tuple(which), log=log, max_train=cfg.get('max_train', 0))


def chosen_lr(cfg, name):
    """Learning rate selected on the S1 validation set (falls back to the first value of the grid)."""
    f = os.path.join(paths.RESULTS, 'S1_main', name, 'chosen_lr.json')
    return [json.load(open(f))['chosen_lr']] if os.path.exists(f) else cfg['encoders'][name][2][:1]


def run_encoder(cfg, name, split='S1_main', seeds=None, run_name=None, lr_grid=None, **over):
    hf, extra, grid, default_seeds = cfg['encoders'][name]
    model_cfg = {'model_name': hf, **extra, **over}
    return safe(f'{split}/{run_name or name}', encoders.run_family, run_name or name, model_cfg,
                list(seeds or default_seeds), paths.split_dir(split), os.path.join(paths.RESULTS, split),
                lr_grid or grid, cfg.get('max_train', 0), log)


def run_variant(cfg, variant, split='S1_main', seeds=None):
    spec = dict(cfg['variants'][variant])
    enc = spec.pop('encoder')
    return run_encoder(cfg, enc, split=split, seeds=seeds or cfg['seeds']['aux'], run_name=variant,
                       lr_grid=chosen_lr(cfg, enc), **spec)


def run_seq2seq(cfg, name, split='S1_main', seeds=None):
    hf, lr = cfg['seq2seq'][name]
    for s in seeds or cfg.get('seq2seq_seeds', [1]):
        if done(split, name, s):
            log(f'[{name}] seed{s} done -> skip')
            continue
        log(f'[{name}] seed{s}')
        safe(f'{split}/{name}/seed{s}', seq2seq.train_seq2seq, hf, paths.split_dir(split), paths.res_dir(split, name, s),
             s, lr=lr, epochs=cfg.get('seq2seq_epochs', 8), max_train=cfg.get('max_train', 0), log=log)


def run_llm(cfg, short, mode, split='S1_main', adapter=None, limit=None):
    """mode: zero (zero-shot), few (10-shot, demonstrations from S1 train) or qlora (needs adapter)."""
    run = f'llm_{short}_{mode}'
    if done(split, run):
        log(f'[{run}] done -> skip')
        return True
    if mode == 'qlora' and adapter is None:
        adapter = os.path.join(paths.CKPT, f'{short}_qlora')
    args = ['vitourabsa.models.llm_infer', '--model', cfg['llms'][short], '--split_dir', paths.split_dir(split),
            '--out_dir', paths.res_dir(split, run), '--mode', mode]
    if mode == 'qlora':
        args += ['--adapter', adapter]
    if mode == 'few':
        args += ['--fewshot_from', paths.split_dir('S1_main')]
    limit = cfg.get('llm_limit', 0) if limit is None else limit
    if limit:
        args += ['--limit', str(limit)]
    return sh(args + list(cfg.get('llm_extra_args', [])), f'{split}/{run}')


def train_qlora(cfg, short):
    ck = os.path.join(paths.CKPT, f'{short}_qlora')
    if os.path.exists(os.path.join(ck, 'adapter_config.json')):
        log(f'[qlora/{short}] adapter exists -> skip')
        return True
    return sh(['vitourabsa.models.llm_qlora', '--model', cfg['llms'][short], '--split_dir', paths.split_dir('S1_main'),
               '--out', ck] + list(cfg.get('qlora_extra_args', [])), f'qlora/{short}')


# --------------------------------------------------------------------------- the benchmark, step by step
def step_classic(cfg):
    run_classic('S1_main', cfg, which=('majority', 'prior', 'tfidf_lr', 'tfidf_svm'), seeds=cfg['seeds']['base'])


def step_encoders(cfg, names=None):
    for name in names or cfg['encoders']:
        run_encoder(cfg, name)


def step_llm_prompting(cfg, names=None):
    for short in names or cfg['llms']:
        for mode in cfg.get('llm_modes', ['zero', 'few']):
            run_llm(cfg, short, mode)


def step_variants(cfg, names=None):
    for v in names or cfg.get('variants', {}):
        run_variant(cfg, v)


def step_seq2seq(cfg, names=None):
    for name in names or cfg.get('seq2seq', {}):
        run_seq2seq(cfg, name)


def step_qlora(cfg, names=None):
    for short in names or cfg.get('qlora_run', []):
        if train_qlora(cfg, short):
            run_llm(cfg, short, 'qlora')


def step_shift(cfg):
    """S2 cross-region and S4 cross-facility (TF-IDF + encoders), zero-shot LLM on S2, S3 leave-one-province-out."""
    for sp in ['S2_out_of_region', 'S4_hotel_to_resort', 'S4_resort_to_hotel']:
        run_classic(sp, cfg)
        for e in cfg.get('shift_encoders', []):
            run_encoder(cfg, e, split=sp, seeds=cfg['seeds']['aux'], lr_grid=chosen_lr(cfg, e))
    for short in cfg.get('llm_on_s2', []):
        run_llm(cfg, short, 'zero', split='S2_out_of_region')
    for p in BIG_PROVINCES:
        sp = f'S3_lopo_{p}'
        run_classic(sp, cfg)
        for e in cfg.get('lopo_encoders', []):
            run_encoder(cfg, e, split=sp, seeds=cfg.get('lopo_seeds', [1]), lr_grid=chosen_lr(cfg, e))


def step_learning_curve(cfg):
    for f in cfg.get('lc_fractions', []):
        sp = f'S5_frac{f}'
        run_classic(sp, cfg)
        for e in cfg.get('lc_encoders', []):
            run_encoder(cfg, e, split=sp, seeds=cfg['seeds']['aux'], lr_grid=chosen_lr(cfg, e))


def step_leakage(cfg):
    sp = 'S6_with_duplicates'
    run_classic(sp, cfg)
    for e in cfg.get('leakage_encoders', []):
        run_encoder(cfg, e, split=sp, seeds=cfg['seeds']['aux'], lr_grid=chosen_lr(cfg, e))


STEPS = {   # in the order used for the paper: core results first
    'classic': step_classic,
    'encoders': step_encoders,
    'llm': step_llm_prompting,
    'variants': step_variants,
    'seq2seq': step_seq2seq,
    'qlora': step_qlora,
    'shift': step_shift,
    'learning_curve': step_learning_curve,
    'leakage': step_leakage,
}


# --------------------------------------------------------------------------- display names used in the tables
def display_names(cfg=None):
    meta = {'classic_majority': ('G0 Floor', 'Majority'), 'classic_prior': ('G0 Floor', 'Prior-random'),
            'classic_tfidf_lr': ('G0 Floor', 'TF-IDF + LR'), 'classic_tfidf_svm': ('G0 Floor', 'TF-IDF + Linear SVM')}
    for k, d in [('phobert_base', 'PhoBERT-base-v2'), ('visobert', 'ViSoBERT'), ('xlmr_base', 'XLM-R-base'),
                 ('mdeberta_base', 'mDeBERTa-v3-base'), ('phobert_large', 'PhoBERT-large'), ('xlmr_large', 'XLM-R-large')]:
        meta[k] = ('G1 Encoders', d)
    for k, d in [('phobert_base_pair', 'PhoBERT-base · review–aspect pair'),
                 ('xlmr_large_pair', 'XLM-R-large · review–aspect pair'),
                 ('phobert_base_coral', 'PhoBERT-base · CORAL ordinal'),
                 ('phobert_base_weighted', 'PhoBERT-base · class-weighted CE'),
                 ('phobert_base_logit_adjust', 'PhoBERT-base · logit adjustment')]:
        meta[k] = ('G1 Variants', d)
    for k, d in [('vit5_base', 'ViT5-base'), ('bartpho_syllable', 'BARTpho-syllable'), ('mt5_base', 'mT5-base')]:
        meta[k] = ('G2 Seq2seq', d)
    for k, d in LLM_DISPLAY.items():
        for mode, lab in [('zero', 'zero-shot'), ('few', '10-shot'), ('qlora', 'QLoRA')]:
            meta[f'llm_{k}_{mode}'] = ('G3 LLMs', f'{d} ({lab})')
    return meta


LLM_DISPLAY = {'qwen25_7b': 'Qwen2.5-7B-Instruct', 'qwen25_3b': 'Qwen2.5-3B-Instruct',
               'llama31_8b': 'Llama-3.1-8B-Instruct', 'aya_expanse_8b': 'Aya-Expanse-8B', 'vistral_7b': 'Vistral-7B-Chat'}
