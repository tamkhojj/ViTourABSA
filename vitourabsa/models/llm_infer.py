"""LLM inference (zero-shot / few-shot / QLoRA adapter) on a full test set. Run as a subprocess:

    python -m vitourabsa.models.llm_infer --model Qwen/Qwen2.5-7B-Instruct --split_dir work/splits/S1_main \
           --out_dir work/results/S1_main/llm_qwen25_7b_zero/seed1 --mode zero
Backend 'auto' uses vLLM when importable (much faster), otherwise transformers.generate (bf16).
Invalid outputs -> all zeros (counted as missed opinions); invalid rate and valid-only metrics are reported.
"""
import argparse
import json
import os
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # less fragmentation

import numpy as np

from ..common import labels_of, read_jsonl, save_json
from .formats import from_json, messages, render, select_fewshot
from ..metrics import evaluate


def has_vllm():
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


def run_vllm(model, prompts, max_new, adapter, max_model_len):
    from vllm import LLM, SamplingParams
    kw = dict(model=model, dtype='bfloat16', max_model_len=max_model_len, gpu_memory_utilization=0.90, seed=0)
    if adapter:
        kw.update(enable_lora=True, max_lora_rank=64)
    llm = LLM(**kw)
    lora = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest('adapter', 1, adapter)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new), lora_request=lora)
    return [o.outputs[0].text for o in outs]


def gpu_report():
    """Print free GPU memory and other processes using the GPU (helps diagnose OOM)."""
    import subprocess
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f'GPU free {free / 1e9:.1f} / {total / 1e9:.1f} GB', flush=True)
        q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        if q:
            print('GPU processes (pid, memory):', q.replace('\n', ' | '), flush=True)
    except Exception:
        pass


def make_batches(lengths, order, max_bs, token_budget):
    """Group length-sorted prompts so that (#prompts x longest prompt) stays under a token budget."""
    batches, cur, cur_max = [], [], 0
    for j in order:
        m = max(cur_max, lengths[j])
        if cur and (len(cur) + 1 > max_bs or m * (len(cur) + 1) > token_budget):
            batches.append(cur)
            cur, m = [], lengths[j]
        cur.append(j)
        cur_max = m
    if cur:
        batches.append(cur)
    return batches


def run_hf(model_name, tok, prompts, max_new, bs, adapter, token_budget=8192, partial_path=None):
    import torch
    from transformers import AutoModelForCausalLM
    gpu_report()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available()
                                                 else torch.float32, device_map='auto' if torch.cuda.is_available() else None)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    gpu_report()
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = [None] * len(prompts)
    # resume from partial outputs (e.g. after a crash / power loss)
    if partial_path and os.path.exists(partial_path):
        for l in open(partial_path, encoding='utf-8'):
            try:
                r = json.loads(l)
                out[r['i']] = r['o']
            except Exception:
                pass
        print(f'resumed {sum(o is not None for o in out)}/{len(out)} outputs', flush=True)
    todo = [j for j in range(len(prompts)) if out[j] is None]
    lengths = [len(x) for x in tok([prompts[j] for j in range(len(prompts))], add_special_tokens=False)['input_ids']]
    order = sorted(todo, key=lambda j: lengths[j])                    # short first: early progress, warm-up
    batches = make_batches(lengths, order, bs, token_budget)
    pf = open(partial_path, 'a', encoding='utf-8') if partial_path else None

    def gen(idx):
        enc = tok([prompts[j] for j in idx], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        enc.pop('token_type_ids', None)
        kw = dict(max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                  pad_token_id=tok.pad_token_id, use_cache=True)       # some checkpoints ship use_cache=false
        with torch.no_grad():
            try:   # stop as soon as the JSON object is closed (avoids generating up to max_new tokens)
                g = model.generate(**enc, **kw, stop_strings=['}'], tokenizer=tok)
            except (TypeError, ValueError):
                g = model.generate(**enc, **kw)
        return tok.batch_decode(g[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)

    def run(idx):
        """Generate for idx; on CUDA OOM split the batch in half and retry (down to a single prompt)."""
        try:
            texts = gen(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                print(f'!! OOM on a single prompt ({lengths[idx[0]]} tokens) -> empty output', flush=True)
                texts = ['']
            else:
                h = len(idx) // 2
                run(idx[:h]); run(idx[h:])
                return
        for j, t in zip(idx, texts):
            out[j] = t
            if pf:
                pf.write(json.dumps({'i': int(j), 'o': t}, ensure_ascii=False) + '\n')
        if pf:
            pf.flush()

    t0, n_done = time.time(), len(prompts) - len(todo)
    for b, idx in enumerate(batches):
        run(idx)
        n_done += len(idx)
        if b % 10 == 0 or b == len(batches) - 1:
            el = time.time() - t0
            print(f'{n_done}/{len(prompts)} ({el / 60:.1f} min elapsed)', flush=True)
    if pf:
        pf.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mode', choices=['zero', 'few', 'qlora'], default='zero')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--fewshot_from', default=None, help='split dir to draw few-shot examples from (default: split_dir)')
    ap.add_argument('--backend', choices=['auto', 'vllm', 'hf'], default='auto')
    ap.add_argument('--max_new', type=int, default=120)
    ap.add_argument('--max_model_len', type=int, default=6144)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--token_budget', type=int, default=8192, help='max (#prompts x longest prompt) per batch')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    te = read_jsonl(os.path.join(a.split_dir, 'test.jsonl'))
    if a.limit:
        te = te[:a.limit]
    shots = []
    if a.mode == 'few':
        shots = select_fewshot(read_jsonl(os.path.join(a.fewshot_from or a.split_dir, 'train.jsonl')), a.shots)
    prompts = [render(tok, messages(r['text'], shots)) for r in te]
    backend = a.backend if a.backend != 'auto' else ('vllm' if has_vllm() else 'hf')
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    adapter = a.adapter if a.mode == 'qlora' else None
    texts = run_vllm(a.model, prompts, a.max_new, adapter, a.max_model_len) if backend == 'vllm' \
        else run_hf(a.model, tok, prompts, a.max_new, a.bs, adapter, a.token_budget,
                    partial_path=os.path.join(a.out_dir, 'partial_outputs.jsonl'))
    sec = time.time() - t0
    parsed = [from_json(t) for t in texts]
    P = np.array([p for p, _ in parsed])
    V = np.array([v for _, v in parsed])
    Y = labels_of(te)
    os.makedirs(a.out_dir, exist_ok=True)
    res = {'run': os.path.basename(os.path.dirname(a.out_dir)), 'seed': 1, 'model_name': a.model, 'mode': a.mode,
           'backend': backend, 'n': len(te), 'invalid_rate': float(1 - V.mean()), 'test_reviews_per_sec': len(te) / sec,
           'fewshot_ids': [s['id'] for s in shots], 'test': evaluate(Y, P)}
    res['test']['invalid_rate'] = res['invalid_rate']
    if V.any():
        res['test_valid_only'] = evaluate(Y[V], P[V], per_aspect=False)
    save_json(res, os.path.join(a.out_dir, 'metrics.json'))
    np.savez_compressed(os.path.join(a.out_dir, 'preds_test.npz'), y=Y, pred=P, valid=V)
    with open(os.path.join(a.out_dir, 'raw_outputs.jsonl'), 'w', encoding='utf-8') as f:
        for r, t, v in zip(te, texts, V):
            f.write(json.dumps({'id': r['id'], 'output': t, 'valid': bool(v)}, ensure_ascii=False) + '\n')
    print(f"{res['run']}: backend={backend} invalid={res['invalid_rate']:.3f} "
          f"t3_macro_f1={res['test']['t3_macro_f1']:.4f} joint={res['test']['joint_f1_macro']:.4f} ({sec / 60:.1f} min)")


if __name__ == '__main__':
    main()