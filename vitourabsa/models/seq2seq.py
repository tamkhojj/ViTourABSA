"""Seq2seq baselines (ViT5, BARTpho, mT5): review -> 'phòng: 2; giá cả: -1' (only polar aspects) or 'không có'."""
import json
import math
import os
import time

import numpy as np
import torch

from ..common import labels_of, read_jsonl, save_json, set_seed
from .formats import from_seq, to_seq
from ..metrics import PRIMARY, evaluate


def pick_format(tok, rows, n=500):
    """Choose the first target format that survives tokenize->decode (avoids <unk> in targets)."""
    for fmt in ('num', 'word'):
        bad = 0
        for r in rows[:n]:
            s = to_seq(r['labels'], fmt)
            lab, ok = from_seq(tok.decode(tok(s).input_ids, skip_special_tokens=True))
            bad += (not ok) or lab != r['labels']
        if bad <= 2:
            return fmt, bad
    raise RuntimeError('no target format round-trips through this tokenizer')


def _enc(tok, texts, max_len, device=None):
    e = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors='pt')
    e.pop('token_type_ids', None)
    return e.to(device) if device is not None else e


@torch.no_grad()
def generate(model, tok, rows, device, bs, max_len, max_new):
    model.eval()
    P, inv = [], 0
    for i in range(0, len(rows), bs):
        e = _enc(tok, [r['text'] for r in rows[i:i + bs]], max_len, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = model.generate(**e, max_new_tokens=max_new, num_beams=1, do_sample=False)
        for s in tok.batch_decode(out, skip_special_tokens=True):
            lab, ok = from_seq(s)
            P.append(lab); inv += (not ok)
    return np.array(P), inv / max(1, len(rows))


def train_seq2seq(model_name, split_dir, out_dir, seed, lr=3e-4, epochs=8, patience=2, bs=16, max_len=256,
                  max_new=96, val_subset=2000, max_train=0, log=print):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tr = read_jsonl(os.path.join(split_dir, 'train.jsonl'))
    va = read_jsonl(os.path.join(split_dir, 'val.jsonl'))
    te = read_jsonl(os.path.join(split_dir, 'test.jsonl'))
    if max_train:
        tr = tr[:max_train]
    fmt, bad = pick_format(tok, tr)
    log(f'  target format = {fmt} (round-trip failures {bad}/500)')
    save_json({'model_name': model_name, 'seed': seed, 'lr': lr, 'format': fmt, 'split_dir': split_dir},
              os.path.join(out_dir, 'config.json'))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = math.ceil(len(tr) / bs) * epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) *
                                            max(0.0, 1 - s / total))
    vs = va[:val_subset]
    best, best_state, badn, times = -1, None, 0, []
    flog = open(os.path.join(out_dir, 'train_log.jsonl'), 'w')
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        idx = rng.permutation(len(tr))
        for i in range(0, len(idx), bs):
            b = [tr[j] for j in idx[i:i + bs]]
            e = _enc(tok, [r['text'] for r in b], max_len, device)
            y = tok([to_seq(r['labels'], fmt) for r in b], truncation=True, max_length=max_new, padding=True,
                    return_tensors='pt').input_ids.to(device)
            y[y == tok.pad_token_id] = -100
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = model(**e, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tot += loss.item(); n += 1
        times.append(time.time() - t0)
        pv, inv = generate(model, tok, vs, device, 64, max_len, max_new)
        mv = evaluate(labels_of(vs), pv, per_aspect=False)
        flog.write(json.dumps({'epoch': ep + 1, 'loss': tot / n, 'val_invalid': inv,
                               **{f'val_{k}': v for k, v in mv.items()}}) + '\n'); flog.flush()
        log(f'  ep{ep + 1} loss={tot / n:.4f} val {PRIMARY}={mv[PRIMARY]:.4f} invalid={inv:.3f} ({times[-1]:.0f}s)')
        if mv[PRIMARY] > best:
            best, badn = mv[PRIMARY], 0
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
        else:
            badn += 1
            if badn >= patience:
                break
    flog.close()
    model.load_state_dict(best_state)
    res = {'run': os.path.basename(os.path.dirname(out_dir)), 'seed': seed, 'model_name': model_name,
           'params': sum(p.numel() for p in model.parameters()), 'sec_per_epoch': float(np.mean(times)),
           'epochs_run': len(times), 'format': fmt, 'best_val': best}
    for part, rows in [('val', va), ('test', te)]:
        t0 = time.time()
        p, inv = generate(model, tok, rows, device, 64, max_len, max_new)
        y = labels_of(rows)
        res[part] = evaluate(y, p)
        res[part]['invalid_rate'] = inv
        if part == 'test':
            res['test_reviews_per_sec'] = len(rows) / (time.time() - t0)
        np.savez_compressed(os.path.join(out_dir, f'preds_{part}.npz'), y=y, pred=p)
    save_json(res, os.path.join(out_dir, 'metrics.json'))
    del model, opt, sch, best_state
    import gc
    gc.collect()
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    return res