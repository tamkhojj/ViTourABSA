"""Fine-tuning of pre-trained encoders (full fine-tuning, bf16).

Formulations
    multihead : one [CLS] vector -> 10 aspect heads (5 levels each)
    pair      : (aspect name, review) sentence pair -> 5 levels  (Sun et al., 2019 auxiliary sentence)
Heads
    flat  : softmax over 5 levels, loss = ce | weighted | logit_adjust
    coral : ordinal regression (CORAL, Cao et al., 2020), 4 cumulative thresholds per aspect
"""
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..common import ASPECT_VI, ASPECTS, labels_of, read_jsonl, save_json, set_seed
from ..metrics import PRIMARY, evaluate

DEFAULT = dict(mode='multihead', head='flat', loss='ce', lr=2e-5, epochs=10, patience=3, batch_size=32,
               grad_accum=1, max_len=256, warmup_ratio=0.1, weight_decay=0.01, grad_ckpt=False, eval_bs=128,
               dropout=0.1)


def uses_segmentation(model_name):
    return 'phobert' in model_name.lower()


# ====================================================================== data
def _find(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    raise ValueError('template not found')


def special_templates(tok):
    """Infer how the tokenizer wraps one sequence / a pair with special tokens, e.g.
    RoBERTa: <s> A </s></s> B </s>;  BERT/DeBERTa: [CLS] A [SEP] B [SEP].
    Works for any transformers version (no build_inputs_with_special_tokens needed)."""
    a = tok('xin', add_special_tokens=False)['input_ids']
    b = tok('chào', add_special_tokens=False)['input_ids']
    one = tok('xin')['input_ids']
    i = _find(one, a)
    single = (one[:i], one[i + len(a):])
    two = tok('xin', 'chào')['input_ids']
    i = _find(two, a)
    j = _find(two[i + len(a):], b) + i + len(a)
    pair = (two[:i], two[i + len(a):j], two[j + len(b):])
    return single, pair


class EncData(Dataset):
    def __init__(self, rows, tok, cfg):
        seg = uses_segmentation(cfg['model_name'])
        texts = [r['text_seg'] if seg else r['text'] for r in rows]
        Y = labels_of(rows) + 2                                        # class index 0..4
        L = cfg['max_len']
        uniq = list(dict.fromkeys(texts))
        enc = dict(zip(uniq, tok(uniq, add_special_tokens=False)['input_ids']))
        ids = [enc[t] for t in texts]
        self.pad = tok.pad_token_id
        (s_pre, s_suf), (p_pre, p_mid, p_suf) = special_templates(tok)
        if cfg['mode'] == 'multihead':
            room = L - len(s_pre) - len(s_suf)
            self.x = [s_pre + i[:room] + s_suf for i in ids]
            self.y = Y
        else:
            if seg:
                from pyvi import ViTokenizer
                names = [ViTokenizer.tokenize(ASPECT_VI[a]) for a in ASPECTS]
            else:
                names = [ASPECT_VI[a] for a in ASPECTS]
            q = tok(names, add_special_tokens=False)['input_ids']
            self.x, ys = [], []
            for i, row in zip(ids, Y):
                for a in range(len(ASPECTS)):
                    room = L - len(q[a]) - len(p_pre) - len(p_mid) - len(p_suf)
                    self.x.append(p_pre + q[a] + p_mid + i[:room] + p_suf)
                    ys.append([row[a]])
            self.y = np.array(ys)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def collate(self, b):
        L = max(len(x) for x, _ in b)
        ids = torch.full((len(b), L), self.pad, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, (x, _) in enumerate(b):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        return ids, att, torch.tensor(np.stack([y for _, y in b]), dtype=torch.long)


# ====================================================================== model
class EncModel(nn.Module):
    def __init__(self, cfg, n_out, prior=None):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(cfg['model_name'])
        if cfg.get('grad_ckpt'):
            self.enc.gradient_checkpointing_enable()
        d = self.enc.config.hidden_size
        self.cfg, self.n_out = cfg, n_out
        self.drop = nn.Dropout(cfg['dropout'])
        if cfg['head'] == 'coral':
            self.w = nn.Linear(d, n_out, bias=False)
            self.b = nn.Parameter(torch.linspace(1.5, -1.5, 4).repeat(n_out, 1))
        else:
            self.fc = nn.Linear(d, n_out * 5)
        self.register_buffer('prior', torch.as_tensor(prior, dtype=torch.float) if prior is not None
                             else torch.full((n_out, 5), 0.2))

    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state[:, 0]
        h = self.drop(h).float()
        if self.cfg['head'] == 'coral':
            return self.w(h).unsqueeze(-1) + self.b[None]                 # (B, A, 4)
        return self.fc(h).view(-1, self.n_out, 5)

    def loss(self, out, y):
        if self.cfg['head'] == 'coral':
            tgt = (y.unsqueeze(-1) > torch.arange(4, device=y.device)).float()
            return F.binary_cross_entropy_with_logits(out, tgt)
        kind = self.cfg['loss']
        if kind == 'logit_adjust':
            out = out + torch.log(self.prior.clamp(min=1e-8))[None]
        logp = F.log_softmax(out, -1)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)              # (B, A)
        if kind == 'weighted':
            w = (1 / self.prior.clamp(min=1e-8)).sqrt()
            w = w / (w * self.prior).sum(-1, keepdim=True)
            nll = nll * w[torch.arange(self.n_out, device=y.device)[None].expand_as(y), y]
        return nll.mean()

    def predict(self, out):
        if self.cfg['head'] == 'coral':
            pc = torch.sigmoid(out)
            cdf = torch.cat([torch.ones_like(pc[..., :1]), pc, torch.zeros_like(pc[..., :1])], -1)
            prob = (cdf[..., :-1] - cdf[..., 1:]).clamp(min=0)
            prob = prob / prob.sum(-1, keepdim=True).clamp(min=1e-8)
            return (pc > 0.5).sum(-1), prob
        prob = F.softmax(out, -1)
        return prob.argmax(-1), prob


def _amp():
    if torch.cuda.is_available():
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return torch.autocast('cpu', enabled=False)


@torch.no_grad()
def predict(model, dl, device, mode):
    model.eval()
    P, PR, Y = [], [], []
    for ids, att, y in dl:
        with _amp():
            out = model(ids.to(device), att.to(device))
        p, pr = model.predict(out)
        P.append(p.cpu()); PR.append(pr.float().cpu()); Y.append(y)
    P, PR, Y = torch.cat(P).numpy(), torch.cat(PR).numpy(), torch.cat(Y).numpy()
    if mode == 'pair':
        P, Y, PR = P.reshape(-1, len(ASPECTS)), Y.reshape(-1, len(ASPECTS)), PR.reshape(-1, len(ASPECTS), 5)
    return Y - 2, P - 2, PR                                             # back to scores -2..2


def train_encoder(cfg, split_dir, out_dir, seed, max_train=0, log=print):
    """Train one configuration with one seed. Returns the metrics dict (also saved to out_dir)."""
    cfg = {**DEFAULT, **cfg}
    os.makedirs(out_dir, exist_ok=True)
    save_json({**cfg, 'seed': seed, 'split_dir': split_dir}, os.path.join(out_dir, 'config.json'))
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['model_name'])
    tr = read_jsonl(os.path.join(split_dir, 'train.jsonl'))
    va = read_jsonl(os.path.join(split_dir, 'val.jsonl'))
    te = read_jsonl(os.path.join(split_dir, 'test.jsonl'))
    if max_train:
        tr = tr[:max_train]
    dtr, dva, dte = EncData(tr, tok, cfg), EncData(va, tok, cfg), EncData(te, tok, cfg)
    n_out = len(ASPECTS) if cfg['mode'] == 'multihead' else 1
    Ytr = labels_of(tr) + 2
    prior = np.stack([np.bincount(Ytr[:, a], minlength=5) + 1 for a in range(len(ASPECTS))]).astype(float)
    if n_out == 1:
        prior = prior.sum(0, keepdims=True)
    prior = prior / prior.sum(1, keepdims=True)
    model = EncModel(cfg, n_out, prior).to(device)
    g = torch.Generator().manual_seed(seed)
    ltr = DataLoader(dtr, batch_size=cfg['batch_size'], shuffle=True, collate_fn=dtr.collate, generator=g)
    lva = DataLoader(dva, batch_size=cfg['eval_bs'], collate_fn=dva.collate)
    lte = DataLoader(dte, batch_size=cfg['eval_bs'], collate_fn=dte.collate)
    head = [p for n, p in model.named_parameters() if not n.startswith('enc.')]
    enc_decay = [p for n, p in model.enc.named_parameters() if not any(k in n for k in ('bias', 'LayerNorm', 'layer_norm'))]
    enc_nodecay = [p for n, p in model.enc.named_parameters() if any(k in n for k in ('bias', 'LayerNorm', 'layer_norm'))]
    opt = torch.optim.AdamW([{'params': enc_decay, 'weight_decay': cfg['weight_decay']},
                             {'params': enc_nodecay, 'weight_decay': 0.0},
                             {'params': head, 'weight_decay': 0.0, 'lr': cfg['lr'] * 10}], lr=cfg['lr'])
    steps = math.ceil(len(ltr) / cfg['grad_accum']) * cfg['epochs']
    warm = int(cfg['warmup_ratio'] * steps)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm
                                            else max(0.0, (steps - s) / max(1, steps - warm)))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    best, best_state, bad, times = -1.0, None, 0, []
    flog = open(os.path.join(out_dir, 'train_log.jsonl'), 'w')
    for ep in range(cfg['epochs']):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad()
        for i, (ids, att, y) in enumerate(ltr):
            with _amp():
                out = model(ids.to(device), att.to(device))
            loss = model.loss(out.float(), y.to(device))
            (loss / cfg['grad_accum']).backward()
            if (i + 1) % cfg['grad_accum'] == 0 or i + 1 == len(ltr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
            tot += loss.item(); n += 1
        times.append(time.time() - t0)
        yv, pv, _ = predict(model, lva, device, cfg['mode'])
        mv = evaluate(yv, pv, per_aspect=False)
        rec = {'epoch': ep + 1, 'loss': tot / n, 'sec': times[-1], **{f'val_{k}': v for k, v in mv.items()}}
        flog.write(json.dumps(rec) + '\n'); flog.flush()
        log(f"  ep{ep + 1} loss={tot / n:.4f} val {PRIMARY}={mv[PRIMARY]:.4f} joint={mv['joint_f1_macro']:.4f} "
            f"({times[-1]:.0f}s)")
        if mv[PRIMARY] > best:
            best, bad = mv[PRIMARY], 0
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    flog.close()
    model.load_state_dict(best_state)
    res = {'run': os.path.basename(os.path.dirname(out_dir)), 'seed': seed, 'model_name': cfg['model_name'],
           'params': sum(p.numel() for p in model.parameters()), 'epochs_run': len(times),
           'sec_per_epoch': float(np.mean(times)),
           'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None,
           'best_val': best}
    for part, dl in [('val', lva), ('test', lte)]:
        t0 = time.time()
        y, p, pr = predict(model, dl, device, cfg['mode'])
        if part == 'test':
            res['test_reviews_per_sec'] = len(y) / (time.time() - t0)
        res[part] = evaluate(y, p)
        np.savez_compressed(os.path.join(out_dir, f'preds_{part}.npz'), y=y, pred=p, prob=pr.astype(np.float16))
    save_json(res, os.path.join(out_dir, 'metrics.json'))
    # free GPU memory completely (the scheduler keeps a reference to the optimizer -> params -> GPU)
    del model, opt, sch, best_state, ltr, lva, lte, dtr, dva, dte
    import gc
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return res


def done(out_dir):
    return os.path.exists(os.path.join(out_dir, 'metrics.json'))


def run_family(name, cfg, seeds, split_dir, results_root, lr_grid=None, max_train=0, log=print):
    """lr selected on VAL with the first seed (one run per lr); remaining seeds use the chosen lr.
    The chosen-lr run of the first seed is reused as seed 1 of the family."""
    base = os.path.join(results_root, name)
    lr_file = os.path.join(base, 'chosen_lr.json')
    if lr_grid and len(lr_grid) > 1 and not os.path.exists(lr_file):
        scores = {}
        for lr in lr_grid:
            d = os.path.join(results_root, f'{name}__lr{lr:g}', f'seed{seeds[0]}')
            if not done(d):
                log(f'[{name}] lr search lr={lr:g} seed{seeds[0]}')
                train_encoder({**cfg, 'lr': lr}, split_dir, d, seeds[0], max_train, log)
            scores[lr] = json.load(open(os.path.join(d, 'metrics.json')))['best_val']
        best_lr = max(scores, key=scores.get)
        src = os.path.join(results_root, f'{name}__lr{best_lr:g}', f'seed{seeds[0]}')
        dst = os.path.join(base, f'seed{seeds[0]}')
        if not done(dst):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            m = json.load(open(os.path.join(dst, 'metrics.json'))); m['run'] = name
            save_json(m, os.path.join(dst, 'metrics.json'))
        save_json({'chosen_lr': best_lr, 'val_scores': scores}, lr_file)
    lr = json.load(open(lr_file))['chosen_lr'] if os.path.exists(lr_file) else (lr_grid[0] if lr_grid else cfg.get('lr'))
    out = []
    for s in seeds:
        d = os.path.join(base, f'seed{s}')
        if done(d):
            log(f'[{name}] seed{s} done -> skip'); continue
        log(f'[{name}] seed{s} lr={lr:g}')
        out.append(train_encoder({**cfg, 'lr': lr}, split_dir, d, s, max_train, log))
    return out