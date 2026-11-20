"""QLoRA fine-tuning (loss on the JSON answer only). Run as a subprocess:

    python -m vitourabsa.models.llm_qlora --model Qwen/Qwen2.5-7B-Instruct --split_dir work/splits/S1_main --out work/ckpt/qwen25_7b_qlora
Defaults for one 24GB GPU: 4-bit NF4 base, LoRA r=16/alpha=32 on all linear projections, 20k training reviews,
1 epoch, lr 2e-4, effective batch 16, gradient checkpointing.
"""
import argparse
import os

import numpy as np
import torch

from ..common import read_jsonl, save_json, set_seed
from .formats import messages, render, to_json


class SFT(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            m = messages(r['text'])
            prompt = render(tok, m, True)
            full = render(tok, m + [{'role': 'assistant', 'content': to_json(r['labels'])}], False)
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            f_ids = tok(full, add_special_tokens=False).input_ids[:max_len]
            lab = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
            if any(l != -100 for l in lab):
                self.items.append((f_ids, lab))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_fn(pad):
    def f(b):
        L = max(len(x) for x, _ in b)
        ids = torch.full((len(b), L), pad); lab = torch.full((len(b), L), -100)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, (x, y) in enumerate(b):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
        return {'input_ids': ids, 'labels': lab, 'attention_mask': att}
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_train', type=int, default=20000)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--r', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--no_4bit', action='store_true', help='only for CPU smoke tests')
    a = ap.parse_args(argv)
    set_seed(a.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tr = read_jsonl(os.path.join(a.split_dir, 'train.jsonl'))
    if a.max_train and a.max_train < len(tr):
        tr = [tr[i] for i in sorted(np.random.RandomState(a.seed).choice(len(tr), a.max_train, replace=False))]
    ds = SFT(tr, tok, a.max_len)
    print('SFT examples:', len(ds))
    if a.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(a.model)
    else:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map='auto',
                                                     torch_dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    names = {n.split('.')[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    targets = [t for t in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] if t in names]
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, bias='none',
                                             task_type='CAUSAL_LM', target_modules=targets or 'all-linear'))
    model.print_trainable_parameters()
    # plain training loop (independent of transformers.Trainer API changes across versions)
    import math
    import time
    dev = next(p for p in model.parameters() if p.requires_grad).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate_fn(tok.pad_token_id),
                                     generator=torch.Generator().manual_seed(a.seed))
    total = max(1, math.ceil(len(dl) * a.epochs / a.accum))
    warm = max(1, int(0.03 * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    model.train()
    step, it, t0, run_loss, n_loss = 0, 0, time.time(), 0.0, 0
    use_amp = torch.cuda.is_available()
    while step < total:
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss / a.accum
            loss.backward()
            run_loss += loss.item() * a.accum
            n_loss += 1
            it += 1
            if it % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sch.step(); opt.zero_grad(); step += 1
                if step % 25 == 0 or step == total:
                    print(f'step {step}/{total} loss={run_loss / max(1, n_loss):.4f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
                    run_loss, n_loss = 0.0, 0
                if step >= total:
                    break
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    save_json({**vars(a), 'n_sft': len(ds), 'targets': targets}, os.path.join(a.out, 'qlora_config.json'))


if __name__ == '__main__':
    main()