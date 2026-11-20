"""Read the released ViTourABSA files and turn them into the experiment format.

Released layout (repository root):
    data/{train,val,test}.csv          main split S1 (all 87,314 deduplicated reviews)
    splits/<name>.json                 review ids of every split (S1 to S6)
    splits/S6_with_duplicates_*.csv.gz full rows of S6, because duplicates are not in data/

Experiment format (one folder per split under work/splits/):
    <name>/{train,val,test}.jsonl      {"id", "text", "text_seg", "labels": [10 ints], "province", "facility", ...}
    split_info.json                    sizes of every split
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from ..common import ASPECTS, save_json, write_jsonl

PARTS = ('train', 'val', 'test')


def load_main(release_root):
    """All released reviews of S1 as one DataFrame with a column `part` (train / val / test)."""
    frames = []
    for p in PARTS:
        f = os.path.join(release_root, 'data', f'{p}.csv')
        if not os.path.exists(f):
            raise FileNotFoundError(f'{f} not found. Run the scripts from the root of the ViTourABSA repository.')
        frames.append(pd.read_csv(f).assign(part=p))
    return pd.concat(frames, ignore_index=True)


def to_analysis_frame(df):
    """Column names used by vitourabsa.analysis.eda: text, text_seg, the ten aspects, province, facility."""
    out = df.rename(columns={'id': 'review_id', 'review': 'text', 'review_segmented': 'text_seg',
                             'facility_type': 'facility'})
    out['text'] = out['text'].fillna('').astype(str)
    out['text_seg'] = out['text_seg'].fillna('').astype(str)
    return out


def _row(r, extra=None):
    d = {'id': int(r['id']), 'text': str(r['review']), 'text_seg': str(r['review_segmented']),
         'labels': [int(r[a]) for a in ASPECTS], 'province': r['province'], 'facility': r['facility_type'],
         'dup_count': 1}
    if extra:
        d.update(extra)
    return d


def _write(out_root, name, parts):
    for p, rows in parts.items():
        write_jsonl(os.path.join(out_root, name, f'{p}.jsonl'), rows)
    return {p: len(rows) for p, rows in parts.items()}


def build_splits(release_root, out_root, log=print):
    """Create work/splits/<name>/{train,val,test}.jsonl for S1 to S6 from the released files.

    The ids in splits/*.json are the ids used in the paper, so the experiments are reproduced exactly.
    """
    main = load_main(release_root)
    for c in ['review', 'review_segmented']:
        main[c] = main[c].fillna('')
    by_id = {int(r['id']): r for r in main.to_dict('records')}
    info = {}

    # S1: exactly the released CSV files
    info['S1_main'] = _write(out_root, 'S1_main',
                             {p: [_row(r) for r in main[main.part == p].to_dict('records')] for p in PARTS})

    # S2 to S5: ids that refer to the released (deduplicated) reviews
    for f in sorted(glob.glob(os.path.join(release_root, 'splits', '*.json'))):
        name = os.path.basename(f)[:-5]
        if name in ('S1_main',) or name.startswith('S6'):
            continue
        spec = json.load(open(f, encoding='utf-8'))
        missing = [i for p in PARTS for i in spec[p] if i not in by_id]
        if missing:
            raise ValueError(f'{name}: {len(missing)} ids not found in data/*.csv')
        info[name] = _write(out_root, name, {p: [_row(by_id[i]) for i in spec[p]] for p in PARTS})

    # S6: duplicates are shipped as full rows in splits/S6_with_duplicates_{train,val,test}.csv.gz
    s6 = os.path.join(release_root, 'splits', 'S6_with_duplicates.json')
    if os.path.exists(s6):
        spec = json.load(open(s6, encoding='utf-8'))
        seen = set(spec.get('test_seen_in_train', []))
        parts = {}
        for p in PARTS:
            df = pd.read_csv(os.path.join(release_root, 'splits', f'S6_with_duplicates_{p}.csv.gz'))
            for c in ['review', 'review_segmented']:
                df[c] = df[c].fillna('')
            if p == 'test':
                parts[p] = [_row(r, {'seen_in_train': int(r['id']) in seen}) for r in df.to_dict('records')]
            else:
                parts[p] =[_row(r, {'seen_in_train': p == 'train'}) for r in df.to_dict('records')]
        if 'val' in parts:   # val flag: text also present in S6 train
            tr_texts = {r['text'] for r in parts['train']}
            for r in parts['val']:
                r['seen_in_train'] = r['text'] in tr_texts
        info['S6_with_duplicates'] = _write(out_root, 'S6_with_duplicates', parts)
        info['S6_test_seen_rate'] = float(np.mean([r['seen_in_train'] for r in parts['test']]))

    save_json(info, os.path.join(out_root, 'split_info.json'))
    for k, v in info.items():
        log(f'  {k:24s} {v}')
    return info
