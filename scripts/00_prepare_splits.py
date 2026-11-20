"""Step 0. Build the experiment splits in work/splits/ (S1 to S6).

Default: read the released dataset (data/, splits/) at the repository root. The splits are identical to the paper.

    python scripts/00_prepare_splits.py

Authors only: rebuild everything from the raw annotated file (cleaning, exact and near deduplication, word
segmentation, stratified splits). This also saves work/prepared.pkl, used by the cleaning and duplicate
analyses in scripts/01_eda.py.

    python scripts/00_prepare_splits.py --from-raw path/to/tourism_dataset.csv
"""
import argparse
import json
import os

import _root  # noqa: F401
import pandas as pd

from vitourabsa import paths
from vitourabsa.pipeline import log


def from_release(force):
    from vitourabsa.data.release import build_splits
    if os.path.exists(os.path.join(paths.SPLITS, 'split_info.json')) and not force:
        log(f'{paths.SPLITS} already exists (use --force to rebuild)')
        return
    log('building splits from the released files ...')
    build_splits(paths.RELEASE, paths.SPLITS, log=log)


def from_raw(csv_path, force):
    from vitourabsa.data import preprocess
    if os.path.exists(paths.PREPARED) and not force:
        P = pd.read_pickle(paths.PREPARED)
        log('loaded cached ' + paths.PREPARED)
    else:
        raw = pd.read_csv(csv_path, low_memory=False)
        clean_df, removed_df, clean_sum = preprocess.clean(raw)
        dedup_df, dedup_sum = preprocess.dedup_majority(clean_df)
        final_df, near_sum = preprocess.near_dedup(dedup_df)
        log('word segmentation with pyvi ...')
        texts = list(set(clean_df.text))
        seg = dict(zip(texts, preprocess.segment_all(texts)))
        final_df['text_seg'] = final_df.text.map(seg)
        clean_df['text_seg'] = clean_df.text.map(seg)
        clean_df['dup_count'] = 1
        os.makedirs(paths.WORK, exist_ok=True)
        removed_df.to_csv(os.path.join(paths.WORK, 'removed_rows.csv'), index=False, encoding='utf-8-sig')
        P = pd.Series({'final': final_df, 'with_dups': clean_df, 'clean_sum': clean_sum, 'dedup_sum': dedup_sum,
                       'near_sum': near_sum})
        pd.to_pickle(P, paths.PREPARED)
    if force or not os.path.exists(os.path.join(paths.SPLITS, 'split_info.json')):
        preprocess.build_all_splits(P['final'], P['with_dups'], paths.SPLITS)
    log(f"final dataset: {len(P['final']):,} reviews")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from-raw', metavar='CSV', help='rebuild from the raw annotated file instead of the release')
    ap.add_argument('--force', action='store_true', help='rebuild even if work/splits already exists')
    a = ap.parse_args()
    paths.ensure()
    from_raw(a.from_raw, a.force) if a.from_raw else from_release(a.force)
    print(json.dumps(json.load(open(os.path.join(paths.SPLITS, 'split_info.json'))), indent=1))
