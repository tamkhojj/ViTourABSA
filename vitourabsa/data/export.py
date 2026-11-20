"""Export the experiment splits (work/splits) as the public ViTourABSA release.

Produces, under an output folder:
    README.md, load_example.py, SHA256SUMS.txt
    data/{train,val,test}.csv and .jsonl          main split S1
    splits/<name>.json                            review ids of S1 to S6
    splits/S6_with_duplicates_{train,val,test}.csv.gz
    stats/                                        label distributions, province x facility, overview
and a zip archive of the folder. The exported ids are exactly those used in the experiments.
"""
import glob
import hashlib
import json
import os
import shutil
import zipfile

import pandas as pd

from ..common import ASPECTS, read_jsonl
from ..models.formats import ASPECT_DEFS

DATASET = 'ViTourABSA'
TITLE = 'A large-scale multi-regional dataset and benchmark for Vietnamese aspect-based sentiment analysis in tourism'
LICENSE = 'CC BY-NC-SA 4.0'
PARTS = ('train', 'val', 'test')
DESC = {
    'S2_out_of_region': 'Train/val: 5 high-volume provinces; test: 4 lower-volume provinces (cross-region).',
    'S3_lopo_': 'Leave-one-province-out: test = all reviews of the held-out province.',
    'S4_hotel_to_resort': 'Train/val: hotels; test: resorts.',
    'S4_resort_to_hotel': 'Train/val: resorts; test: hotels.',
    'S5_frac': 'Learning curve: nested random subsets of S1 train (val/test = S1).',
    'S6_with_duplicates': 'Random split WITHOUT deduplication (duplicate-leakage analysis).',
}


def to_df(rows):
    return pd.DataFrame([{'id': r['id'], 'review': r['text'], 'review_segmented': r['text_seg'],
                          **dict(zip(ASPECTS, r['labels'])), 'province': r['province'], 'facility_type': r['facility']}
                         for r in rows])


def _md_table(df):
    df = df.reset_index()
    head = '| ' + ' | '.join(map(str, df.columns)) + ' |\n|' + '---|' * len(df.columns) + '\n'
    return head + '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in df.values)


def export_release(splits_root, out, prepared=None, log=print):
    """splits_root: work/splits; out: output folder (deleted first); prepared: optional prepared.pkl with cleaning
    statistics (only available when the splits were rebuilt from the raw file)."""
    if os.path.exists(out):
        shutil.rmtree(out)
    for d in ['data', 'splits', 'stats']:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    # ---- S1 as CSV and JSON Lines, with integrity checks
    S1 = {p: to_df(read_jsonl(os.path.join(splits_root, 'S1_main', f'{p}.jsonl'))) for p in PARTS}
    ids = {p: set(d['id']) for p, d in S1.items()}
    txt = {p: set(d['review']) for p, d in S1.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not ids[a] & ids[b], f'duplicate ids between {a} and {b}'
        assert not txt[a] & txt[b], f'duplicate texts between {a} and {b}'
    for p, d in S1.items():
        assert d['id'].is_unique and d[ASPECTS].isin([-2, -1, 0, 1, 2]).all().all()
        d.to_csv(os.path.join(out, 'data', f'{p}.csv'), index=False, encoding='utf-8')
        with open(os.path.join(out, 'data', f'{p}.jsonl'), 'w', encoding='utf-8') as f:
            for r in d.to_dict('records'):
                f.write(json.dumps({'id': int(r['id']), 'review': r['review'], 'review_segmented': r['review_segmented'],
                                    'labels': {a: int(r[a]) for a in ASPECTS}, 'province': r['province'],
                                    'facility_type': r['facility_type']}, ensure_ascii=False) + '\n')
    sizes = {p: len(d) for p, d in S1.items()}
    log(f'S1 sizes: {sizes}; integrity checks passed')

    # ---- ids of S2 to S6 (full rows for S6, because duplicates are not in data/)
    for d in sorted(glob.glob(os.path.join(splits_root, '*'))):
        name = os.path.basename(d)
        if name == 'S1_main' or not os.path.isdir(d):
            continue
        obj = {'description': next((v for k, v in DESC.items() if name.startswith(k)), '')}
        for p in PARTS:
            rows = read_jsonl(os.path.join(d, f'{p}.jsonl'))
            obj[p] = [r['id'] for r in rows]
            if name.startswith('S6') and p == 'test':
                obj['test_seen_in_train'] = [r['id'] for r in rows if r.get('seen_in_train')]
            if name.startswith('S6'):
                to_df(rows).assign(split=p).to_csv(os.path.join(out, 'splits', f'{name}_{p}.csv.gz'), index=False,
                                                    compression='gzip')
        with open(os.path.join(out, 'splits', f'{name}.json'), 'w', encoding='utf-8') as f:
            json.dump(obj, f)
    with open(os.path.join(out, 'splits', 'S1_main.json'), 'w', encoding='utf-8') as f:
        json.dump({'description': 'Main stratified split 80/10/10 (= data/*.csv).',
                   **{p: S1[p]['id'].tolist() for p in PARTS}}, f)

    # ---- statistics
    rows = []
    for p, d in S1.items():
        for a in ASPECTS:
            vc = d[a].value_counts().reindex([-2, -1, 0, 1, 2]).fillna(0).astype(int)
            rows.append({'split': p, 'aspect': a, **{str(k): int(v) for k, v in vc.items()}})
    pd.DataFrame(rows).to_csv(os.path.join(out, 'stats', 'label_distribution_by_split.csv'), index=False)
    ALL = pd.concat(S1.values(), ignore_index=True)
    tot = pd.DataFrame({a: ALL[a].value_counts().reindex([-2, -1, 0, 1, 2]).fillna(0).astype(int) for a in ASPECTS}).T
    tot.columns = ['-2', '-1', '0', '+1', '+2']
    tot.index.name = 'aspect'
    tot['polar %'] = (100 * (1 - tot['0'] / len(ALL))).round(2)
    tot.to_csv(os.path.join(out, 'stats', 'label_distribution_total.csv'))
    prov = pd.crosstab(ALL.province, ALL.facility_type, margins=True, margins_name='Total')
    prov.index.name = 'province'
    prov.to_csv(os.path.join(out, 'stats', 'province_by_facility.csv'))
    w = ALL.review.str.split().str.len()
    overview = {'reviews': len(ALL), **sizes, 'provinces': ALL.province.nunique(),
                'facility types': ALL.facility_type.nunique(), 'aspects': len(ASPECTS),
                'polar aspect opinions': int((ALL[ASPECTS] != 0).values.sum()),
                'mean length (syllables)': round(float(w.mean()), 1), 'median length': int(w.median())}
    json.dump(overview, open(os.path.join(out, 'stats', 'overview.json'), 'w'), indent=1)

    # ---- README and loading example
    clean = {}
    if prepared and os.path.exists(prepared):
        P = pd.read_pickle(prepared)
        clean = {'raw': P['clean_sum']['n_raw'], 'removed': P['clean_sum']['n_removed'],
                 'dup_groups': P['dedup_sum']['dup_groups'], 'dup_rows': P['dedup_sum']['rows_in_dup_groups'],
                 'conflict': P['dedup_sum']['dup_groups_conflicting_labels'],
                 'ties': P['dedup_sum']['dup_groups_dropped_tie'], 'near': P['near_sum']['removed'],
                 'final': P['near_sum']['n_out']}
    with open(os.path.join(out, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(_readme(overview, sizes, tot, prov, clean))
    with open(os.path.join(out, 'load_example.py'), 'w', encoding='utf-8') as f:
        f.write(LOAD_EXAMPLE)

    # ---- checksums and zip
    lines = []
    for root, _, files in os.walk(out):
        for fn in sorted(files):
            if fn != 'SHA256SUMS.txt':
                p = os.path.join(root, fn)
                lines.append(f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}  {os.path.relpath(p, out)}')
    with open(os.path.join(out, 'SHA256SUMS.txt'), 'w') as f:
        f.write('\n'.join(sorted(lines, key=lambda l: l.split()[1])) + '\n')
    zip_path = out.rstrip('/\\') + '_release.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(out.rstrip('/\\')) or '.'))
    big = [l.split()[1] for l in lines if os.path.getsize(os.path.join(out, l.split()[1])) > 100e6]
    log(f'zip: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB); files over 100 MB: {big or "none"}')
    return overview


LOAD_EXAMPLE = '''import pandas as pd

ASPECTS = ["room", "service", "location", "price", "food_and_beverage", "amenities",
           "cleanliness", "transportation", "policy", "others"]

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/{split}.csv")
        print(split, df.shape)
    print(df[["review"] + ASPECTS].head())
'''


def _readme(overview, sizes, tot, prov, clean):
    aspect_rows = '\n'.join(f'| `{a}` | {ASPECT_DEFS.get(a, "")} | {tot.loc[a, "polar %"]} |' for a in ASPECTS)
    cleaning = ''
    if clean:
        cleaning = f'''
## Construction and cleaning

| Step | Count |
|---|---|
| Raw records | {clean['raw']:,} |
| Removed (missing or invalid labels, invalid metadata) | -{clean['removed']:,} |
| Exact-duplicate groups (identical texts; {clean['dup_rows']:,} rows) | {clean['dup_groups']:,} |
| groups with conflicting labels (resolved by majority vote) | {clean['conflict']:,} |
| groups dropped because of a tie | -{clean['ties']:,} |
| Near duplicates removed (MinHash, Jaccard >= 0.85) | -{clean['near']:,} |
| **Final dataset** | **{clean['final']:,}** |
'''
    return f'''# {DATASET}

**{DATASET}** is a large-scale, multi-regional dataset for **Vietnamese aspect-based sentiment analysis (ABSA) in tourism**.
It contains **{overview["reviews"]:,} accommodation reviews** (hotels, resorts, hostels) from **{overview["provinces"]} Vietnamese provinces**,
each annotated with a five-level sentiment score for **10 aspect categories**.

This repository accompanies the paper: *{TITLE}*.

## Label scheme

| Score | Meaning |
|---|---|
| +2 | strongly positive |
| +1 | positive |
| 0 | no polar opinion: the aspect is not mentioned or is mentioned neutrally (the two cases are not distinguished) |
| -1 | negative |
| -2 | strongly negative |

| Aspect | Description | % reviews with a polar opinion |
|---|---|---|
{aspect_rows}

## Files

| File | Content |
|---|---|
| `data/{{train,val,test}}.csv` | Main split (S1), 80/10/10, iterative multi-label stratification |
| `data/{{train,val,test}}.jsonl` | Same content as JSON Lines |
| `splits/*.json` | Review ids of every split used in the paper (S1 to S6) |
| `stats/` | Label distributions, province x facility counts, overview |

Columns: `id`, `review` (pre-processed text: lower-cased, punctuation and stop words removed), `review_segmented`
(word-segmented with pyvi, used by PhoBERT), the 10 aspect columns, `province`, `facility_type`.

## Statistics

| Split | Reviews |
|---|---|
| train | {sizes["train"]:,} |
| val | {sizes["val"]:,} |
| test | {sizes["test"]:,} |
| **total** | **{overview["reviews"]:,}** |

Label distribution (all splits):

{_md_table(tot)}

Reviews per province and facility type:

{_md_table(prov)}
{cleaning}
## Additional splits

| Split | Purpose |
|---|---|
| S2_out_of_region | train on 5 high-volume provinces, test on 4 lower-volume provinces |
| S3_lopo_<province> | leave one province out |
| S4_hotel_to_resort / S4_resort_to_hotel | cross-facility transfer |
| S5_frac<f> | learning curve (nested subsets of S1 train) |
| S6_with_duplicates | split without deduplication; full rows in `splits/S6_with_duplicates_{{train,val,test}}.csv.gz` |

## Intended use and ethics

- Released for non-commercial research only under {LICENSE}.
- Reviews were collected from public review platforms; reviewer names, profiles and other metadata are not included.
- Do not attempt to re-identify reviewers, staff or facilities, and do not use the data to rank individual facilities.
'''
