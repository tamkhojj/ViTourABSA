"""Inter-annotator agreement between two annotation files of the same reviews (the 300-review study of the paper).

Each CSV needs a `review_id` column and the ten aspect columns. In the paper, file A contains the adjudicated
labels and file B the independent annotation.

    python scripts/10_agreement.py --a iaa_annotator_A_tourism.csv --b iaa_annotator_B_tourism.csv
"""
import argparse
import json
import os

import _root  # noqa: F401
import pandas as pd

from vitourabsa import paths
from vitourabsa.analysis import iaa

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', required=True, help='reference annotation (adjudicated labels)')
    ap.add_argument('--b', required=True, help='second, independent annotation')
    a = ap.parse_args()
    A, B, ids = iaa.load_pair(a.a, a.b)
    table = iaa.agreement_table(A, B)
    kinds = iaa.disagreement_types(A, B)
    as_system = iaa.score_as_system(A, B)
    with pd.option_context('display.width', 220, 'display.max_columns', 20):
        print(table.round(3))
    print('\nDisagreements:', kinds)
    print('B scored against A with the benchmark metrics:', as_system)
    os.makedirs(paths.EXTRA_OUT, exist_ok=True)
    table.to_csv(os.path.join(paths.EXTRA_OUT, 'IAA_agreement.csv'), encoding='utf-8-sig')
    with open(os.path.join(paths.EXTRA_OUT, 'IAA_summary.json'), 'w') as f:
        json.dump({'n_reviews': int(len(ids)), 'disagreements': kinds, 'b_vs_a_metrics': as_system}, f, indent=1)
    print('saved to', paths.EXTRA_OUT)
