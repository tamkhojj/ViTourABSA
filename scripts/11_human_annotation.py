"""Optional: second-annotator study on random S1 test reviews (human reference scores).

1. export a sheet of 1,500 random S1 test reviews:   python scripts/11_human_annotation.py export
2. the annotator fills the ten aspect columns and saves work/human_annotation/annotation_filled.csv
3. compute agreement and human scores:            python scripts/11_human_annotation.py evaluate
   (08_make_report.py then adds the human row to the main table automatically)
"""
import argparse

import _root  # noqa: F401

from vitourabsa import outputs, paths
from vitourabsa.analysis import human

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['export', 'evaluate'])
    ap.add_argument('--n', type=int, default=1500)
    a = ap.parse_args()
    outputs.configure()
    if a.action == 'export':
        sheet = human.export_sheet(paths.split_dir('S1_main'), n=a.n)
        print(f'{len(sheet)} reviews -> {human.SHEET}')
    else:
        res = human.evaluate_human(paths.split_dir('S1_main'))
        if res is None:
            raise SystemExit(f'{human.FILLED} not found')
        print(res['iaa'].round(3))
        print(res['metrics'])
