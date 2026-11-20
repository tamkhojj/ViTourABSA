"""Step 1. Dataset statistics and exploratory analysis (tables T01 to T13, figures F01 to F06).

Outputs: work/reports/eda/*.csv, *.tex, *.pdf, *.png

    python scripts/01_eda.py

The cleaning table (T01) and the duplicate analysis (T09, T10, F06) need the data before deduplication, which is
only available after `00_prepare_splits.py --from-raw`; they are skipped otherwise.
"""
import os

import _root  # noqa: F401
import pandas as pd

from vitourabsa import outputs, paths
from vitourabsa.analysis import eda
from vitourabsa.common import read_jsonl
from vitourabsa.data.release import load_main, to_analysis_frame
from vitourabsa.pipeline import log


def main():
    outputs.configure()
    P = pd.read_pickle(paths.PREPARED) if os.path.exists(paths.PREPARED) else None
    final = P['final'] if P is not None else to_analysis_frame(load_main(paths.RELEASE))
    log(f'{len(final):,} reviews')

    if P is not None:
        eda.table_pipeline(P['clean_sum'], P['dedup_sum'], P['near_sum'])                 # T01
    eda.table_overview(final)                                                              # T02
    eda.table_by_group(final, 'province')                                                  # T03
    eda.table_by_group(final, 'facility')
    eda.table_province_facility(final)                                                     # T04
    eda.fig_length(final)                                                                  # F01
    eda.table_label_distribution(final)                                                    # T05
    eda.fig_label_distribution(final)                                                      # F02
    eda.fig_cooccurrence(final)                                                            # F03, T06
    eda.province_aspect(final, 'province')                                                 # F04, T07
    eda.province_aspect(final, 'facility')
    eda.fig_review_polarity(final)                                                         # F05
    eda.lexical_markers(final)                                                             # T08
    if P is not None:
        eda.duplicates_report(P['with_dups'], P['dedup_sum'])                              # T09, T10, F06

    s1 = paths.split_dir('S1_main')
    if os.path.exists(os.path.join(s1, 'train.jsonl')):
        tr = read_jsonl(os.path.join(s1, 'train.jsonl'))
        te = read_jsonl(os.path.join(s1, 'test.jsonl'))
        eda.text_characteristics(final, [r['text'] for r in tr], [r['text'] for r in te])   # T11
        names = [n for n in sorted(os.listdir(paths.SPLITS)) if os.path.isdir(os.path.join(paths.SPLITS, n))]
        eda.split_statistics(paths.SPLITS, names)                                          # T12, T13
    else:
        log('work/splits not found: run scripts/00_prepare_splits.py for T11 to T13')
    log(f'saved to {paths.EDA_OUT}')


if __name__ == '__main__':
    main()
