"""Additional analyses of the paper (no training; reads work/results). Outputs go to work/reports/extra/.

X01 shift measured on identical test reviews, X02 learning-curve table, X03 hyper-parameters actually used,
X04 LLM detection precision / recall, X05 bootstrap confidence intervals, X06 per-level F1.

    python scripts/09_extra_analysis.py
"""
import _root  # noqa: F401
import pandas as pd

from vitourabsa import outputs, paths
from vitourabsa.analysis import extra
from vitourabsa.pipeline import log

if __name__ == '__main__':
    outputs.configure()
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    for title, fn in [('X01 shift on identical reviews', extra.shift_same_reviews),
                      ('X02 learning curve', extra.learning_curve),
                      ('X04 LLM detection', extra.llm_detection),
                      ('X05 bootstrap confidence intervals', extra.bootstrap_intervals),
                      ('X06 per-level F1', extra.per_class_f1)]:
        print(f'\n### {title}')
        print(fn())
    hp, other = extra.hyperparameters()
    print('\n### X03 hyper-parameters')
    print(hp)
    for f, c in other.items():
        print(f, c)
    log(f'saved to {paths.EXTRA_OUT}')
