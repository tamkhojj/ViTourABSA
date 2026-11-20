"""Agreement between two annotations of the same reviews (used for the 300-review study in the paper).

Both files need a `review_id` column and the ten aspect columns with scores in {-2,...,2}. In the paper,
file A holds the adjudicated (released) labels and file B the independent annotation.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..common import ASPECTS
from ..metrics import evaluate


def _alpha(x, y, level):
    import krippendorff
    kw = {'value_domain': [-2, -1, 0, 1, 2]} if level == 'ordinal' else {}
    return krippendorff.alpha(reliability_data=np.vstack([x, y]).astype(float), level_of_measurement=level, **kw)


def load_pair(file_a, file_b):
    a = pd.read_csv(file_a)
    b = pd.read_csv(file_b).set_index('review_id').loc[a.review_id].reset_index()
    return a[ASPECTS].values.astype(int), b[ASPECTS].values.astype(int), a.review_id.values


def agreement_table(a, b):
    """Per-aspect and overall agreement: ordinal and nominal alpha, quadratic kappa, exact and detection agreement,
    and sign / level agreement on slots that both annotations mark as polar."""
    rows = []
    for k, asp in enumerate(ASPECTS + ['ALL']):
        x, y = (a.ravel(), b.ravel()) if asp == 'ALL' else (a[:, k], b[:, k])
        both = (x != 0) & (y != 0)
        rows.append({'aspect': asp, 'polar A': int((x != 0).sum()), 'polar B': int((y != 0).sum()),
                     'alpha ordinal': _alpha(x, y, 'ordinal'), 'alpha nominal': _alpha(x, y, 'nominal'),
                     'QWK': cohen_kappa_score(x, y, weights='quadratic', labels=[-2, -1, 0, 1, 2]),
                     'exact %': 100 * (x == y).mean(), 'detection %': 100 * ((x != 0) == (y != 0)).mean(),
                     'sign agree % (both polar)': 100 * (np.sign(x[both]) == np.sign(y[both])).mean() if both.any() else np.nan,
                     'level agree % (both polar)': 100 * (x[both] == y[both]).mean() if both.any() else np.nan})
    return pd.DataFrame(rows).set_index('aspect')


def disagreement_types(a, b):
    d = a != b
    return {'slots': int(d.size), 'disagreeing': int(d.sum()),
            'detection (one 0, other polar)': int(((a == 0) != (b == 0)).sum()),
            'intensity (same sign)': int((d & (a != 0) & (b != 0) & (np.sign(a) == np.sign(b))).sum()),
            'polarity flip': int((np.sign(a) * np.sign(b) == -1).sum())}


def score_as_system(reference, hypothesis):
    """Scores of one annotation evaluated against the other with the benchmark metrics (percentages, QWK raw)."""
    e = evaluate(reference, hypothesis, per_aspect=False)
    keys = ['t1_f1_micro', 't2_macro_f1', 't3_macro_f1', 'joint_f1_macro', 't3_qwk', 'exact_match']
    return {k: (round(e[k], 3) if k == 't3_qwk' else round(100 * e[k], 2)) for k in keys}
