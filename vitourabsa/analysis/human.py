"""Human re-annotation: export a sheet, then compute IAA (Krippendorff's alpha, weighted kappa) and the
human upper bound once the filled sheet exists."""
import os

import numpy as np
import pandas as pd

from ..common import ASPECTS, read_jsonl, save_json
from ..metrics import evaluate

HUMAN_DIR = 'human_annotation'
SHEET = os.path.join(HUMAN_DIR, 'annotation_sheet.csv')
FILLED = os.path.join(HUMAN_DIR, 'annotation_filled.csv')


def set_dir(path):
    """Change the folder that holds the annotation sheet and the filled sheet."""
    global HUMAN_DIR, SHEET, FILLED
    HUMAN_DIR = path
    SHEET = os.path.join(HUMAN_DIR, 'annotation_sheet.csv')
    FILLED = os.path.join(HUMAN_DIR, 'annotation_filled.csv')

GUIDE = """HƯỚNG DẪN GÁN NHÃN LẠI (dùng ĐÚNG guideline gốc của dataset)
- Mỗi dòng là một review (đã tiền xử lý). Với MỖI aspect, điền một số nguyên trong {-2,-1,0,1,2}:
   2 = rất tích cực, 1 = tích cực, 0 = không nhắc tới HOẶC trung tính, -1 = tiêu cực, -2 = rất tiêu cực.
- Ô để trống được hiểu là 0.
- KHÔNG xem nhãn gốc. Người gán nhãn phải khác người đã gán nhãn review đó ban đầu.
- Lưu file đã điền thành:  annotation_filled.csv trong cùng thư mục (giữ nguyên cột id).
"""


def export_sheet(split_dir, n=1500, seed=2026):
    os.makedirs(HUMAN_DIR, exist_ok=True)
    if os.path.exists(SHEET):
        return pd.read_csv(SHEET)
    te = read_jsonl(os.path.join(split_dir, 'test.jsonl'))
    idx = np.sort(np.random.RandomState(seed).choice(len(te), min(n, len(te)), replace=False))
    df = pd.DataFrame([{'id': te[i]['id'], 'text': te[i]['text'], **{a: '' for a in ASPECTS}} for i in idx])
    df.to_csv(SHEET, index=False, encoding='utf-8-sig')
    try:
        df.to_excel(SHEET.replace('.csv', '.xlsx'), index=False)
    except Exception:
        pass
    with open(os.path.join(HUMAN_DIR, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(GUIDE)
    return df


def _kripp_ordinal(a, b):
    """Krippendorff's alpha (ordinal metric) for two coders, values -2..2, no missing data."""
    try:
        import krippendorff
        return float(krippendorff.alpha(reliability_data=np.vstack([a, b]).astype(float),
                                        level_of_measurement='ordinal', value_domain=[-2, -1, 0, 1, 2]))
    except ImportError:
        return np.nan


def evaluate_human(split_dir):
    """Returns None if the filled sheet does not exist yet."""
    if not os.path.exists(FILLED):
        return None
    from sklearn.metrics import cohen_kappa_score
    h = pd.read_csv(FILLED)
    for a in ASPECTS:
        h[a] = pd.to_numeric(h[a], errors='coerce').fillna(0).clip(-2, 2).round().astype(int)
    gold = {r['id']: r['labels'] for r in read_jsonl(os.path.join(split_dir, 'test.jsonl'))}
    h = h[h['id'].isin(gold)]
    G = np.array([gold[i] for i in h['id']])
    H = h[ASPECTS].values
    rows = []
    for k, a in enumerate(ASPECTS):
        rows.append({'aspect': a, 'n_polar_gold': int((G[:, k] != 0).sum()),
                     "Krippendorff's α (ordinal)": _kripp_ordinal(G[:, k], H[:, k]),
                     'Weighted κ (quadratic)': cohen_kappa_score(G[:, k], H[:, k], weights='quadratic', labels=[-2, -1, 0, 1, 2]),
                     'Exact agreement %': 100 * (G[:, k] == H[:, k]).mean(),
                     'Detection agreement %': 100 * ((G[:, k] != 0) == (H[:, k] != 0)).mean()})
    iaa = pd.DataFrame(rows).set_index('aspect')
    iaa.loc['All slots'] = [int((G != 0).sum()), _kripp_ordinal(G.ravel(), H.ravel()),
                            cohen_kappa_score(G.ravel(), H.ravel(), weights='quadratic', labels=[-2, -1, 0, 1, 2]),
                            100 * (G == H).mean(), 100 * ((G != 0) == (H != 0)).mean()]
    human_metrics = evaluate(G, H)
    save_json({'n': len(h), 'metrics': human_metrics}, os.path.join(HUMAN_DIR, 'human_metrics.json'))
    iaa.to_csv(os.path.join(HUMAN_DIR, 'iaa.csv'))
    return {'iaa': iaa, 'metrics': human_metrics, 'ids': h['id'].tolist()}