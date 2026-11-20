"""Shared constants / helpers.

Label convention (as released): each aspect has a score in {-2,-1,0,+1,+2}.
0 = "no polar opinion" (aspect not mentioned OR neutral; the two are not distinguished in the data).
Internally we store scores as ints -2..2; class index = score + 2 (0..4).
"""
import json
import os
import random

import numpy as np

ASPECTS = ['room', 'service', 'location', 'price', 'food_and_beverage',
           'amenities', 'cleanliness', 'transportation', 'policy', 'others']
A = len(ASPECTS)
SCORES = [-2, -1, 0, 1, 2]
SCORE_NAMES = ['-2', '-1', '0', '+1', '+2']
PROVINCES = ['khanh_hoa', 'da_nang', 'hanoi', 'ho_chi_minh', 'quangninh',
             'thanh_hoa', 'an_giang', 'gia_lai', 'lam_dong']
BIG_PROVINCES = ['khanh_hoa', 'da_nang', 'hanoi', 'ho_chi_minh', 'quangninh']
SMALL_PROVINCES = ['thanh_hoa', 'an_giang', 'gia_lai', 'lam_dong']
PROVINCE_VI = {'khanh_hoa': 'Khánh Hòa', 'da_nang': 'Đà Nẵng', 'hanoi': 'Hà Nội', 'ho_chi_minh': 'TP. Hồ Chí Minh',
               'quangninh': 'Quảng Ninh', 'thanh_hoa': 'Thanh Hóa', 'an_giang': 'An Giang', 'gia_lai': 'Gia Lai',
               'lam_dong': 'Lâm Đồng'}
ASPECT_VI = {'room': 'phòng', 'service': 'dịch vụ', 'location': 'vị trí', 'price': 'giá cả',
             'food_and_beverage': 'đồ ăn thức uống', 'amenities': 'tiện nghi', 'cleanliness': 'vệ sinh',
             'transportation': 'di chuyển', 'policy': 'chính sách', 'others': 'khác'}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_np_default)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_np_default) + '\n')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def labels_of(rows):
    return np.array([r['labels'] for r in rows], dtype=np.int64)