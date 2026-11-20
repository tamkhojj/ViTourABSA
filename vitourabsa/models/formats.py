"""Output formats for generative models (seq2seq, LLMs), prompt, parsing and few-shot selection."""
import json
import re

import numpy as np

from ..common import ASPECT_VI, ASPECTS

# ---------------------------------------------------------------- seq2seq targets
VI2ASP = {v: k for k, v in ASPECT_VI.items()}
SCORE_WORD = {-2: 'rất tiêu cực', -1: 'tiêu cực', 1: 'tích cực', 2: 'rất tích cực'}
WORD2SCORE = {v: k for k, v in SCORE_WORD.items()}
EMPTY = 'không có'


def to_seq(labels, fmt='num'):
    """fmt 'num': 'phòng: 2; giá cả: -1'   fmt 'word': 'phòng: rất tích cực; giá cả: tiêu cực'"""
    parts = []
    for a, s in zip(ASPECTS, labels):
        if s != 0:
            parts.append(f'{ASPECT_VI[a]}: {s if fmt == "num" else SCORE_WORD[s]}')
    return '; '.join(parts) if parts else EMPTY


def from_seq(text):
    """Accepts both formats. Returns (labels, valid)."""
    out = [0] * len(ASPECTS)
    t = re.sub(r'\s+', ' ', text.strip().lower())
    if t in ('', EMPTY):
        return out, t == EMPTY
    valid = True
    for chunk in t.split(';'):
        if ':' not in chunk:
            valid = False
            continue
        name, val = (x.strip() for x in chunk.split(':', 1))
        if name not in VI2ASP:
            valid = False
            continue
        v = val.replace(' ', '')
        if re.fullmatch(r'[+-]?[12]', v):
            out[ASPECTS.index(VI2ASP[name])] = int(v)
        elif val in WORD2SCORE:
            out[ASPECTS.index(VI2ASP[name])] = WORD2SCORE[val]
        else:
            valid = False
    return out, valid


# ---------------------------------------------------------------- LLM prompt
ASPECT_DEFS = {
    'room': 'the guest room itself: size, bed, bathroom, furniture, view from the room, noise, room condition',
    'service': 'staff and service quality: attitude, friendliness, helpfulness, reception, responsiveness',
    'location': 'where the property is: proximity to beach / centre / attractions, convenience, surroundings',
    'price': 'price, cost, value for money, surcharges',
    'food_and_beverage': 'food, breakfast, buffet, restaurant, drinks, bar',
    'amenities': 'facilities: swimming pool, spa, gym, kids club, wifi, elevator, garden, beach facilities',
    'cleanliness': 'hygiene and cleanliness of room, bathroom, bedding, public areas; smells, dirt, insects',
    'transportation': 'getting around: airport pick-up / shuttle, taxi, bicycles, electric cars, parking',
    'policy': 'rules and procedures: booking, check-in / check-out time, deposit, payment, refund, cancellation',
    'others': 'overall impression of the stay or property not covered by the aspects above '
              '(e.g. "great holiday", "will come back", general atmosphere)',
}

SYSTEM = 'You are an expert annotator for aspect-based sentiment analysis of Vietnamese hotel and resort reviews.'


def instruction():
    defs = '\n'.join(f'- {a}: {d}' for a, d in ASPECT_DEFS.items())
    return (
        'Reviews are mostly pre-processed (lower-cased, punctuation and Vietnamese stop words removed).\n'
        'For EACH of the 10 aspects, give a score:\n'
        '  2 = strongly positive, 1 = positive / mildly positive, 0 = aspect not mentioned OR neutral, '
        '-1 = negative / mildly negative, -2 = strongly negative.\n'
        f'Aspects:\n{defs}\n'
        'Answer with ONLY a JSON object containing exactly these 10 keys and integer values, e.g.\n'
        '{"room": 2, "service": 0, "location": 0, "price": -1, "food_and_beverage": 0, "amenities": 0, '
        '"cleanliness": 1, "transportation": 0, "policy": 0, "others": 0}')


def user_msg(review):
    return f'Review: "{review}"'


def to_json(labels):
    return json.dumps(dict(zip(ASPECTS, [int(x) for x in labels])))


def from_json(text):
    m = re.search(r'\{.*?\}', text, flags=re.S)
    if not m:
        return [0] * len(ASPECTS), False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [0] * len(ASPECTS), False
    out, valid = [], True
    for a in ASPECTS:
        if a not in obj:
            valid = False
        v = obj.get(a, 0)
        try:
            v = int(round(float(v if v is not None else 0)))
        except (TypeError, ValueError):
            valid, v = False, 0
        if not -2 <= v <= 2:
            valid, v = False, max(-2, min(2, v))
        out.append(v)
    return out, valid


def messages(review, shots=()):
    msgs = [{'role': 'system', 'content': SYSTEM + '\n\n' + instruction()}]   # instructions once, shots stay short
    for ex in shots:
        msgs.append({'role': 'user', 'content': user_msg(ex['text'])})
        msgs.append({'role': 'assistant', 'content': to_json(ex['labels'])})
    msgs.append({'role': 'user', 'content': user_msg(review)})
    return msgs


def render(tok, msgs, add_generation_prompt=True):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:                      # template without a system role -> merge into first user turn
        m = [dict(x) for x in msgs]
        if m[0]['role'] == 'system':
            sys_ = m.pop(0)['content']
            m[0]['content'] = sys_ + '\n\n' + m[0]['content']
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=add_generation_prompt)


def select_fewshot(train_rows, k=10, seed=0, max_words=50):
    """Greedy, deterministic: cover every aspect, every polar level, a mixed-polarity review,
    a review with <=1 polar aspect and one with >=4, preferring short reviews."""
    rng = np.random.RandomState(seed)
    pool = [r for r in train_rows if 5 <= len(r['text'].split()) <= max_words]
    rng.shuffle(pool)
    pool = pool[:20000]
    need = {('a', a) for a in range(len(ASPECTS))} | {('s', s) for s in (-2, -1, 1, 2)} | \
           {('few',), ('many',), ('mixed',)}

    def feats(r):
        f, lab = set(), r['labels']
        for a, s in enumerate(lab):
            if s:
                f |= {('a', a), ('s', s)}
        nz = sum(1 for s in lab if s)
        if nz <= 1:
            f.add(('few',))
        if nz >= 4:
            f.add(('many',))
        if any(s > 0 for s in lab) and any(s < 0 for s in lab):
            f.add(('mixed',))
        return f

    chosen = []
    while len(chosen) < k and pool:
        best = max(pool, key=lambda r: (len(feats(r) & need), -len(r['text'])))
        chosen.append(best)
        need -= feats(best)
        pool.remove(best)
    return chosen