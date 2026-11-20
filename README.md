# ViTourABSA

**ViTourABSA** is a large-scale, multi-regional dataset and benchmark for **Vietnamese aspect-based sentiment
analysis (ABSA) in tourism**. It contains **87,314 accommodation reviews** (hotels, resorts and hostels) from
**nine Vietnamese provinces**. Each review is annotated for **ten aspects** on a **five-level scale from -2 to +2**.

This repository accompanies the paper *A large-scale multi-regional dataset and benchmark for Vietnamese
aspect-based sentiment analysis in tourism* and contains the data, every evaluation split used in the paper and
the code to reproduce all experiments.

| | |
|---|---|
| Reviews | 87,314 (train 69,851 / val 8,729 / test 8,734) |
| Sources | Google Reviews, Agoda, Traveloka |
| Collection period | May 2025 to February 2026 |
| Provinces | 9 (administrative boundaries after 1 July 2025) |
| Facility types | hotel, resort, hostel |
| Aspects | 10, each scored in {-2, -1, 0, +1, +2} |
| Polar aspect opinions | 259,814 |
| Review length | mean 38.9 syllables, median 24 |
| Inter-annotator agreement | ordinal Krippendorff's α = 0.930 (300 reviews, 3,000 aspect slots) |

## Contents

- [Repository layout](#repository-layout)
- [Label scheme](#label-scheme)
- [Data format](#data-format)
- [Statistics](#statistics)
- [Data collection and annotation](#data-collection-and-annotation)
- [Evaluation splits](#evaluation-splits)
- [Evaluation metrics](#evaluation-metrics)
- [Benchmark results](#benchmark-results)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Intended use, limitations and ethics](#intended-use-limitations-and-ethics)
- [Citation](#citation)

## Repository layout

```
ViTourABSA/
├── README.md             this file (dataset card)
├── CODE.md               detailed guide to the code
├── data/                 main split S1: {train,val,test}.csv and .jsonl
├── splits/               review ids of S1 to S6 (*.json); full rows of S6 (*.csv.gz)
├── stats/                label distributions, province x facility counts, overview
├── load_example.py       minimal loading example
├── SHA256SUMS.txt        checksums of the released files
├── requirements.txt      Python dependencies
├── pyproject.toml        optional: pip install -e .
├── configs/              benchmark.yaml: models, seeds and learning-rate grids of the paper
├── scripts/              command-line entry points, numbered in the order they are run
├── vitourabsa/           Python package: data, models, metrics, analysis
└── work/                 created by the scripts (git-ignored): splits, results, checkpoints, reports
```

The data folders (`data/`, `splits/`, `stats/`) are never modified by the code. Everything the code generates is
written to `work/`.

## Label scheme

| Score | Meaning |
|---|---|
| +2 | strongly positive |
| +1 | positive |
| 0 | no polar opinion: the aspect is not mentioned or is mentioned neutrally (the two cases are not distinguished) |
| -1 | negative |
| -2 | strongly negative |

| Aspect | Covers | Reviews with a polar opinion (%) |
|---|---|---|
| `room` | the guest room: size, bed, bathroom, furniture, view, noise, condition | 37.61 |
| `service` | staff and service: attitude, friendliness, helpfulness, reception, responsiveness | 71.59 |
| `location` | proximity to the beach, centre or attractions, convenience, surroundings | 32.56 |
| `price` | price, cost, value for money, surcharges | 9.01 |
| `food_and_beverage` | food, breakfast, buffet, restaurant, drinks, bar | 36.61 |
| `amenities` | swimming pool, spa, gym, kids club, wifi, elevator, garden, beach facilities | 22.10 |
| `cleanliness` | hygiene of room, bathroom, bedding and public areas; smells, dirt, insects | 25.65 |
| `transportation` | airport pick-up, shuttle, taxi, bicycles, electric cars, parking | 7.71 |
| `policy` | booking, check-in and check-out, deposit, payment, refund, cancellation | 5.91 |
| `others` | overall impression not covered above (e.g. "great holiday", "will come back") | 48.83 |

## Data format

`data/{train,val,test}.csv` contain one review per row:

| Column | Content |
|---|---|
| `id` | review id, unique across the dataset and used in `splits/*.json` |
| `review` | pre-processed text: lower-cased, punctuation and stop words removed, proper names replaced by placeholders |
| `review_segmented` | the same text word-segmented with pyvi (input for PhoBERT) |
| `room` ... `others` | the ten aspect scores, integers in {-2, ..., 2} |
| `province` | one of the nine provinces |
| `facility_type` | `hotel`, `resort` or `hostel` |

`data/{train,val,test}.jsonl` contain the same records, with the ten scores grouped under `labels`.

```python
import pandas as pd

ASPECTS = ["room", "service", "location", "price", "food_and_beverage", "amenities",
           "cleanliness", "transportation", "policy", "others"]

train = pd.read_csv("data/train.csv")
X, Y = train["review"], train[ASPECTS].values          # Y has shape (n, 10), values in {-2, ..., 2}

# Hugging Face datasets
from datasets import load_dataset
ds = load_dataset("csv", data_files={"train": "data/train.csv",
                                     "validation": "data/val.csv",
                                     "test": "data/test.csv"})
```

Run `python load_example.py` for a minimal check, and `sha256sum -c SHA256SUMS.txt` to verify the files.

## Statistics

Label distribution over the whole dataset (also in `stats/label_distribution_total.csv`; per split in
`stats/label_distribution_by_split.csv`):

| Aspect | -2 | -1 | 0 | +1 | +2 | Polar (%) |
|---|---:|---:|---:|---:|---:|---:|
| room | 1,862 | 2,806 | 54,477 | 4,746 | 23,423 | 37.61 |
| service | 4,507 | 2,465 | 24,809 | 2,186 | 53,347 | 71.59 |
| location | 189 | 966 | 58,888 | 1,855 | 25,416 | 32.56 |
| price | 625 | 1,892 | 79,443 | 3,553 | 1,801 | 9.01 |
| food_and_beverage | 1,132 | 2,529 | 55,351 | 3,399 | 24,903 | 36.61 |
| amenities | 743 | 1,950 | 68,016 | 2,140 | 14,465 | 22.10 |
| cleanliness | 1,345 | 1,503 | 64,922 | 1,167 | 18,377 | 25.65 |
| transportation | 673 | 1,273 | 80,585 | 765 | 4,018 | 7.71 |
| policy | 1,142 | 1,116 | 82,156 | 282 | 2,618 | 5.91 |
| others | 3,745 | 3,808 | 44,679 | 2,655 | 32,427 | 48.83 |

Reviews per province and facility type (`stats/province_by_facility.csv`):

| Province | Hostel | Hotel | Resort | Total |
|---|---:|---:|---:|---:|
| An Giang | 0 | 1,592 | 1 | 1,593 |
| Da Nang | 0 | 6,859 | 12,272 | 19,131 |
| Gia Lai | 0 | 1,919 | 0 | 1,919 |
| Hanoi | 1,075 | 16,198 | 0 | 17,273 |
| Ho Chi Minh City | 0 | 13,899 | 4,696 | 18,595 |
| Khanh Hoa | 0 | 7,314 | 12,307 | 19,621 |
| Lam Dong | 0 | 1,102 | 0 | 1,102 |
| Quang Ninh | 0 | 6,088 | 0 | 6,088 |
| Thanh Hoa | 0 | 1,767 | 225 | 1,992 |
| **Total** | **1,075** | **56,738** | **29,501** | **87,314** |

## Data collection and annotation

**Collection.** Public reviews of accommodation facilities were collected from Google Reviews, Agoda and
Traveloka between May 2025 and February 2026. Province labels follow the administrative boundaries in force after
the provincial merger of 1 July 2025.

**Annotation.** Four native Vietnamese annotators (three master's-level annotators and one PhD-level expert)
labelled each review for the ten aspects following written guidelines. Disagreements were first resolved by a
vote among the annotators and then reviewed with the expert, who gave final approval. Annotators were paid for
their work.

**Agreement.** A second annotator independently labelled 300 reviews without access to the existing labels.
Over the 3,000 aspect slots, the ordinal Krippendorff's α is 0.930, the nominal α is 0.791 and the quadratically
weighted kappa is 0.918. Most of the 331 disagreeing slots differ only in intensity (207) or in whether an
opinion is present (113); only 11 have opposite polarities.

**Cleaning.**

| Step | Count |
|---|---:|
| Raw records | 100,054 |
| Removed: corrupted rows, missing labels, invalid metadata | -995 |
| Exact-duplicate groups (two or more identical texts, 16,975 rows) | 5,820 |
| of which with conflicting labels, resolved by majority vote | 633 |
| of which with a tied vote, dropped | -328 |
| Removed: near duplicates (MinHash, Jaccard ≥ 0.85) | -262 |
| **Final dataset** | **87,314** |

## Evaluation splits

Each `splits/<name>.json` file contains the review ids of `train`, `val` and `test` and a short `description`.
All splits except S6 are built from the deduplicated dataset with iterative multi-label stratification (seed 42).

| Split | Files | Test data | Train | Val | Test |
|---|---|---|---:|---:|---:|
| S1 in-distribution | `S1_main.json` (= `data/`) | random 10% | 69,851 | 8,729 | 8,734 |
| S2 cross-region | `S2_out_of_region.json` | 4 lower-volume provinces | 72,638 | 8,070 | 6,606 |
| S3 leave one province out | `S3_lopo_<province>.json` | 1 of 5 high-volume provinces | 54,978 to 67,159 | 6,109 to 7,461 | 6,088 to 19,621 |
| S4 hotel to resort | `S4_hotel_to_resort.json` | all resort reviews | 51,064 | 5,674 | 29,501 |
| S4 resort to hotel | `S4_resort_to_hotel.json` | all hotel reviews | 26,549 | 2,952 | 56,738 |
| S5 learning curve | `S5_frac{0.01,0.05,0.1,0.25,0.5}.json` | S1 test | 698 to 34,925 | 8,729 | 8,734 |
| S6 with duplicates | `S6_with_duplicates.json` + `S6_with_duplicates_*.csv.gz` | random 10% | 79,247 | 9,905 | 9,907 |

S6 is split before deduplication, so it contains reviews that are not in `data/`. Its full rows are therefore
released in `splits/S6_with_duplicates_{train,val,test}.csv.gz`. The key `test_seen_in_train` lists the 1,551
test reviews (15.7%) that also occur verbatim in the training set.

## Evaluation metrics

All subtasks are computed from the same five-level predictions.

- **T1, polar-aspect detection:** F1 for predicting whether a score is different from 0, over all review-aspect slots.
- **T2, intensity classification:** macro-F1 over {-2, -1, +1, +2} on slots with a polar gold label; predicting 0 counts as an error.
- **T3, end-to-end five-level classification (primary metric):** for each aspect, macro-F1 over the levels present in the gold labels, then averaged over the ten aspects.
- **Joint F1** over (aspect, score) pairs, **quadratic weighted kappa** and **exact match** (all ten scores correct).

```python
import numpy as np
from vitourabsa.metrics import evaluate

y = np.array([[2, 0, 0, -1, 0, 0, 0, 0, 0, 1]])        # gold, shape (n_reviews, 10)
p = np.array([[2, 0, 0, -2, 0, 0, 0, 0, 0, 1]])        # predictions
print(evaluate(y, p, per_aspect=False)["t3_macro_f1"])
```

## Benchmark results

T3 macro-F1 on the S1 test set (mean over seeds for fine-tuned encoders). The full table with all metrics, standard
deviations and significance tests is in the paper.

| Group | System | T3 macro-F1 | Joint F1 |
|---|---|---:|---:|
| Baseline | Majority | 16.79 | 7.12 |
| Baseline | TF-IDF + logistic regression | 51.88 | 65.30 |
| Encoder | PhoBERT-base-v2 | 59.66 | 70.47 |
| Encoder | XLM-R-large | 60.49 | 70.64 |
| Encoder | **PhoBERT-base, review-aspect pair** | **61.15** | **71.09** |
| Encoder | PhoBERT-base, class-weighted CE | 60.95 | 69.08 |
| Seq2seq | ViT5-base | 52.53 | 66.41 |
| LLM | Llama-3.1-8B-Instruct, 10-shot | 49.66 | 56.08 |
| LLM | Qwen2.5-7B-Instruct, 10-shot | 48.07 | 51.43 |
| LLM | Qwen2.5-7B-Instruct, QLoRA | 53.43 | 66.45 |
| LLM | Vistral-7B-Chat, QLoRA | 55.45 | 68.03 |

Main findings:

- The best system, PhoBERT-base with the review-aspect pair formulation, is not significantly better than four other encoder configurations (paired bootstrap, Holm correction).
- Prompted LLMs trail the best encoder by 11 to 22 points; QLoRA fine-tuning narrows the gap.
- On identical test reviews, fine-tuned encoders lose 7.6 to 7.9 points when the test provinces are excluded from training and 8.9 to 10.5 points when transferred between hotels and resorts.
- Keeping duplicate reviews (S6) inflates the TF-IDF baseline by 4.3 points.

## Reproducing the experiments

Full instructions are in **[CODE.md](CODE.md)**. In short:

```bash
conda create -n vitourabsa python=3.11 -y && conda activate vitourabsa
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
export HF_TOKEN=hf_xxx                    # only for the gated LLMs (Llama-3.1, Vistral)

python scripts/00_prepare_splits.py       # build S1 to S6 in work/splits from data/ and splits/
python scripts/01_eda.py                  # dataset statistics, tables and figures
python scripts/07_run_benchmark.py        # all experiments; finished runs are skipped
python scripts/08_make_report.py          # result tables and figures
python scripts/09_extra_analysis.py       # additional analyses
```

| Script | Purpose |
|---|---|
| `00_prepare_splits.py` | build the experiment splits (identical to those used in the paper) |
| `01_eda.py` | dataset statistics, tables and figures |
| `02_run_classic.py` | majority, prior-random, TF-IDF + LR / SVM |
| `03_run_encoder.py` | PhoBERT, ViSoBERT, XLM-R, mDeBERTa and their variants |
| `04_run_seq2seq.py` | ViT5 |
| `05_run_llm.py` | LLM zero-shot, 10-shot and QLoRA evaluation |
| `06_train_qlora.py` | QLoRA adapters |
| `07_run_benchmark.py` | the whole benchmark, step by step (`--steps` selects a subset) |
| `08_make_report.py` | result tables, significance tests and figures |
| `09_extra_analysis.py` | shift on identical reviews, confidence intervals, per-level F1 |
| `10_agreement.py` | inter-annotator agreement between two annotation files |
| `11_human_annotation.py` | optional second-annotator study on the test set |
| `12_export_release.py` | rebuild the public release (authors only, needs the raw file) |

All experiments in the paper were run on a single NVIDIA RTX 4090 (24 GB); the complete benchmark takes several
days. Models, seeds and learning-rate grids are set in `configs/benchmark.yaml`.

## Intended use, limitations and ethics

- The dataset is released for **non-commercial research only**
- Reviews were collected from public platforms. Reviewer names, profiles and other metadata are not included, and
  names of persons and accommodation properties in the review text are replaced by placeholders.
- Do not attempt to re-identify reviewers, staff members or facilities, do not use the data to evaluate or rank
  individual facilities, and comply with the terms of service of the original platforms.
- Score 0 combines "not mentioned" and "neutral", so neutral mentions cannot be studied.
- Texts are pre-processed (lower-cased, punctuation and stop words removed); the raw texts cannot be released.
  Models trained on these texts should be applied to text pre-processed in the same way.
- Coverage is uneven: four provinces have fewer than 2,000 reviews, all hostel reviews come from Hanoi, and the
  agreement subsample contains only hotel reviews from Ho Chi Minh City.
- If you find content that should be removed, please open an issue.

## Citation

If you use ViTourABSA, please cite:

```bibtex
@article{tran2026vitourabsa,
  title   = {A large-scale multi-regional dataset and benchmark for {Vietnamese} aspect-based sentiment analysis in tourism},
  author  = {Tran, Tam Khoi and Tran, Phuoc},
  journal = {Under review},
  year    = {2026},
  url     = {https://github.com/tamkhojj/ViTourABSA}
}
```
# ViTourABSA
