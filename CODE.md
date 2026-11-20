# Code for the ViTourABSA benchmark

This document explains how to reproduce the experiments of the paper
*A large-scale multi-regional dataset and benchmark for Vietnamese aspect-based sentiment analysis in tourism*
from the released data.

## Repository layout

```
ViTourABSA/
├── README.md                 dataset card
├── CODE.md                   this file
├── data/                     main split S1: train / val / test (.csv and .jsonl)
├── splits/                   review ids of S1 to S6; full rows of S6 (*.csv.gz)
├── stats/                    label distributions, province x facility, overview
├── load_example.py           minimal loading example
├── SHA256SUMS.txt            checksums of the released files
│
├── requirements.txt
├── pyproject.toml            optional: pip install -e .
├── configs/
│   └── benchmark.yaml        models, seeds and learning-rate grids used in the paper
├── scripts/                  command-line entry points, numbered in the order they are run
│   ├── 00_prepare_splits.py  released files -> work/splits (identical to the paper)
│   ├── 01_eda.py             dataset statistics, tables and figures
│   ├── 02_run_classic.py     majority, prior-random, TF-IDF + LR / SVM
│   ├── 03_run_encoder.py     PhoBERT, ViSoBERT, XLM-R, mDeBERTa and their variants
│   ├── 04_run_seq2seq.py     ViT5
│   ├── 05_run_llm.py         LLM zero-shot / 10-shot / QLoRA evaluation
│   ├── 06_train_qlora.py     QLoRA adapters
│   ├── 07_run_benchmark.py   the whole benchmark, step by step
│   ├── 08_make_report.py     result tables and figures
│   ├── 09_extra_analysis.py  shift on identical reviews, confidence intervals, per-level F1, ...
│   ├── 10_agreement.py       inter-annotator agreement between two annotation files
│   ├── 11_human_annotation.py optional second-annotator study on the test set
│   └── 12_export_release.py  (authors) rebuild the public release
├── vitourabsa/               Python package
│   ├── common.py             aspects, provinces, label convention, I/O helpers
│   ├── metrics.py            T1 / T2 / T3, joint F1, QWK, paired bootstrap, Holm
│   ├── paths.py              folder layout (everything generated goes to work/)
│   ├── pipeline.py           runners used by the scripts (skip finished runs, log failures)
│   ├── data/
│   │   ├── release.py        read the released files, build the experiment splits
│   │   ├── preprocess.py     (authors) cleaning, deduplication, segmentation, stratified splits
│   │   └── export.py         (authors) write the public release
│   ├── models/
│   │   ├── classic.py        G0 baselines
│   │   ├── encoders.py       G1 encoders: multi-head, review-aspect pair, CORAL, weighted CE, logit adjustment
│   │   ├── seq2seq.py        G2 sequence-to-sequence
│   │   ├── formats.py        prompts, output formats, few-shot selection
│   │   ├── llm_infer.py      G3 LLM inference (subprocess)
│   │   └── llm_qlora.py      G3 QLoRA training (subprocess)
│   └── analysis/
│       ├── eda.py            dataset statistics and figures
│       ├── report.py         result aggregation, significance, result tables and figures
│       ├── extra.py          additional analyses of the paper
│       ├── iaa.py            agreement between two annotations
│       └── human.py          second-annotator sheet and evaluation
└── work/                     created by the scripts (git-ignored): splits, results, checkpoints, reports
```

## Installation

```bash
conda create -n vitourabsa python=3.11 -y && conda activate vitourabsa
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

The LLM experiments download models from Hugging Face. Llama-3.1-8B-Instruct and Vistral-7B-Chat are gated: accept
their licences on the model pages and set a token before running.

```bash
export HF_TOKEN=hf_xxx          # never write the token into the code
```

## Quick start

All commands are run from the repository root.

```bash
python scripts/00_prepare_splits.py      # 20 s: builds work/splits/S1_main ... S6_with_duplicates
python scripts/01_eda.py                 # 30 s: tables and figures in work/reports/eda/
python scripts/02_run_classic.py         # a few minutes: G0 baselines on S1
python scripts/03_run_encoder.py --model phobert_base --seeds 1
python scripts/08_make_report.py         # tables in work/reports/results/
```

The splits built by `00_prepare_splits.py` contain exactly the reviews, labels and order used in the paper.

## Reproducing the full benchmark

```bash
nohup python scripts/07_run_benchmark.py > work/benchmark.log 2>&1 &
python scripts/08_make_report.py
python scripts/09_extra_analysis.py
```

`07_run_benchmark.py` runs these steps in order (`--steps` selects a subset):

| Step | Content |
|---|---|
| `classic` | G0 baselines on S1 |
| `encoders` | six encoders on S1; learning rate chosen on the validation set with seed 1, then remaining seeds |
| `llm` | five LLMs, zero-shot and 10-shot, on S1 |
| `variants` | PhoBERT-base pair / CORAL / weighted CE / logit adjustment, XLM-R-large pair |
| `seq2seq` | ViT5-base |
| `qlora` | QLoRA adapters for Qwen2.5-7B and Vistral-7B, then evaluation |
| `shift` | S2 cross-region, S4 hotel to resort and resort to hotel, S3 leave one province out |
| `learning_curve` | S5 (1 to 50% of the S1 training set) |
| `leakage` | S6 (data with duplicates) |

Every run writes `work/results/<split>/<run>/seed<k>/metrics.json` together with the predictions. Finished runs
are skipped, so the command can be started again after an interruption. Failed runs are written to
`work/reports/failures.log` and the next run starts. On one RTX 4090, the complete benchmark takes several days.
To change what is run, edit `configs/benchmark.yaml` or pass another file with `--config`.

## Running single experiments

```bash
python scripts/02_run_classic.py --split S2_out_of_region --which tfidf_lr
python scripts/03_run_encoder.py --model xlmr_large                      # S1, 3 seeds, lr search
python scripts/03_run_encoder.py --variant phobert_base_pair
python scripts/03_run_encoder.py --model phobert_base --split S3_lopo_quangninh --seeds 1
python scripts/04_run_seq2seq.py --model vit5_base
python scripts/05_run_llm.py --model llama31_8b --mode few
python scripts/05_run_llm.py --model qwen25_7b --mode zero --limit 50    # quick check on 50 reviews
python scripts/06_train_qlora.py --model vistral_7b
```

LLM inference uses vLLM when it is installed and Hugging Face `generate` otherwise (the paper used the latter).

## Evaluation

All subtasks are derived from the same five-level predictions.

* **T1** polar-aspect detection (score different from 0): F1 over all slots.
* **T2** intensity on gold polar slots, classes {-2, -1, +1, +2}; predicting 0 counts as an error: macro-F1.
* **T3** end-to-end five-level prediction: macro-F1 over the levels present in the gold labels of each aspect,
  averaged over aspects. This is the primary metric.
* Joint F1 over (aspect, score) pairs, quadratic weighted kappa, exact match.

```python
import numpy as np
from vitourabsa.metrics import evaluate
y = np.array([[2, 0, 0, -1, 0, 0, 0, 0, 0, 1]])        # gold, shape (n_reviews, 10), values in {-2..2}
p = np.array([[2, 0, 0, -2, 0, 0, 0, 0, 0, 1]])        # predictions
print(evaluate(y, p, per_aspect=False)['t3_macro_f1'])
```

## Inter-annotator agreement

```bash
python scripts/10_agreement.py --a iaa_annotator_A_tourism.csv --b iaa_annotator_B_tourism.csv
```

## Outputs

| Folder | Content |
|---|---|
| `work/splits/` | experiment-ready splits (`train/val/test.jsonl` per split) |
| `work/results/` | metrics, predictions and training logs of every run |
| `work/ckpt/` | QLoRA adapters |
| `work/reports/eda/` | T01 to T13, F01 to F06 (CSV, LaTeX, PDF, PNG) |
| `work/reports/results/` | R01 to R13 and result figures |
| `work/reports/extra/` | X01 to X06, agreement tables |

## For the authors: rebuilding the release from the raw file

```bash
python scripts/00_prepare_splits.py --from-raw tourism_dataset.csv --force
python scripts/12_export_release.py --out work/release/ViTourABSA
```

The first command repeats the cleaning (removal of invalid rows, majority vote over exact duplicates, MinHash
near-duplicate removal, pyvi segmentation) and the stratified splitting with seed 42. It also enables the
cleaning and duplicate analyses in `01_eda.py`.
