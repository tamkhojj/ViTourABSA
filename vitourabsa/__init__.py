"""ViTourABSA: data loading, models, evaluation and analysis code for the paper
"A large-scale multi-regional dataset and benchmark for Vietnamese aspect-based sentiment analysis in tourism".

Sub-packages
------------
vitourabsa.data      read the released dataset, rebuild splits from the raw file, export a release
vitourabsa.models    TF-IDF baselines, fine-tuned encoders, seq2seq, LLM prompting and QLoRA
vitourabsa.analysis  EDA, result tables and figures, agreement studies, additional analyses
vitourabsa.metrics   T1/T2/T3 metrics, paired bootstrap, Holm correction
"""
__version__ = '1.0.0'
