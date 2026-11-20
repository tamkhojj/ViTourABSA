"""Collect all results and write the result tables (CSV + LaTeX) and figures (PDF + PNG) to work/reports/results/.

R01 main results, R02 significance (paired bootstrap, Holm), R03 per aspect, F1 heatmap and confusion matrix,
R04 error types, R05 breakdown by review characteristics, R06 cross-region and cross-facility, R07 leave one
province out, R08 learning curve, R09 duplicate leakage, R10 LLMs, R11 computational cost, R13 error sample.

    python scripts/08_make_report.py
"""
import _root  # noqa: F401
import pandas as pd

from vitourabsa import metrics, outputs, paths
from vitourabsa.analysis import human, report
from vitourabsa.pipeline import LLM_DISPLAY, display_names, log


def show(title, df):
    if df is None or (hasattr(df, 'empty') and df.empty):
        return
    print(f'\n### {title}')
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        print(df.round(4) if hasattr(df, 'round') else df)


def main():
    outputs.configure()
    META = display_names()
    DF = report.collect(paths.RESULTS)
    if DF.empty:
        raise SystemExit('no results in work/results yet')
    s1_dir, s6_dir = paths.split_dir('S1_main'), paths.split_dir('S6_with_duplicates')
    HUMAN = human.evaluate_human(s1_dir)

    show('Main results (S1 test)', report.main_table(DF, META, human=HUMAN['metrics'] if HUMAN else None))
    best = report.best_run(DF)
    log(f'best system by T3 macro-F1: {META.get(best, ("", best))[1]}')
    show('Significance vs best system (paired bootstrap, Holm)', report.significance(DF))

    s1 = DF[DF.split == 'S1_main'].groupby('run')[metrics.PRIMARY].mean().sort_values(ascending=False)
    top = [r for r in s1.index if not r.startswith('classic_')][:4]
    top += [r for r in ['classic_tfidf_lr'] if r in s1.index]
    top = list(dict.fromkeys(top + [r for r in s1.index if r.startswith('llm_')][:1]))
    show('Per aspect (T3 macro-F1)', report.per_aspect_table(DF, top, META))
    report.f1_heatmap(DF, best, META)
    report.confusion(DF, best, META)
    show('Error types (% of slots)', report.error_types(DF, top, META))
    t, _ = report.breakdown(DF, top, META, s1_dir)
    show('Breakdown by review characteristics', t)
    show('LLMs', report.llm_table(DF, LLM_DISPLAY))

    gen = [r for r in ['classic_tfidf_lr', 'phobert_base', 'xlmr_large', 'llm_qwen25_7b_zero'] if r in set(DF.run)]
    show('Cross-region and cross-facility (S2, S4)', report.generalization(DF, gen, META))
    show('Leave one province out (S3)', report.lopo(DF, gen, META, s1_dir))
    t, _ = report.learning_curve(DF, ['classic_tfidf_lr', 'phobert_base'], META)
    show('Learning curve (S5)', t)
    show('Duplicate leakage (S6)', report.leakage(DF, ['classic_tfidf_lr', 'phobert_base'], META, s6_dir))
    show('Computational cost', report.efficiency(DF, META))
    if HUMAN:
        show('Second annotator vs models on the re-annotated subset',
             report.subset_comparison(DF, top, META, HUMAN['ids'], HUMAN['metrics'], s1_dir))
    report.error_sample(DF, best, s1_dir)
    log(f'saved to {paths.RESULTS_OUT}')


if __name__ == '__main__':
    main()
