"""Point the analysis modules to the work/reports folders (call once at the start of a script)."""
from . import paths


def configure():
    from .analysis import eda, human, report
    paths.ensure()
    eda.OUT = paths.EDA_OUT
    report.OUT = paths.RESULTS_OUT
    human.set_dir(paths.HUMAN)
