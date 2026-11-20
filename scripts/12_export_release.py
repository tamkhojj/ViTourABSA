"""Export work/splits as the public release (data/, splits/, stats/, README.md, SHA256SUMS.txt) plus a zip.

Used by the authors to create the release from splits rebuilt with `00_prepare_splits.py --from-raw`.

    python scripts/12_export_release.py --out work/release/ViTourABSA
"""
import argparse
import os

import _root  # noqa: F401

from vitourabsa import paths
from vitourabsa.data.export import export_release
from vitourabsa.pipeline import log

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(paths.WORK, 'release', 'ViTourABSA'))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if out == os.path.abspath(paths.ROOT):
        raise SystemExit('refusing to overwrite the repository itself; choose another --out')
    print(export_release(paths.SPLITS, out, prepared=paths.PREPARED, log=log))
