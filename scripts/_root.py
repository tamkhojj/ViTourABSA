"""Makes `import vitourabsa` work when a script is run as `python scripts/<name>.py` from any folder."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
