"""
pytest configuration of the pysolve-gn test suite.

- The root of the repository is added to ``sys.path`` so that ``pysolvegn`` can be
  imported without installing the package.
- The ``tests`` folder is added to ``sys.path`` so that the shared module
  ``problems.py`` can be imported whatever the pytest import mode (``tests/__init__.py``
  present or ``--import-mode=importlib``).
- matplotlib uses a non-interactive backend (no window is opened by ``plt.show()``).
"""

import os
import sys

TESTS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(TESTS, ".."))
for path in (ROOT, TESTS):  # pysolvegn and the shared module problems.py
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib

matplotlib.use("Agg")