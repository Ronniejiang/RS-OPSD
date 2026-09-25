"""Small pytest compatibility fallback used only by PA-OPD preflight.

The target vLLM environment intentionally has no pytest dependency.  When a
real pytest installation exists outside this source tree, this module delegates
to it.  Otherwise ``python -m pytest`` can run the repository's simple
zero-argument smoke tests without installing packages.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SEARCH_PATHS = []
for _path in sys.path:
    try:
        if Path(_path or ".").resolve() != _ROOT:
            _SEARCH_PATHS.append(_path)
    except OSError:
        _SEARCH_PATHS.append(_path)

_real_spec = importlib.machinery.PathFinder.find_spec("pytest", _SEARCH_PATHS)
if _real_spec is not None and _real_spec.loader is not None:
    _real_pytest = importlib.util.module_from_spec(_real_spec)
    sys.modules[__name__] = _real_pytest
    _real_spec.loader.exec_module(_real_pytest)
else:
    __all__ = []
