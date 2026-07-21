"""Pytest bootstrap: ensure /python is on sys.path so `from config.xxx import ...` works
whether pytest is invoked from /python or from the repo root.
"""
import sys
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
