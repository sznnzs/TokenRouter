#!/usr/bin/env python
"""Wrapper around the packaged experiment CLI."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenrouter.cli import run_experiments_main


if __name__ == "__main__":
    run_experiments_main()
