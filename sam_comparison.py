#!/usr/bin/env python3
"""Compatibility wrapper for the SA-V benchmark CLI."""

from __future__ import annotations

import sys

from sav_benchmark.main import main, run_simple_test


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_simple_test()
    else:
        main()
