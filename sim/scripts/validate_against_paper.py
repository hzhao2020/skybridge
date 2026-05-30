#!/usr/bin/env python3
"""Run paper-aligned validation checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validation import known_implementation_gaps, run_all_validations  # noqa: E402


def main() -> None:
    report = run_all_validations()
    print(report.summary())
    print()
    print("Known prototype gaps (paper vs current data):")
    for i, gap in enumerate(known_implementation_gaps(), 1):
        print(f"  {i}. {gap}")
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
