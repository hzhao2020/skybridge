#!/usr/bin/env python3
"""Populate sim/data/synthetic endpoints with measurement-backed latencies."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.measurement.populate import DEFAULT_SEED, populate  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Fill data/synthetic/endpoints.csv with latencies derived from "
            "data/measurement execution-time CSVs."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR,
        help=f"Target synthetic data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic node scaling (default: 42)",
    )
    parser.add_argument(
        "--regenerate-all",
        action="store_true",
        help="Regenerate all synthetic CSVs before applying measurement latencies",
    )
    args = parser.parse_args()
    populate(
        args.output,
        seed=args.seed,
        regenerate_all=args.regenerate_all,
    )
    print(f"Updated measurement-backed latencies in {args.output / 'endpoints.csv'}")


if __name__ == "__main__":
    main()
