"""Generate result tables and figures from completed final evaluations."""
from __future__ import annotations

import argparse
from pathlib import Path

from .report import report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="regbench-report")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--ece-bins", type=int, default=10)
    args = parser.parse_args(argv)
    report(Path(args.out), args.ece_bins)


if __name__ == "__main__":
    main()
