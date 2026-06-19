#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob(f"{args.prefix}*.csv"))

    if not files:
        raise RuntimeError(f"No files matching {args.prefix}*.csv in {results_dir}")

    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    xlsx_output = output.with_suffix(".xlsx")
    merged.to_excel(xlsx_output, index=False)

    print(f"Merged {len(files)} files")
    print(f"Saved CSV: {output}")
    print(f"Saved Excel: {xlsx_output}")


if __name__ == "__main__":
    main()
