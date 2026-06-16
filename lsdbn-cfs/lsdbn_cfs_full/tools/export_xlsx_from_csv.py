from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default="../outputs/final_paper_results/tables")
    args = parser.parse_args()
    tables_dir = Path(args.tables_dir)
    for csv_path in sorted(tables_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        out = csv_path.with_suffix(".xlsx")
        df.to_excel(out, index=False)
        print("wrote", out)


if __name__ == "__main__":
    main()

