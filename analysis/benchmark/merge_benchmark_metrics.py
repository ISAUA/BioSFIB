"""Create a downstream benchmark CSV from immutable historical and incremental results.

The script never changes either input CSV.  It refuses to merge overlapping
method/slice/condition/metric rows, preventing a newly rerun result from
silently replacing a historical result.

Example
-------
python merge_benchmark_metrics.py \
  --base benchmark_results/benchmark_metrics.csv \
  --increment benchmark_results_expanded/benchmark_metrics_expanded.csv benchmark_results_seurat_wnn/seurat_wnn_native_metrics.csv \
  --output benchmark_results_combined/benchmark_metrics_combined.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd


KEY_COLUMNS = ["evaluation_id", "evaluation_role", "dataset", "slice", "method", "metric_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge historical and incremental benchmark metric CSV files.")
    parser.add_argument("--base", default="results/benchmark_results_expanded/benchmark_metrics_expanded.csv", help="Base benchmark CSV; read only.")
    parser.add_argument("--increment", nargs="+", default=["results/benchmark_results_seurat_wnn/seurat_wnn_native_metrics.csv"], help="One or more disjoint benchmark CSV files; read only.")
    parser.add_argument("--output", default="results/benchmark_results_combined/benchmark_metrics_combined.csv", help="New downstream combined CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing combined CSV; never changes either input CSV.")
    return parser.parse_args()


def read_metrics(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing merge key columns: {missing}")
    duplicated = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        preview = frame.loc[duplicated, KEY_COLUMNS].head(5).to_dict(orient="records")
        raise ValueError(f"{label} CSV contains duplicate metric rows; fix it before merging. Examples: {preview}")
    return frame


def main() -> int:
    args = parse_args()
    base_path = Path(args.base)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Combined output already exists: {output_path}. Use --overwrite to replace it.")

    base = read_metrics(base_path, "Base")
    increments = [read_metrics(Path(path), f"Increment {index}") for index, path in enumerate(args.increment, start=1)]
    increment = pd.concat(increments, ignore_index=True)
    duplicated_increment = increment.duplicated(KEY_COLUMNS, keep=False)
    if duplicated_increment.any():
        preview = increment.loc[duplicated_increment, KEY_COLUMNS].head(5).to_dict(orient="records")
        raise ValueError(f"Increment CSV files overlap on benchmark metric rows. Examples: {preview}")
    overlap = base.merge(increment[KEY_COLUMNS], on=KEY_COLUMNS, how="inner")
    if not overlap.empty:
        preview = overlap[KEY_COLUMNS].head(5).to_dict(orient="records")
        raise ValueError(
            "Base and increment CSVs overlap on benchmark metric rows. "
            f"Do not silently replace historical results. Examples: {preview}"
        )

    all_columns = list(base.columns) + [column for column in increment.columns if column not in base.columns]
    combined = pd.concat([base.reindex(columns=all_columns), increment.reindex(columns=all_columns)], ignore_index=True)
    combined = combined.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Base rows: {len(base)}")
    print(f"Increment rows: {len(increment)}")
    print(f"Combined rows: {len(combined)}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Merge failed: {type(error).__name__}: {error}")
        raise SystemExit(1)
