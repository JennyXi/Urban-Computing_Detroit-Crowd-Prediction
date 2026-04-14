"""
Filter local Dewey Weekly Patterns (or similar) rows where CITY is Detroit.

Usage (CMD, with .venv activated and duckdb installed):
  python scripts/filter_detroit_duckdb.py --input "D:\\data\\michigan\\*.csv.gz" --output data\\detroit.parquet
  python scripts/filter_detroit_duckdb.py --input "D:\\data\\michigan\\*.parquet" --format parquet

Defaults below match the original Try_0412 paths; override with --input/--output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

DEFAULT_INPUT_GLOB = r"E:\Urban Computing Final Project\dewey-downloads\jenny-try-0412\*.csv.gz"
DEFAULT_OUTPUT = r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter rows to CITY=Detroit with DuckDB.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_GLOB,
        help="Glob of local files, e.g. D:\\\\michigan\\\\*.csv.gz or ...\\\\*.parquet",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--format",
        choices=("csv_gz", "parquet"),
        default="csv_gz",
        help="Input file type (default: gzip CSV shards).",
    )
    args = parser.parse_args()

    input_glob = args.input
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Input glob:", input_glob, flush=True)
    print("Output file:", out, flush=True)
    print(
        "Running DuckDB (scan + filter + write). Large folders may take many minutes; "
        "this window will look idle until it finishes.",
        flush=True,
    )

    con = duckdb.connect(database=":memory:")

    if args.format == "csv_gz":
        from_clause = "read_csv_auto($pattern, union_by_name=true, header=true)"
    else:
        from_clause = "read_parquet($pattern, union_by_name=true)"

    query = f"""
    COPY (
      SELECT
        *
      FROM {from_clause}
      WHERE
        upper(trim("CITY")) = 'DETROIT'
    )
    TO '{out.as_posix()}'
    (FORMAT PARQUET,
     COMPRESSION ZSTD);
    """

    con.execute(query, {"pattern": input_glob})
    con.close()

    if out.exists():
        mb = out.stat().st_size / (1024 * 1024)
        print(f"Done. Wrote: {out} ({mb:.2f} MB)", flush=True)
    else:
        print("Finished, but output file was not found:", out, flush=True)


if __name__ == "__main__":
    main()
