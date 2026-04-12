"""
Filter local Dewey Weekly Patterns (or similar) rows where CITY is Detroit.

Adjust INPUT_GLOB / column names if your Data Dictionary differs.
"""

from pathlib import Path

import duckdb

# Folder where your downloaded *.csv.gz or *.parquet live
INPUT_GLOB = r"E:\Urban Computing Final Project\dewey-downloads\jenny-try-0412\*.csv.gz"

# Output: single Parquet file (change if you prefer a folder)
OUTPUT_PARQUET = r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet"


def main() -> None:
    out = Path(OUTPUT_PARQUET)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Input glob:", INPUT_GLOB, flush=True)
    print("Output file:", out, flush=True)
    print(
        "Running DuckDB (scan + filter + write). Large folders may take many minutes; "
        "this window will look idle until it finishes.",
        flush=True,
    )

    con = duckdb.connect(database=":memory:")

    # If your files are Parquet instead of gzip CSV, swap the FROM clause below to:
    #   FROM read_parquet($pattern, union_by_name=true)
    # and set INPUT_GLOB to .../*.parquet
    query = f"""
    COPY (
      SELECT
        *
      FROM read_csv_auto($pattern, union_by_name=true, header=true)
      WHERE
        upper(trim("CITY")) = 'DETROIT'
    )
    TO '{out.as_posix()}'
    (FORMAT PARQUET,
     COMPRESSION ZSTD);
    """

    con.execute(query, {"pattern": INPUT_GLOB})
    con.close()

    if out.exists():
        mb = out.stat().st_size / (1024 * 1024)
        print(f"Done. Wrote: {out} ({mb:.2f} MB)", flush=True)
    else:
        print("Finished, but output file was not found:", out, flush=True)


if __name__ == "__main__":
    main()
