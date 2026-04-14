"""

Aggregate Detroit POI-week Parquet to CBG x week (long table).



Use for neighborhood-scale modeling, maps (GEOID + TIGER), and inflow/outflow next steps.



Usage:

  python scripts/aggregate_cbg_weekly.py

  python scripts/aggregate_cbg_weekly.py --date-start 2025-01-01 --date-end 2025-12-31

"""



from __future__ import annotations



import argparse

from pathlib import Path



import duckdb





def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--input",

        default=r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet",

        help="detroit_filtered.parquet (POI x week).",

    )

    parser.add_argument(

        "--output",

        default=r"E:\Urban Computing Final Project\Try_0412\data\cbg_weekly_2025_sample.parquet",

        help="Output Parquet (long: cbg, week_start, visits, visitors).",

    )

    parser.add_argument("--date-start", default="2025-01-01")

    parser.add_argument("--date-end", default="2025-12-31")

    args = parser.parse_args()



    inp = Path(args.input)

    out = Path(args.output)

    out.parent.mkdir(parents=True, exist_ok=True)

    path_sql = str(inp).replace("\\", "/")

    out_sql = str(out).replace("\\", "/")

    d0, d1 = args.date_start, args.date_end



    duckdb.sql(

        f"""

        COPY (

          SELECT

            trim(cast("POI_CBG" AS VARCHAR)) AS cbg,

            cast(date_trunc('day', "DATE_RANGE_START") AS DATE) AS week_start,

            sum(coalesce("VISIT_COUNTS", 0))::DOUBLE AS visits,

            sum(coalesce("VISITOR_COUNTS", 0))::DOUBLE AS visitors

          FROM read_parquet('{path_sql}')

          WHERE "POI_CBG" IS NOT NULL

            AND trim(cast("POI_CBG" AS VARCHAR)) != ''

            AND cast(date_trunc('day', "DATE_RANGE_START") AS DATE) >= DATE '{d0}'

            AND cast(date_trunc('day', "DATE_RANGE_START") AS DATE) <= DATE '{d1}'

          GROUP BY 1, 2

          ORDER BY 2, 1

        )

        TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD);

        """

    )

    print(f"Wrote {out}  filter=[{d0}, {d1}]")





if __name__ == "__main__":

    main()


