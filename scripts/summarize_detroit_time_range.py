import duckdb

path = r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet"
path_sql = path.replace("\\", "/")

con = duckdb.connect()
sql = f"""
SELECT
  min("DATE_RANGE_START") AS week_start_min,
  max("DATE_RANGE_START") AS week_start_max,
  count(DISTINCT "DATE_RANGE_START") AS n_distinct_weeks,
  count(*) AS n_rows
FROM read_parquet('{path_sql}')
"""
print(con.execute(sql).fetchall())

con.close()
