"""Print column names and first rows of detroit_filtered.parquet (small read)."""
import duckdb

path = r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet"
con = duckdb.connect()
rel = con.sql(f"SELECT * FROM read_parquet('{path.replace(chr(92), '/')}') LIMIT 5")
print("Columns:", rel.columns)
for row in rel.fetchall():
    print(row)
con.close()
