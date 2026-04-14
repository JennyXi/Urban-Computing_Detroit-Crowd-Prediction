import duckdb

p = r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet".replace("\\", "/")

cols = duckdb.sql(f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{p}'))").fetchall()
hourish = [c[0] for c in cols if "hour" in c[0].lower() or "visitor" in c[0].lower()]
print("Columns with hour/visitor in name:", hourish)

rows = duckdb.sql(
    f"""
    SELECT "VISITS_BY_EACH_HOUR"
    FROM read_parquet('{p}')
    WHERE "VISITS_BY_EACH_HOUR" IS NOT NULL AND trim(cast("VISITS_BY_EACH_HOUR" AS VARCHAR)) != ''
    LIMIT 3
    """
).fetchall()

print("\n--- VISITS_BY_EACH_HOUR: first 3 non-empty samples ---")
for i, (s,) in enumerate(rows, 1):
    t = str(s)
    print(f"\n[Row {i}] length={len(t)} chars")
    print(t if len(t) <= 800 else t[:800] + "...")
