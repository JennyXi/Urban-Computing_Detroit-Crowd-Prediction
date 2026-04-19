"""
Render grid-week Parquet on an OpenStreetMap basemap (Folium / Leaflet).

- Polygons: reconstructs each cell footprint in lon/lat from (gx, gy) + cell size + EPSG
  (must match aggregate_grid_weekly.py).
- Heatmap (optional): fast for many cells; uses cell_lon/cell_lat + visits.

Usage:
  python scripts/visualize_grid_osm.py --input data/grid100_weekly_sample.parquet --week 2025-01-06
  python scripts/visualize_grid_osm.py --input data/grid100_weekly_sample.parquet --week 2025-01-06 --mode heatmap
  python scripts/visualize_grid_osm.py --help

Output: HTML file you open in a browser (offline-friendly map tiles may still fetch OSM).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pyproj import Transformer

try:
    import folium
    from folium.plugins import HeatMap
except ImportError as e:
    raise SystemExit("Install folium: pip install folium") from e


def cell_polygon_lonlat(
    gx: int,
    gy: int,
    cell_m: float,
    epsg: int,
) -> list[tuple[float, float]]:
    """Four corners of the cell in WGS84, order: SW, SE, NE, NW (closed ring friendly)."""
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x0, y0 = gx * cell_m, gy * cell_m
    corners_xy = [
        (x0, y0),
        (x0 + cell_m, y0),
        (x0 + cell_m, y0 + cell_m),
        (x0, y0 + cell_m),
    ]
    ring = []
    for x, y in corners_xy:
        lon, lat = to_wgs.transform(x, y)
        ring.append((lat, lon))
    ring.append(ring[0])
    return ring


def main() -> None:
    parser = argparse.ArgumentParser(description="OSM map for grid x week Parquet.")
    parser.add_argument(
        "--input",
        default=r"E:\Urban Computing Final Project\Try_0412\data\grid100_weekly_sample.parquet",
    )
    parser.add_argument(
        "--output",
        default=r"E:\Urban Computing Final Project\Try_0412\data\grid_map.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--week",
        default=None,
        help="Week start date YYYY-MM-DD (must exist in data). Default: latest week in file.",
    )
    parser.add_argument("--cell-meters", type=float, default=100.0)
    parser.add_argument("--epsg", type=int, default=32617)
    parser.add_argument(
        "--value-col",
        default="visits",
        choices=("visits", "visitors"),
        help="Color scale by this column.",
    )
    parser.add_argument(
        "--mode",
        choices=("polygons", "heatmap"),
        default="polygons",
        help="polygons: draw grid cells (slower if many). heatmap: point-weight heat (faster).",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=4000,
        help="Max cells to draw in polygon mode (0 = all; may be slow in browser).",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    if "week_start" not in df.columns:
        raise SystemExit("Expected column week_start in Parquet.")
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.normalize()

    if args.week:
        wk = pd.Timestamp(args.week).normalize()
        df = df[df["week_start"] == wk]
    else:
        wk = df["week_start"].max()
        df = df[df["week_start"] == wk]

    if df.empty:
        raise SystemExit("No rows for selected week. Pass --week or check input.")

    vcol = args.value_col
    if vcol not in df.columns:
        raise SystemExit(f"Missing column {vcol!r}")

    center_lat = float(df["cell_lat"].median())
    center_lon = float(df["cell_lon"].median())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

    vmin, vmax = float(df[vcol].min()), float(df[vcol].max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    if args.mode == "heatmap":
        pts = [
            [float(r["cell_lat"]), float(r["cell_lon"]), max(float(r[vcol]), 0.0)]
            for _, r in df.iterrows()
        ]
        HeatMap(pts, min_opacity=0.2, max_zoom=18, radius=12, blur=18).add_to(m)
        folium.LayerControl().add_to(m)
    else:
        sub = df.sort_values(vcol, ascending=False)
        nmax = args.max_cells
        if nmax > 0 and len(sub) > nmax:
            sub = sub.head(nmax)
            print(f"Note: showing top {nmax} cells by {vcol} (use --max-cells 0 for all).")
        for _, r in sub.iterrows():
            gx, gy = int(r["gx"]), int(r["gy"])
            val = float(r[vcol])
            t = (val - vmin) / (vmax - vmin)
            color = _color_ramp(t)
            poly = cell_polygon_lonlat(gx, gy, args.cell_meters, args.epsg)
            folium.Polygon(
                locations=poly,
                color=color,
                weight=0,
                fill=True,
                fill_color=color,
                fill_opacity=0.55,
                popup=f"{vcol}={val:.1f}<br>grid={r.get('grid_id', '')}",
            ).add_to(m)

    title = f"{inp.name} | week={wk.date()} | {vcol} | {args.mode}"
    m.get_root().html.add_child(folium.Element(f"<title>{title}</title>"))
    m.save(out)
    print(f"Wrote {out}  rows_week={len(df)}  week={wk.date()}  mode={args.mode}")


def _color_ramp(t: float) -> str:
    t = max(0.0, min(1.0, t))
    # blue -> yellow -> red
    if t < 0.5:
        g = int(255 * (2 * t))
        return f"#{0:02x}{g:02x}ff"
    r = int(255 * (2 * (t - 0.5)))
    return f"#{r:02x}ff00"


if __name__ == "__main__":
    main()
