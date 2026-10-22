"""
db/load_data.py
---------------
ETL script: reads the project shapefiles and loads them into the
PostGIS database defined in docker-compose.yml.

Usage (from repo root, with the Docker container running):
    python db/load_data.py

Dependencies:
    geopandas, sqlalchemy, psycopg2-binary, GeoAlchemy2

Connection (matches docker-compose.yml defaults):
    Override with env vars: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text

# ── Connection ────────────────────────────────────────────────────────────────

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "urban_crime")
DB_USER = os.getenv("DB_USER", "erfan")
DB_PASS = os.getenv("DB_PASS", "localdev")

CONN_STR = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
NPA_SHP   = REPO_ROOT / "data" / "neighborhoods"  / "prpcrmarea_Ctrl.shp"
FINAL_SHP = REPO_ROOT / "data" / "final_dataset"  / "extracted_measures.shp"

# Buffer distances encoded in column-name suffixes → miles
BUFFER_MAP = {"1": 0.10, "025": 0.25, "05": 0.50, "075": 0.75, "10": 1.00}

# The 8 network metrics (prefix part of the shapefile column name)
METRICS = ["int_den", "st_den", "nd_deg", "cl_cof",
           "shtpth", "bt_cnt", "cl_cnt", "pr_cnt"]


# ── DB connection ─────────────────────────────────────────────────────────────

def get_engine():
    engine = create_engine(CONN_STR, future=True)
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        print(f"✓ Connected to {DB_NAME} on {DB_HOST}:{DB_PORT}")
    except Exception as e:
        print(f"✗ Cannot connect: {e}")
        sys.exit(1)
    return engine


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_npa(engine, gdf: gpd.GeoDataFrame) -> None:
    """Insert geometry + area into the npa table."""
    rows = gdf[["id", "area", "area_acre", "geometry"]].copy()
    rows = rows.rename(columns={"id": "npa_id", "area": "area_sqft"})
    # to_postgis writes the active geometry column; rename it to match schema
    rows = rows.rename_geometry("geom")
    rows.to_postgis(
        "npa", engine, if_exists="append", index=False,
    )
    print(f"  ✓ npa: {len(rows)} rows")


def load_crime(engine, gdf: gpd.GeoDataFrame) -> None:
    df = gdf[["id", "propcrimct", "propcrim1k", "prpcrmarea"]].copy()
    df = df.rename(columns={
        "id":         "npa_id",
        "propcrimct": "propcrim_ct",
        "propcrim1k": "propcrim_1k",
        "prpcrmarea": "propcrim_acre",
    })
    df.to_sql("npa_crime", engine, if_exists="append", index=False)
    print(f"  ✓ npa_crime: {len(df)} rows")


def load_socioeconomic(engine, gdf: gpd.GeoDataFrame) -> None:
    rename = {
        "id":          "npa_id",
        "bachdg":      "bach_deg",
        "emp":         "employment",
        "home_own":    "home_ownership",
        "resoccup":    "res_occupancy",
        "sinfamrate":  "singlefam_rate",
        "sinfamage":   "singlefam_age",
        "hs_den":      "housing_density",
        "sinfamsize":  "singlefam_size",
        "salesprice":  "median_sale_price",
        "med_age":     "median_age",
        "income":      "median_income",
        "race_other":  "race_other",
        "race_asian":  "race_asian",
        "race_black":  "race_black",
        "race_latin":  "race_latin",
        "race_white":  "race_white",
        "heatedarea":  "heated_area",
    }
    df = gdf[list(rename.keys())].copy().rename(columns=rename)
    df.to_sql("npa_socioeconomic", engine, if_exists="append", index=False)
    print(f"  ✓ npa_socioeconomic: {len(df)} rows")


def load_network_measures(engine, gdf: gpd.GeoDataFrame) -> None:
    """
    Melt the 40 wide columns (metric × buffer suffix) into long format:
        (npa_id, buffer_mi, metric, value)
    """
    records = []
    for _, row in gdf.iterrows():
        npa_id = row["id"]
        for metric in METRICS:
            for suffix, buf_mi in BUFFER_MAP.items():
                col = f"{metric}{suffix}"
                if col in gdf.columns:
                    records.append({
                        "npa_id":    npa_id,
                        "buffer_mi": buf_mi,
                        "metric":    metric,
                        "value":     row[col],
                    })

    df = pd.DataFrame(records)
    df.to_sql("npa_network_measures", engine, if_exists="append", index=False)
    n_metrics = len(METRICS) * len(BUFFER_MAP)
    print(f"  ✓ npa_network_measures: {len(df)} rows  "
          f"({n_metrics} measures × {len(gdf)} NPAs)")


# ── Verification ──────────────────────────────────────────────────────────────

def verify(engine) -> None:
    tables = ["npa", "npa_crime", "npa_socioeconomic", "npa_network_measures"]
    print()
    with engine.connect() as c:
        for t in tables:
            n = c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            status = "✓" if n > 0 else "✗"
            print(f"  {status} {t}: {n} rows")
        n = c.execute(text("SELECT COUNT(*) FROM v_feature_matrix")).scalar()
        status = "✓" if n > 0 else "✗"
        print(f"  {status} v_feature_matrix: {n} rows")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading shapefiles …")
    npa_gdf   = gpd.read_file(NPA_SHP)
    final_gdf = gpd.read_file(FINAL_SHP)
    print(f"  NPA layer:     {len(npa_gdf)} features, "
          f"{len(npa_gdf.columns)} columns")
    print(f"  Final dataset: {len(final_gdf)} features, "
          f"{len(final_gdf.columns)} columns")

    engine = get_engine()

    print("\nInserting …")
    load_npa(engine, npa_gdf)
    load_crime(engine, npa_gdf)
    load_socioeconomic(engine, npa_gdf)
    load_network_measures(engine, final_gdf)

    print("\nVerifying row counts …")
    verify(engine)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
