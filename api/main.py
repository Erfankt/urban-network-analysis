"""
api/main.py
Urban Street Network & Crime — REST API
----------------------------------------
Endpoints
  GET  /                           health check + API info
  GET  /npa                        list all 371 NPAs (summary)
  GET  /npa/{npa_id}               full feature row for one NPA
  GET  /npa/{npa_id}/network       network measures (long format) for one NPA
  GET  /metrics/top                top-N NPAs by a given network metric + buffer
  GET  /spatial/near               NPAs whose centroid is within radius of a point

Run locally:
  uvicorn api.main:app --reload --port 8000
Run in Docker (from /app):
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import json
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db
from schemas import (
    MetricTopRow,
    NPADetail,
    NPANetwork,
    NPASummary,
    NetworkMeasureRow,
    SocioeconomicAttrs,
    SpatialNearRow,
)

app = FastAPI(
    title="Urban Street Network & Crime API",
    description=(
        "Query the Charlotte / Mecklenburg County NPA dataset. "
        "Serves PostGIS-backed spatial and tabular data derived from "
        "peer-reviewed research (Kefayat & Thill, 2025)."
    ),
    version="1.0.0",
)

# ── helpers ───────────────────────────────────────────────────────────────────

VALID_METRICS = {
    "int_den", "st_den", "nd_deg", "cl_cof",
    "shtpth", "bt_cnt", "cl_cnt", "pr_cnt",
}
VALID_BUFFERS = {0.10, 0.25, 0.50, 0.75, 1.00}


def _row_to_dict(row) -> dict:
    return dict(row._mapping)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Urban Street Network & Crime API",
        "version": "1.0.0",
        "docs": "/docs",
        "npas": 371,
        "crs": "EPSG:2264 (NC State Plane, feet)",
        "reference": "Kefayat & Thill (2025), ISPRS Int. J. Geo-Information 14(5), 200",
    }


@app.get("/npa", response_model=List[NPASummary], tags=["npa"])
def list_npas(db: Session = Depends(get_db)):
    """Return summary info for all 371 NPAs, ordered by property-crime rate."""
    rows = db.execute(text("""
        SELECT n.npa_id, n.area_acre,
               c.propcrim_ct, c.propcrim_1k, c.propcrim_acre
        FROM   npa n
        LEFT JOIN npa_crime c ON c.npa_id = n.npa_id
        ORDER  BY c.propcrim_1k DESC NULLS LAST
    """)).fetchall()
    return [_row_to_dict(r) for r in rows]


@app.get("/npa/{npa_id}", response_model=NPADetail, tags=["npa"])
def get_npa(npa_id: str, db: Session = Depends(get_db)):
    """Return the full feature row for one NPA, including geometry as GeoJSON."""
    row = db.execute(text("""
        SELECT
            n.npa_id, n.area_acre,
            ST_AsGeoJSON(n.geom)::json  AS geometry,
            c.propcrim_ct, c.propcrim_1k, c.propcrim_acre,
            s.bach_deg, s.employment, s.home_ownership, s.res_occupancy,
            s.singlefam_rate, s.singlefam_age, s.housing_density, s.singlefam_size,
            s.median_sale_price, s.median_age, s.median_income,
            s.race_other, s.race_asian, s.race_black, s.race_latin, s.race_white,
            s.heated_area
        FROM  npa n
        LEFT JOIN npa_crime         c ON c.npa_id = n.npa_id
        LEFT JOIN npa_socioeconomic s ON s.npa_id = n.npa_id
        WHERE n.npa_id = :npa_id
    """), {"npa_id": npa_id}).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"NPA '{npa_id}' not found")

    d = _row_to_dict(row)
    socio_keys = [
        "bach_deg", "employment", "home_ownership", "res_occupancy",
        "singlefam_rate", "singlefam_age", "housing_density", "singlefam_size",
        "median_sale_price", "median_age", "median_income",
        "race_other", "race_asian", "race_black", "race_latin", "race_white",
        "heated_area",
    ]
    return NPADetail(
        npa_id=d["npa_id"],
        area_acre=d["area_acre"],
        propcrim_ct=d["propcrim_ct"],
        propcrim_1k=d["propcrim_1k"],
        propcrim_acre=d["propcrim_acre"],
        geometry=d["geometry"],
        socioeconomic=SocioeconomicAttrs(**{k: d[k] for k in socio_keys}),
    )


@app.get("/npa/{npa_id}/network", response_model=NPANetwork, tags=["npa"])
def get_npa_network(
    npa_id: str,
    metric: Optional[str] = Query(None, description="Filter by metric name"),
    buffer_mi: Optional[float] = Query(None, description="Filter by buffer distance (mi)"),
    db: Session = Depends(get_db),
):
    """Return network measures for one NPA, optionally filtered by metric and/or buffer."""
    # verify NPA exists
    exists = db.execute(
        text("SELECT 1 FROM npa WHERE npa_id = :id"), {"id": npa_id}
    ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail=f"NPA '{npa_id}' not found")

    if metric and metric not in VALID_METRICS:
        raise HTTPException(status_code=422, detail=f"Unknown metric '{metric}'. "
                            f"Valid: {sorted(VALID_METRICS)}")
    if buffer_mi and buffer_mi not in VALID_BUFFERS:
        raise HTTPException(status_code=422, detail=f"Unknown buffer '{buffer_mi}'. "
                            f"Valid: {sorted(VALID_BUFFERS)}")

    clauses = ["npa_id = :npa_id"]
    params: dict = {"npa_id": npa_id}
    if metric:
        clauses.append("metric = :metric")
        params["metric"] = metric
    if buffer_mi:
        clauses.append("buffer_mi = :buffer_mi")
        params["buffer_mi"] = buffer_mi

    rows = db.execute(text(
        f"SELECT metric, buffer_mi, value FROM npa_network_measures "
        f"WHERE {' AND '.join(clauses)} ORDER BY metric, buffer_mi"
    ), params).fetchall()

    return NPANetwork(
        npa_id=npa_id,
        measures=[NetworkMeasureRow(**_row_to_dict(r)) for r in rows],
    )


@app.get("/metrics/top", response_model=List[MetricTopRow], tags=["metrics"])
def top_npas(
    metric: str = Query(..., description="Network metric name"),
    buffer_mi: float = Query(0.25, description="Buffer distance in miles"),
    n: int = Query(10, ge=1, le=371, description="Number of results"),
    db: Session = Depends(get_db),
):
    """Return the top-N NPAs ranked by a chosen network metric at a given buffer distance."""
    if metric not in VALID_METRICS:
        raise HTTPException(status_code=422, detail=f"Unknown metric '{metric}'. "
                            f"Valid: {sorted(VALID_METRICS)}")
    if buffer_mi not in VALID_BUFFERS:
        raise HTTPException(status_code=422, detail=f"Unknown buffer. "
                            f"Valid: {sorted(VALID_BUFFERS)}")

    rows = db.execute(text("""
        SELECT m.npa_id, m.value, c.propcrim_1k
        FROM   npa_network_measures m
        LEFT JOIN npa_crime c ON c.npa_id = m.npa_id
        WHERE  m.metric    = :metric
          AND  m.buffer_mi = :buffer_mi
        ORDER  BY m.value DESC NULLS LAST
        LIMIT  :n
    """), {"metric": metric, "buffer_mi": buffer_mi, "n": n}).fetchall()

    return [_row_to_dict(r) for r in rows]


@app.get("/spatial/near", response_model=List[SpatialNearRow], tags=["spatial"])
def near_point(
    lon: float = Query(..., description="Longitude (WGS84)"),
    lat: float = Query(..., description="Latitude (WGS84)"),
    radius_mi: float = Query(1.0, description="Search radius in miles"),
    db: Session = Depends(get_db),
):
    """
    Return NPAs whose centroid falls within radius_mi of the given WGS84 point.
    Internally reprojects to EPSG:2264 (feet); 1 mile = 5280 ft.
    """
    radius_ft = radius_mi * 5280
    rows = db.execute(text("""
        SELECT
            n.npa_id,
            ST_Distance(
                ST_Centroid(n.geom),
                ST_Transform(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 2264)
            ) AS dist_ft,
            c.propcrim_1k
        FROM  npa n
        LEFT JOIN npa_crime c ON c.npa_id = n.npa_id
        WHERE ST_DWithin(
            ST_Centroid(n.geom),
            ST_Transform(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 2264),
            :radius_ft
        )
        ORDER BY dist_ft
    """), {"lon": lon, "lat": lat, "radius_ft": radius_ft}).fetchall()

    return [_row_to_dict(r) for r in rows]
