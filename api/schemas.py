"""
api/schemas.py
Pydantic response models for the Urban Crime API.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class NPASummary(BaseModel):
    """Lightweight row returned by GET /npa."""
    npa_id: str
    area_acre: float
    propcrim_ct: Optional[int]
    propcrim_1k: Optional[float]
    propcrim_acre: Optional[float]

    model_config = ConfigDict(from_attributes=True)


class SocioeconomicAttrs(BaseModel):
    bach_deg: Optional[float]
    employment: Optional[float]
    home_ownership: Optional[float]
    res_occupancy: Optional[float]
    singlefam_rate: Optional[float]
    singlefam_age: Optional[float]
    housing_density: Optional[float]
    singlefam_size: Optional[float]
    median_sale_price: Optional[float]
    median_age: Optional[float]
    median_income: Optional[float]
    race_other: Optional[float]
    race_asian: Optional[float]
    race_black: Optional[float]
    race_latin: Optional[float]
    race_white: Optional[float]
    heated_area: Optional[float]


class NPADetail(BaseModel):
    """Full feature row returned by GET /npa/{npa_id}."""
    npa_id: str
    area_acre: Optional[float]
    # crime
    propcrim_ct: Optional[int]
    propcrim_1k: Optional[float]
    propcrim_acre: Optional[float]
    # socioeconomic (nested)
    socioeconomic: SocioeconomicAttrs
    # geometry as GeoJSON (from ST_AsGeoJSON)
    geometry: Optional[Dict[str, Any]]


class NetworkMeasureRow(BaseModel):
    """One (metric, buffer) observation."""
    metric: str
    buffer_mi: float
    value: Optional[float]


class NPANetwork(BaseModel):
    """All network measures for one NPA."""
    npa_id: str
    measures: List[NetworkMeasureRow]


class MetricTopRow(BaseModel):
    """Row in GET /metrics/top response."""
    npa_id: str
    value: Optional[float]
    propcrim_1k: Optional[float]


class SpatialNearRow(BaseModel):
    """Row in GET /spatial/near response."""
    npa_id: str
    dist_ft: float
    propcrim_1k: Optional[float]
