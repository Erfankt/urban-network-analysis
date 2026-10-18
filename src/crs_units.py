"""
Units of the working CRS.

The pipeline measures on the street layer, so its coordinate system defines the length unit. Any
projected CRS works (feet, US survey feet, metres, ...): buffers are given in miles, areas are in
acres, and the street-network measures keep their published units (street density in feet per
acre, shortest path in miles) whatever the layer unit is. A street layer in latitude/longitude is
moved to the UTM zone that fits its extent (ensure_projected), because degrees cannot be measured.

For layers in feet the constants are exactly 5280 and 43560, so results do not change.
"""

METRES_PER_MILE = 1609.344
FEET_PER_METRE = 1 / 0.3048


def unit_name(crs):
    """Name of the first axis unit of a CRS, lower case ('' when unknown)."""
    if crs is None or not crs.axis_info:
        return ""
    return (crs.axis_info[0].unit_name or "").lower()


def metres_per_unit(crs):
    """Metres in one unit of a projected CRS (US survey foot: 0.3048006..., metre: 1)."""
    factor = crs.axis_info[0].unit_conversion_factor if crs is not None and crs.axis_info else None
    return float(factor) if factor else None


def is_foot(crs):
    name = unit_name(crs)
    return "foot" in name or "feet" in name


def units_per_mile(crs):
    """Layer units in a mile: 5280 for feet (either kind), otherwise from the unit's size in metres."""
    if is_foot(crs):
        return 5280.0
    m = metres_per_unit(crs)
    if not m:
        raise ValueError(f"The CRS length unit is unknown ({unit_name(crs) or 'none'}).")
    return METRES_PER_MILE / m


def units_per_acre(crs):
    """Layer area units (unit squared) in an acre; exactly 43560 for feet, 4046.8564224 for metres."""
    return units_per_mile(crs) ** 2 / 640.0


def feet_per_unit(crs):
    """Feet in one layer unit: 1 for feet, 3.2808... for metres (street density is reported per foot)."""
    if is_foot(crs):
        return 1.0
    return metres_per_unit(crs) * FEET_PER_METRE


def unit_label(crs):
    """'ft' for feet, 'm' for metres, otherwise the unit's own name."""
    if is_foot(crs):
        return "ft"
    name = unit_name(crs)
    return "m" if name.startswith("met") else name


def working_crs(edges):
    """The CRS the pipeline measures in: the streets CRS, or the UTM zone of the streets when they are in degrees."""
    if edges.crs is not None and edges.crs.is_geographic:
        return edges.estimate_utm_crs()
    return edges.crs


def ensure_projected(edges, neighborhoods=None):
    """
    Return (edges, neighborhoods, note). A street layer in a geographic CRS (degrees) is reprojected to
    the UTM zone of its extent, and the neighborhoods layer with it when it has a CRS; a projected layer is
    returned as it is. note is None when nothing changed.
    """
    crs = edges.crs
    if crs is None or not crs.is_geographic:
        return edges, neighborhoods, None
    utm = working_crs(edges)
    edges = edges.to_crs(utm)
    if neighborhoods is not None and neighborhoods.crs is not None:
        neighborhoods = neighborhoods.to_crs(utm)
    name = utm.name or utm.to_string()
    code = utm.to_epsg()
    note = (f"the streets are in latitude/longitude, which cannot be measured in miles or acres: "
            f"both layers were moved to {('EPSG:%d ' % code) if code else ''}({name}), the UTM zone of the streets")
    return edges, neighborhoods, note
