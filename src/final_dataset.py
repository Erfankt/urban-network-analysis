"""
Final step of this repository: assemble the modeling dataset that the spatial models
in the paper consume (they are fitted outside this repository). Everything they need
is written to data/final_dataset/ in one place:

    final_dataset.shp               one row per neighborhood, with geometry
    final_dataset.csv               the same table without geometry
    final_dataset_dictionary.csv    one row per column: its role and how to use it
"""
import os
import pandas as pd, geopandas as gpd


def area_in_acres(gdf):
    """Polygon area in acres; the layer must be in a projected CRS in feet or metres."""
    unit = ""
    if gdf.crs is not None and gdf.crs.axis_info:
        unit = (gdf.crs.axis_info[0].unit_name or "").lower()
    if "foot" in unit or "feet" in unit:
        return gdf.geometry.area / 43560.0
    if "met" in unit:
        return gdf.geometry.area / 4046.8564224
    raise ValueError(f"Cannot convert areas to acres: CRS unit is '{unit or 'unknown'}'. "
                     "Reproject the neighborhoods layer to a projected CRS in feet or metres.")


def assemble_final_dataset(gdf, crime_col, representatives, pca_net, pca_ctrl):
    """
    Build the final dataset and its data dictionary.

    gdf              : extracted_measures.shp as a GeoDataFrame (neighborhood attributes + measures)
    crime_col        : name of the dependent variable (property crime count)
    representatives  : one street-network measure per retained network component
    pca_net, pca_ctrl: the report dicts returned by explore_pca() for the two blocks
    Returns (final_gdf, dictionary_df).
    """
    base = gdf[["id", "geometry"]].copy().reset_index(drop=True)
    centroids = gdf.geometry.centroid
    base["cent_x"] = centroids.x.values
    base["cent_y"] = centroids.y.values
    base[crime_col] = gdf[crime_col].values
    base["area_acre"] = area_in_acres(gdf).values
    for v in representatives:
        base[v] = gdf[v].values

    net = pca_net["rotated_scores"].copy().reset_index(drop=True)
    net.columns = [f"PC{i+1}_net" for i in range(net.shape[1])]
    ctrl = pca_ctrl["rotated_scores"].copy().reset_index(drop=True)
    ctrl.columns = [f"PC{i+1}_ctrl" for i in range(ctrl.shape[1])]

    final = gpd.GeoDataFrame(pd.concat([base, net, ctrl], axis=1), geometry="geometry", crs=gdf.crs)

    crs_name = gdf.crs.to_string() if gdf.crs is not None else "unknown CRS"
    rot = pca_net["rotated_loadings"]
    rows = [
        ("id", "identifier", "neighborhood identifier (unique)"),
        ("cent_x", "coordinates", f"polygon centroid x in the layer CRS ({crs_name}); e.g. for GWR"),
        ("cent_y", "coordinates", f"polygon centroid y in the layer CRS ({crs_name}); e.g. for GWR"),
        (crime_col, "dependent variable", "property crime count per neighborhood"),
        ("area_acre", "exposure", "neighborhood area in acres; use log(area_acre) as the offset in the count models"),
    ]
    for comp, v in zip(rot.columns, representatives):
        rows.append((v, "street network (representative measure)",
                     f"stands for {comp} (rotated loading {rot.loc[v, comp]:+.2f}); "
                     "apply log(x + 1) before modeling, as in the paper"))
    for c in net.columns:
        rows.append((c, "street network (component score)",
                     "rotated component score of the street-network block; alternative to the representative "
                     "measures for a robustness check; do not log-transform"))
    for c in ctrl.columns:
        rows.append((c, "control (component score)",
                     "rotated component score of the socio-spatial block; enter as a control"))
    dictionary = pd.DataFrame(rows, columns=["column", "role", "description"])
    return final, dictionary


def write_final_dataset(final_gdf, dictionary, out_dir="../data/final_dataset", stem="final_dataset"):
    """Write the shapefile, the CSV (without geometry) and the dictionary; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    shp = os.path.join(out_dir, f"{stem}.shp")
    csv = os.path.join(out_dir, f"{stem}.csv")
    dic = os.path.join(out_dir, f"{stem}_dictionary.csv")
    final_gdf.to_file(shp)
    final_gdf.drop(columns="geometry").to_csv(csv, index=False)
    dictionary.to_csv(dic, index=False)
    return shp, csv, dic
