"""
Script version of pca_analysis.ipynb: the same steps and the same shared functions
(common.py, user_inputs.py, final_dataset.py), run from a terminal instead of a notebook.

    python pca_analysis.py [--no-windows] [--no-distribution-plots]

Reads   ../data/final_dataset/extracted_measures.shp   (written by run_analysis.py)
        ../data/final_dataset/user_selection.json      (inputs and choices made in run_analysis.py)
Writes  ../data/final_dataset/final_dataset.shp / .csv  (the modeling dataset: the repository's end product)
        ../data/final_dataset/final_dataset_dictionary.csv  (what each column is and how to use it)
        ../data/final_dataset/user_selection.json      (all interactive choices, for the record)
        ../output/pca_diagnostics/                     (every figure as PNG, plus the neighborhood map)

Every figure is saved. By default the PCA-stage figures (overview and each candidate
report) also open in windows so they can be inspected while the prompt waits: close the
windows to continue. --no-windows saves them silently (for remote or headless runs);
--no-distribution-plots skips the per-variable histogram/boxplot figures.
"""
import argparse, os, sys, warnings
from pathlib import Path

# The shared modules use paths relative to src/, so run from there whatever the caller's cwd is.
os.chdir(Path(__file__).resolve().parent)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--no-windows", action="store_true",
                    help="never open figure windows (figures are still saved)")
parser.add_argument("--no-distribution-plots", action="store_true",
                    help="skip the per-variable histogram/boxplot figures")
args = parser.parse_args()

import matplotlib
if args.no_windows:
    matplotlib.use("Agg")          # must happen before pyplot is imported anywhere
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, geopandas as gpd, seaborn as sns, folium

from street_network_processor import StreetNetworkProcessor, NETWORK_METRICS
from user_inputs import load_selection, update_selection, select_pca_method
from common import explore_pca, select_representatives, FigureSaver
from final_dataset import assemble_final_dataset, write_final_dataset

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.precision", 3)

OUT_DIR = Path("../output/pca_diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Figure handling: every figure common.py would "show" is saved to OUT_DIR and,
# unless --no-windows, also opened in a window. plt.show() blocks until the
# windows are closed, which lets the user look at a report before answering the
# next prompt.
# --------------------------------------------------------------------------
figures = FigureSaver(OUT_DIR, show=not args.no_windows).install()


def section(title):
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")


# --------------------------------------------------------------------------
# 1. Inputs: the variable selection made in run_analysis.py and the measures table
# --------------------------------------------------------------------------
section("1. Loading inputs")
selection = load_selection()                  # inputs and choices recorded by run_analysis.py
crime_col, ctrl_vars, buffer_list_mile = selection["crime_col"], selection["ctrl_vars"], selection["buffer_list_mile"]
gdf = gpd.read_file("../data/final_dataset/extracted_measures.shp")

final_list_net = StreetNetworkProcessor.measure_columns(buffer_list_mile)   # same naming rule as stage 1

missing_cols = [c for c in final_list_net + ctrl_vars if c not in gdf.columns]
if missing_cols:
    sys.exit(f"Columns missing from extracted_measures.shp: {missing_cols}\n"
             "Run run_analysis.py first, or check buffer_list_mile and the selected control variables.")
print(f"{len(gdf)} neighborhoods, {len(final_list_net)} street-network measures "
      f"({len(buffer_list_mile)} buffers x {len(NETWORK_METRICS)} metrics), {len(ctrl_vars)} control variables")
print(f"Dependent variable: {crime_col}")

# --------------------------------------------------------------------------
# 2. Interactive map of the neighborhoods with their values (HTML file)
# --------------------------------------------------------------------------
section("2. Neighborhood map")
popup_vars = ["id"] + final_list_net + ctrl_vars
gdf_vis = gdf.to_crs(4326)


def fmt(v):
    return f"{v:.2f}" if isinstance(v, (float, int)) and pd.notna(v) else v


features = []
for _, r in gdf_vis.iterrows():
    popup_html = "<br>".join(f"{v}: {fmt(r.get(v, 'NA'))}" for v in popup_vars)
    features.append({"type": "Feature", "geometry": r.geometry.__geo_interface__,
                     "properties": {"id": r["id"], "popup": popup_html}})

m = folium.Map([gdf_vis.geometry.centroid.y.mean(), gdf_vis.geometry.centroid.x.mean()],
               zoom_start=11, tiles="CartoDB positron")
folium.GeoJson({"type": "FeatureCollection", "features": features},
               tooltip=folium.GeoJsonTooltip(fields=["id"], aliases=["ID:"]),
               popup=folium.GeoJsonPopup(fields=["popup"], labels=False, max_width=350)).add_to(m)
map_path = OUT_DIR / "neighborhood_map.html"
m.save(str(map_path))
print(f"map saved: {map_path} (open it in a browser)")

# --------------------------------------------------------------------------
# 3. Descriptive statistics and missing values
# --------------------------------------------------------------------------
section("3. Descriptive statistics")
print(gdf[final_list_net + ctrl_vars].describe().T.round(3).to_string())

section("4. Missing values")
missing_values = gdf[final_list_net + ctrl_vars].isnull().sum()
missing_values = missing_values[missing_values > 0]
if missing_values.empty:
    print("No missing values in any feature")
else:
    print(missing_values.to_string())
    print("\nWARNING: missing values must be handled before the component analysis "
          "(factor_analyzer imputes the median by default).")

# --------------------------------------------------------------------------
# 5. Distribution diagnostics per variable (one PNG each: histogram, boxplot, points)
# --------------------------------------------------------------------------
section("5. Distribution diagnostics")
if args.no_distribution_plots:
    print("skipped (--no-distribution-plots)")
else:
    numerical_columns = gdf[final_list_net + ctrl_vars].select_dtypes(include="number").columns
    for col in numerical_columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(gdf[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"{col}  (skew = {gdf[col].skew():.2f})")
        sns.boxplot(x=gdf[col], ax=axes[1])
        axes[1].set_title("boxplot")
        sns.boxplot(x=gdf[col], ax=axes[2])
        sns.stripplot(x=gdf[col], color="black", jitter=True, alpha=0.6, ax=axes[2])
        axes[2].set_title("boxplot with observations")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"dist_{FigureSaver.slug(col)}.png", dpi=110)
        plt.close(fig)
    print(f"{len(numerical_columns)} figures saved as {OUT_DIR}/dist_<variable>.png")

# --------------------------------------------------------------------------
# 6. PCA: method, then the interactive choice of components for each block
# --------------------------------------------------------------------------
section("6. Component analysis settings")
pca_method = select_pca_method()   # principal / minres / ml (see the README)
pca_rotation = "varimax"           # rotation of the retained components (paper: varimax)
print(f"method = {pca_method}, rotation = {pca_rotation}")

section("7. Street-network measures: overview, candidate solutions, final choice")
figures.stage = "network"
n_net, pca_net = explore_pca(gdf, final_list_net, label="street network",
                             rotation=pca_rotation, method=pca_method)

section("8. Street-network measures: one representative measure per component")
network_representatives = select_representatives(pca_net, label="street network")

section("9. Socio-spatial controls: overview, candidate solutions, final choice")
figures.stage = "controls"
n_ctrl, pca_ctrl = explore_pca(gdf, ctrl_vars, label="socio-spatial controls",
                               rotation=pca_rotation, method=pca_method)

update_selection(pca_method=pca_method, pca_rotation=pca_rotation,
                 pca_factors_network=n_net, pca_factors_ctrl=n_ctrl,
                 network_representatives=network_representatives)

# --------------------------------------------------------------------------
# 10. Assemble and write the final modeling dataset (the repository's end product)
# --------------------------------------------------------------------------
section("10. Final dataset")
final_gdf, dictionary = assemble_final_dataset(gdf, crime_col, network_representatives, pca_net, pca_ctrl)
shp, csv, dic = write_final_dataset(final_gdf, dictionary)
print(f"{len(final_gdf)} rows x {final_gdf.shape[1] - 1} columns written to:\n  {shp}\n  {csv}\n  {dic}")
print(dictionary.to_string(index=False))
print(f"\nChoices recorded in ../data/final_dataset/user_selection.json; figures in {OUT_DIR}")
