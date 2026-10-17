"""
The only module of the app that talks to src/: it loads and checks the inputs, starts and
watches the stage-1 process, runs the stage-2 computations and writes the final dataset.
The pages call these functions and render what they return; nothing here prints, prompts
or imports Streamlit, so it can be tested from a plain Python session.

Everything goes through the same functions as the terminal scripts, from the same working
directory (src/), so a run made from the app leaves the same files in the same places:
data/final_dataset/, data/checkpoints/ and output/pca_diagnostics/.
"""
import io, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
SRC_DIR = REPO_DIR / "src"
DATA_DIR = REPO_DIR / "data"
FINAL_DIR = DATA_DIR / "final_dataset"
OUT_DIR = REPO_DIR / "output" / "pca_diagnostics"
OUTPUT_DIR = REPO_DIR / "output"                       # everything the app and the scripts write besides data/
APP_OUT_DIR = REPO_DIR / "output" / "app"           # the stage-1 log and run record
STAGE1_LOG = APP_OUT_DIR / "stage1.log"
STAGE1_RUN = APP_OUT_DIR / "stage1_run.json"

# The src modules address data/ and output/ relative to src/ (../data/...), exactly as the
# terminal scripts do, so the app works from src/ as well.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)

import matplotlib
matplotlib.use("Agg")                                 # figures are rendered to images, never to windows
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, geopandas as gpd, seaborn as sns, folium, psutil

from run_analysis import DEFAULTS
from user_inputs import (check_layers, identifier_candidates, default_id_column, id_column, parse_buffers,
                         numeric_columns, save_selection, load_selection, update_selection, selection_summary,
                         PCA_METHODS, SELECTION_PATH)
from street_network_processor import (StreetNetworkProcessor, NETWORK_METRICS, align_neighborhoods_crs, crs_label,
                                      SAME_CRS_TOLERANCE)
from common import (missing_summary, overview_stats, rules_of_thumb_text, correlation_figure, scree_figure,
                    fit_solution, unrotated_loadings_figure, rotated_loadings_figure, solution_summary_text,
                    component_candidates, candidate_text, FigureSaver)
from final_dataset import assemble_final_dataset, write_final_dataset

MEASURES_PATH = FINAL_DIR / "extracted_measures.shp"
STAGE1_DONE = "Network Processing Complete"           # printed by run_analysis.py at the end
STAGE1_FAILED = "An error occurred during processing"


# ===========================================================================
# Inputs
# ===========================================================================

LAYER_DIRS = {"streets": "streetnetwork", "neighborhoods": "neighborhoods"}
LAYER_SUFFIXES = (".shp", ".gpkg", ".geojson")
SHAPEFILE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix")
FT_PER_MILE, SQFT_PER_ACRE, SQFT_PER_SQMILE = 5280.0, 43560.0, 5280.0 ** 2


def discover_shapefiles():
    """The layers under data/streetnetwork and data/neighborhoods (.shp, .gpkg, .geojson), as paths relative to src/."""
    found = {"streets": [], "neighborhoods": []}
    for key, name in LAYER_DIRS.items():
        folder = DATA_DIR / name
        if folder.is_dir():
            found[key] = [f"../data/{name}/{p.name}" for p in sorted(folder.iterdir())
                          if p.suffix.lower() in LAYER_SUFFIXES]
    return found


def _safe_name(name):
    """A file name without folders or odd characters (uploads are written under data/ only)."""
    name = Path(str(name).replace("\\", "/")).name
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def save_uploaded_layer(kind, uploads):
    """
    Save a layer the user uploaded into data/streetnetwork/ or data/neighborhoods/, so it appears in the pickers.

    `uploads` is a list of (file name, bytes): the parts of one shapefile (.shp .shx .dbf .prj ...), a single .zip
    holding one shapefile, or a single .gpkg / .geojson. Returns (path relative to src/, message). Raises ValueError
    with a plain explanation when the files cannot be used. An existing layer with the same name is replaced.
    """
    folder = DATA_DIR / LAYER_DIRS[kind]
    files = {}
    for name, data in uploads:
        name = _safe_name(name)
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for member in z.namelist():
                        base = _safe_name(member)
                        if not member.endswith("/") and Path(base).suffix.lower() in SHAPEFILE_PARTS + (".gpkg", ".geojson") \
                                and not base.startswith("."):
                            files[base] = z.read(member)
            except zipfile.BadZipFile:
                raise ValueError(f"{name} is not a valid zip file.")
        else:
            files[name] = data
    if not files:
        raise ValueError("No files were uploaded.")

    single = [n for n in files if Path(n).suffix.lower() in (".gpkg", ".geojson")]
    shp = [n for n in files if n.lower().endswith(".shp")]
    if single and shp:
        raise ValueError("Upload either one shapefile (its parts together) or one .gpkg/.geojson, not both.")
    if single:
        if len(single) > 1:
            raise ValueError("Upload one .gpkg or .geojson at a time.")
        keep, main = [single[0]], single[0]
    else:
        if len(shp) != 1:
            raise ValueError("Upload exactly one shapefile: select its .shp and the other parts with the same name "
                             "(.shx, .dbf, .prj), or one .zip holding them." if not shp else
                             "Several shapefiles were found; upload one at a time.")
        stem = Path(shp[0]).stem
        keep = [n for n in files if Path(n).stem == stem and Path(n).suffix.lower() in SHAPEFILE_PARTS]
        have = {Path(n).suffix.lower() for n in keep}
        missing = [e for e in (".dbf",) if e not in have]
        if missing:
            raise ValueError(f"The shapefile {shp[0]} needs its {', '.join(missing)} file as well.")
        main = shp[0]
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(main).stem
    for old in folder.iterdir():                     # replace the previous copy completely (no stale .cpg/.sbn)
        if old.stem == stem and old.suffix.lower() in SHAPEFILE_PARTS + (".gpkg", ".geojson"):
            old.unlink()
    for n in keep:
        (folder / n).write_bytes(files[n])
    note = ""
    if not main.lower().endswith((".gpkg", ".geojson")):
        have = {Path(n).suffix.lower() for n in keep}
        if ".prj" not in have:
            note = " No .prj file came with it, so the coordinate system is unknown; the check will say so."
    return f"../data/{LAYER_DIRS[kind]}/{main}", f"Saved {main} ({len(keep)} file{'s' if len(keep) > 1 else ''}) in data/{LAYER_DIRS[kind]}/.{note}"


def read_layers(streets_path, neighborhoods_path):
    """The two layers as read from disk, nothing checked yet. Raises when a file cannot be read."""
    for p in (streets_path, neighborhoods_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} does not exist")
    return {"streets_path": str(streets_path), "neighborhoods_path": str(neighborhoods_path),
            "edges": gpd.read_file(streets_path), "neighborhoods": gpd.read_file(neighborhoods_path)}


def id_candidates(neighborhoods):
    return identifier_candidates(neighborhoods)


def crs_relation(edges, neighborhoods):
    """
    How the two CRS compare: 'same', 'equivalent' (the same system written differently: no coordinate
    moves by more than SAME_CRS_TOLERANCE ft), 'different', or 'unknown' (a layer without a CRS).
    """
    if edges.crs is None or neighborhoods.crs is None:
        return "unknown"
    if edges.crs == neighborhoods.crs:
        return "same"
    moved = neighborhoods.to_crs(edges.crs)
    shift = float(np.nanmax(np.abs(moved.bounds.to_numpy() - neighborhoods.bounds.to_numpy())))
    return "equivalent" if shift < SAME_CRS_TOLERANCE else "different"


def check_inputs(layers, id_col):
    """
    Validate the two layers for the chosen identifier column (user_inputs.check_layers) and put the
    neighborhoods in the streets CRS (user_inputs.align_neighborhoods_crs, as the pipeline does).
    Returns a dict: problems (empty = fine; then the rest is set), crs {relation, streets, neighborhoods},
    neighborhoods (aligned), n_segments, n_neighborhoods, columns (the numeric columns a variable can be
    chosen from), id_col, summary, note.
    """
    edges, nb = layers["edges"], layers["neighborhoods"]
    info = {"streets_path": layers["streets_path"], "neighborhoods_path": layers["neighborhoods_path"],
            "id_col": id_col, "problems": check_layers(edges, nb, id_col), "note": None, "summary": ""}
    if info["problems"]:
        return info
    relation = crs_relation(edges, nb)
    aligned, note = align_neighborhoods_crs(edges, nb)
    info.update(neighborhoods=aligned, n_segments=int(len(edges)), n_neighborhoods=int(len(nb)),
                columns=numeric_columns(nb, id_col), note=note if relation == "different" else None,
                crs={"relation": relation, "streets": crs_label(edges.crs), "neighborhoods": crs_label(nb.crs)},
                summary=(f"streets: {len(edges)} segments | neighborhoods: {len(nb)} polygons "
                         f"| CRS: {crs_label(edges.crs)}"))
    return info


def write_matched_neighborhoods(layers):
    """
    Write a copy of the neighborhoods layer reprojected to the streets CRS, next to the original
    (data/neighborhoods/<name>_epsg<code>.shp, or .gpkg for a GeoPackage), and return its path relative
    to src/. The pipeline would reproject on the fly anyway; the copy makes the match explicit on disk.
    """
    edges, nb = layers["edges"], layers["neighborhoods"]
    src = Path(layers["neighborhoods_path"])
    code = edges.crs.to_epsg()
    tag = f"epsg{code}" if code else "reprojected"
    ext = ".gpkg" if src.suffix.lower() == ".gpkg" else ".shp"
    folder = DATA_DIR / LAYER_DIRS["neighborhoods"]
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{src.stem}_{tag}{ext}"
    nb.to_crs(edges.crs).to_file(out)
    return f"../data/{LAYER_DIRS['neighborhoods']}/{out.name}"


def _histogram(values, bins=30, clip_quantile=0.99):
    """Counts of `values` in `bins` equal bins up to the clip quantile; returns a dict for a bar chart."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"start": [], "end": [], "count": [], "clipped": 0, "upper": 0.0}
    upper = float(np.quantile(v, clip_quantile)) if v.size > 20 else float(v.max())
    upper = upper if upper > 0 else float(v.max() or 1.0)
    inside = v[v <= upper]
    counts, edges_ = np.histogram(inside, bins=bins, range=(0.0, upper))
    return {"start": edges_[:-1].tolist(), "end": edges_[1:].tolist(), "count": counts.astype(int).tolist(),
            "clipped": int(v.size - inside.size), "upper": upper}


def layer_profile(layers, info):
    """
    The numbers behind the two dataset panels of the Inputs page: counts, lengths or areas (in the
    streets CRS, US feet), geometry notes, columns, CRS, and a binned histogram each.
    """
    edges, nb = layers["edges"], info["neighborhoods"]
    length_ft = edges.geometry.length.to_numpy(dtype=float)
    area_sqft = nb.geometry.area.to_numpy(dtype=float)
    attr_cols = [c for c in edges.columns if c != "geometry"]
    numeric = info["columns"]
    other = [c for c in nb.columns if c not in numeric and c not in (info["id_col"], "geometry")]
    missing_cells = int(nb[numeric].isna().sum().sum()) if numeric else 0
    return {
        "streets": {
            "n": int(len(edges)), "total_mi": float(length_ft.sum() / FT_PER_MILE),
            "median_ft": float(np.nanmedian(length_ft)), "mean_ft": float(np.nanmean(length_ft)),
            "p90_ft": float(np.nanquantile(length_ft, 0.9)), "max_ft": float(np.nanmax(length_ft)),
            "multipart": int((edges.geom_type == "MultiLineString").sum()),
            "empty": int(edges.geometry.is_empty.sum()),
            "columns": attr_cols, "crs": crs_label(edges.crs),
            "hist": _histogram(length_ft),
        },
        "neighborhoods": {
            "n": int(len(nb)), "total_sqmi": float(area_sqft.sum() / SQFT_PER_SQMILE),
            "median_acre": float(np.nanmedian(area_sqft) / SQFT_PER_ACRE),
            "min_acre": float(np.nanmin(area_sqft) / SQFT_PER_ACRE), "max_acre": float(np.nanmax(area_sqft) / SQFT_PER_ACRE),
            "multipart": int((nb.geom_type == "MultiPolygon").sum()),
            "id_col": info["id_col"], "numeric": numeric, "other": other, "missing_cells": missing_cells,
            "crs": crs_label(layers["neighborhoods"].crs), "crs_relation": info["crs"]["relation"],
            "hist": _histogram(area_sqft / SQFT_PER_ACRE),
        },
    }


# ---- the map -----------------------------------------------------------------------------------------------
BLANK_TILE = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="

BASEMAPS = [   # (name, tile URL, attribution); the first is shown at first, the others are offered in the control
    ("OpenStreetMap", "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
     "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"),
    ("Light gray (Esri)",
     "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
     "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"),
    ("Imagery (Esri)",
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"),
    ("Topographic (OpenTopoMap)", "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
     "Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; <a href='https://opentopomap.org'>OpenTopoMap</a> (CC-BY-SA)"),
]
STREET_SIMPLIFY_FT = 10.0          # vertices closer than this to the line are dropped before drawing (display only)
STREET_POPUP_COLUMNS = 8           # attribute columns listed for a clicked street (the layer has tens of thousands)


def _json_value(v):
    """A value as the popup should show it (JSON-safe)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 3)
    if isinstance(v, (int, bool)):
        return v
    return str(v)


def _feature_collection(gdf, columns, simplify_ft=None, precision=5, max_columns=None):
    """A GeoJSON FeatureCollection (dict) in WGS 84 with the chosen columns (at most max_columns) as properties."""
    geom = gdf.geometry.simplify(simplify_ft, preserve_topology=False) if simplify_ft else gdf.geometry
    g = gpd.GeoSeries(geom, crs=gdf.crs).to_crs(4326)
    cols = [c for c in columns if c in gdf.columns][:max_columns]
    values = {c: gdf[c].tolist() for c in cols}
    features = []
    for i, shape in enumerate(g):
        if shape is None or shape.is_empty:
            continue
        features.append({"type": "Feature", "geometry": _round_geometry(shape.__geo_interface__, precision),
                         "properties": {c: _json_value(values[c][i]) for c in cols}})
    return {"type": "FeatureCollection", "features": features}, cols


def _round_geometry(geometry, precision):
    def rnd(coords):
        if isinstance(coords[0], (int, float)):
            return [round(coords[0], precision), round(coords[1], precision)]
        return [rnd(c) for c in coords]
    return {"type": geometry["type"], "coordinates": rnd(geometry["coordinates"])}


def layers_map_html(neighborhoods, edges=None, id_col="id", neighborhood_columns=None, height_note=None):
    """
    An interactive map (folium): the neighborhoods and, when given, the streets (drawn above them, so a
    click on a street reaches it) as two layers that can be switched on and off, a choice of basemaps (OpenStreetMap first), a fullscreen button, and the
    attributes of a feature shown when it is clicked. Returns the HTML.
    neighborhood_columns limits the attributes listed for a neighborhood (default: all, up to a dozen).
    """
    from folium.plugins import Fullscreen

    nb_cols = list(neighborhood_columns) if neighborhood_columns is not None else \
        [id_col] + [c for c in neighborhoods.columns if c not in (id_col, "geometry")]
    nb_fc, nb_cols = _feature_collection(neighborhoods, nb_cols)
    b = gpd.GeoSeries(neighborhoods.geometry, crs=neighborhoods.crs).to_crs(4326).total_bounds
    m = folium.Map(location=[(b[1] + b[3]) / 2, (b[0] + b[2]) / 2], zoom_start=11, tiles=None,
                   prefer_canvas=True, control_scale=True)
    for i, (name, url, attr) in enumerate(BASEMAPS):
        folium.TileLayer(tiles=url, name=name, attr=attr, show=(i == 0), max_zoom=19, control=True).add_to(m)
    # "No basemap": a blank (transparent) base layer on a white background, so only the data shows
    folium.TileLayer(tiles=BLANK_TILE, name="No basemap", attr="no basemap", show=False, max_zoom=19, control=True).add_to(m)
    m.get_root().html.add_child(folium.Element("<style>.leaflet-container{background:#fff}</style>"))

    folium.GeoJson(nb_fc, name=f"Neighborhoods ({len(nb_fc['features'])})", show=True,
                   style_function=lambda f: {"color": "#0e8a7f", "weight": 2, "fillColor": "#0e8a7f", "fillOpacity": 0.14},
                   highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.3},
                   popup=folium.GeoJsonPopup(fields=nb_cols, aliases=nb_cols, labels=True, localize=True,
                                             max_width=360)).add_to(m)
    if edges is not None:
        st_cols = [c for c in edges.columns if c != "geometry"]
        st_fc, st_cols = _feature_collection(edges, st_cols, simplify_ft=STREET_SIMPLIFY_FT,
                                             max_columns=STREET_POPUP_COLUMNS)
        folium.GeoJson(st_fc, name=f"Streets ({len(st_fc['features'])} segments)", show=True, smooth_factor=1.0,
                       style_function=lambda f: {"color": "#34445a", "weight": 1.2, "opacity": 0.8},
                       popup=folium.GeoJsonPopup(fields=st_cols, aliases=st_cols, labels=True, localize=True,
                                                 max_width=360) if st_cols else None).add_to(m)
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    Fullscreen(position="topleft", title="Full screen", title_cancel="Exit full screen").add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    # the popup table: a compact, readable layout
    m.get_root().header.add_child(folium.Element(
        "<style>.leaflet-popup-content table{border-collapse:collapse;font:12px/1.35 system-ui,sans-serif}"
        ".leaflet-popup-content th{text-align:left;padding:1px 10px 1px 0;color:#52514e;font-weight:600}"
        ".leaflet-popup-content td{padding:1px 0}"
        ".leaflet-control-layers{font:12px system-ui,sans-serif}</style>"))
    return m.get_root().render()


def neighborhoods_map_html(neighborhoods, popup_vars=("id",), id_col="id"):
    """The map written with the final dataset: the neighborhoods with the selected variables on click."""
    return layers_map_html(neighborhoods, edges=None, id_col=id_col, neighborhood_columns=list(popup_vars))


def buffer_codes(buffers):
    return {b: StreetNetworkProcessor.buffer_code(b) for b in buffers}


# ===========================================================================
# Stage 1: run_analysis.py as a background process, watched through its log and checkpoints
# ===========================================================================

def stage1_command(sel, recompute=False):
    """The run_analysis.py command line that answers every prompt up front."""
    cmd = [sys.executable, "run_analysis.py",
           "--streets", sel["streets_path"], "--neighborhoods", sel["neighborhoods_path"], "--id-col", id_column(sel),
           "--buffers", ",".join(str(b) for b in sel["buffer_list_mile"]),
           "--crime", sel["crime_col"], "--controls", ",".join(sel["ctrl_vars"])]
    if recompute:
        cmd.append("--recompute")
    return cmd


def start_stage1(sel, recompute=False, n_neighborhoods=None):
    """
    Record the selection (the same user_selection.json the terminal writes) and start
    run_analysis.py in the background, its output going to output/app/stage1.log. Returns the
    run record, which is also kept in output/app/stage1_run.json so the page finds the run
    again after a browser refresh.
    """
    if stage1_status()["state"] == "running":
        raise RuntimeError("a stage-1 run is already in progress")
    save_selection(sel)
    APP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = stage1_command(sel, recompute)
    log = open(STAGE1_LOG, "w", encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=SRC_DIR, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env)
    log.close()
    try:
        created = psutil.Process(proc.pid).create_time()
    except psutil.Error:
        created = time.time()
    run = {"pid": proc.pid, "create_time": created, "started": time.time(), "command": cmd,
           "selection": sel, "recompute": recompute, "log": str(STAGE1_LOG),
           "n_neighborhoods": int(n_neighborhoods) if n_neighborhoods else None}
    with open(STAGE1_RUN, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    return run


def _process_of(run):
    """The psutil.Process of a run record, or None when it has ended (guarding against pid reuse)."""
    try:
        p = psutil.Process(run["pid"])
        if abs(p.create_time() - run["create_time"]) > 2 or p.status() == psutil.STATUS_ZOMBIE:
            return None
        return p
    except psutil.Error:
        return None


def read_log(max_chars=200_000):
    if not STAGE1_LOG.exists():
        return ""
    with open(STAGE1_LOG, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text[-max_chars:]


def log_lines(text, n=25):
    """
    The last n lines of the log, with the progress-bar updates collapsed: only the latest state
    of each buffer's bar is kept (tqdm writes one update per line when the output is a file).
    """
    lines = []
    for raw in text.splitlines():
        line = raw.split("\r")[-1].rstrip()
        if not line:
            continue
        # a tqdm bar ("Description:  25%|####   | 117/475 [...]") redraws itself: keep one line per bar,
        # the latest, like a terminal does
        m = re.match(r"(Analyzing NPAs \([^)]*\))", line) or re.match(r"^(.*?):\s+\d{1,3}%\|", line)
        if m and lines and lines[-1].startswith(m.group(1)):
            lines[-1] = line
        else:
            lines.append(line)
    return lines[-n:]


def _checkpoint_dir_from_log(text):
    m = re.findall(r"^Checkpoints: (.+)$", text, flags=re.M)
    return Path(m[-1].strip()) if m else None


def _rows_in_checkpoint(path):
    """Neighborhoods already measured in one checkpoint CSV (its data rows)."""
    try:
        with open(path, "rb") as f:
            n = sum(1 for _ in f)
        return max(n - 1, 0)
    except OSError:
        return 0


def stage1_progress(run, text):
    """Per buffer: (neighborhoods done, total), read from the checkpoint CSVs named in the log."""
    sel = run["selection"]
    total = run.get("n_neighborhoods") or 0
    ckpt = _checkpoint_dir_from_log(text)
    progress = {}
    for b in sel["buffer_list_mile"]:
        done = 0
        if ckpt is not None:
            done = _rows_in_checkpoint(ckpt / f"measures_{StreetNetworkProcessor.buffer_code(b)}.csv")
        progress[b] = (min(done, total) if total else done, total)
    return progress, ckpt


def stage1_status():
    """
    Where stage 1 stands: a dict with
      state     : 'none' (never started), 'running', 'finished', 'failed' or 'stopped' (ended without finishing)
      run       : the run record (None when state is 'none')
      progress  : {buffer: (done, total)}
      log_tail  : the last lines of the log
      elapsed   : seconds since the start
    """
    if not STAGE1_RUN.exists():
        return {"state": "none", "run": None, "progress": {}, "log_tail": [], "elapsed": 0, "checkpoint_dir": None}
    with open(STAGE1_RUN, encoding="utf-8") as f:
        run = json.load(f)
    text = read_log()
    alive = _process_of(run) is not None
    if alive:
        state = "running"
    elif STAGE1_DONE in text:
        state = "finished"
    elif STAGE1_FAILED in text or "Traceback" in text:
        state = "failed"
    else:
        state = "stopped"
    progress, ckpt = stage1_progress(run, text)
    end = time.time() if alive else (STAGE1_LOG.stat().st_mtime if STAGE1_LOG.exists() else run["started"])
    return {"state": state, "run": run, "progress": progress, "log_tail": log_lines(text),
            "elapsed": max(end - run["started"], 0), "checkpoint_dir": str(ckpt) if ckpt else None}


def stop_stage1():
    """End the running process. The checkpoints keep what was measured; a resume continues from there."""
    status = stage1_status()
    p = _process_of(status["run"]) if status["run"] else None
    if p is not None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except psutil.TimeoutExpired:
            p.kill()
    return status


def clear_output_dir():
    """
    Empty output/ (the stage-1 log and run record, the diagnostic figures) when the app shuts down.
    Only the contents go: the folder stays, and so do data/ (the measures, the checkpoints, the final
    dataset) and everything outside the repository's output/ folder. A stage-1 run that is still going
    keeps output/app, its log and run record. Set KEEP_OUTPUT=1 in the environment to keep the folder as is.
    """
    try:
        if os.environ.get("KEEP_OUTPUT") == "1":
            return
        out = OUTPUT_DIR
        if out.is_symlink() or not out.is_dir() or out.resolve() != (REPO_DIR.resolve() / "output"):
            return
        try:
            running = stage1_status()["state"] == "running"
        except Exception:
            running = False
        for item in out.iterdir():
            if running and item.name == "app":
                continue
            try:
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except OSError:
                pass
    except Exception:
        pass                                    # never block a shutdown


def measures_available():
    """True when extracted_measures.shp exists and matches the saved selection (its columns are there)."""
    if not MEASURES_PATH.exists() or not Path(SELECTION_PATH).exists():
        return False
    try:
        sel = load_selection()
        cols = set(gpd.read_file(MEASURES_PATH, rows=1).columns)
    except Exception:
        return False
    needed = (set(StreetNetworkProcessor.measure_columns(sel["buffer_list_mile"])) | set(sel["ctrl_vars"])
              | {sel["crime_col"], id_column(sel)})
    return needed <= cols


# ===========================================================================
# Stage 2: component analysis (in-process; the same functions as pca_analysis.py)
# ===========================================================================

def load_measures():
    """extracted_measures.shp with the saved selection. Returns (gdf, selection, network_columns)."""
    sel = load_selection()
    sel["id_col"] = id_column(sel)
    gdf = gpd.read_file(MEASURES_PATH)
    net = StreetNetworkProcessor.measure_columns(sel["buffer_list_mile"])
    missing_cols = [c for c in net + sel["ctrl_vars"] + [sel["crime_col"], sel["id_col"]] if c not in gdf.columns]
    if missing_cols:
        raise ValueError(f"columns missing from extracted_measures.shp: {missing_cols}; run stage 1 again")
    return gdf, sel, net


def measures_mtime():
    return MEASURES_PATH.stat().st_mtime if MEASURES_PATH.exists() else 0


def check_missing(gdf, net, ctrl_vars, id_col="id"):
    return missing_summary(gdf, net, ctrl_vars, id_col)


def block_overview(gdf, variables):
    return overview_stats(gdf, variables)


def block_solution(gdf, variables, n, method, rotation="varimax"):
    return fit_solution(gdf, variables, n, rotation=rotation, method=method)


def figure_png(fig, dpi=110):
    """Render a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def overview_pngs(stats, label):
    return {"correlation": figure_png(correlation_figure(stats, label)),
            "scree": figure_png(scree_figure(stats, label))}


def solution_pngs(report):
    return {"unrotated": figure_png(unrotated_loadings_figure(report)),
            "rotated": figure_png(rotated_loadings_figure(report))}


def distribution_figure(gdf, col):
    """The per-variable figure of stage 2: histogram, boxplot, boxplot with observations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(gdf[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"{col}  (skew = {gdf[col].skew():.2f})")
    sns.boxplot(x=gdf[col], ax=axes[1])
    axes[1].set_title("boxplot")
    sns.boxplot(x=gdf[col], ax=axes[2])
    sns.stripplot(x=gdf[col], color="black", jitter=True, alpha=0.6, ax=axes[2])
    axes[2].set_title("boxplot with observations")
    fig.tight_layout()
    return fig


def candidates_for(report, component, exclude=(), top_k=5):
    return component_candidates(report, component, exclude=exclude, top_k=top_k)


# ===========================================================================
# Final dataset
# ===========================================================================

def build_final_dataset(gdf, sel, method, rotation, net_report, ctrl_report, representatives,
                        distribution_plots=True, progress=None):
    """
    Write the final dataset, its dictionary, the choices and the diagnostics figures, exactly
    as pca_analysis.py does at the end of a terminal run. `progress(fraction, message)` is
    called along the way when given. Returns a dict with the paths and the dictionary.
    """
    def tick(frac, msg):
        if progress:
            progress(frac, msg)

    tick(0.05, "recording the choices")
    update_selection(pca_method=method, pca_rotation=rotation,
                     pca_factors_network=net_report["n_factors"], pca_factors_ctrl=ctrl_report["n_factors"],
                     network_representatives=list(representatives))

    tick(0.15, "assembling the final dataset")
    id_col = id_column(sel)
    final_gdf, dictionary = assemble_final_dataset(gdf, sel["crime_col"], list(representatives), net_report, ctrl_report,
                                                   id_col=id_col)
    shp, csv, dic = write_final_dataset(final_gdf, dictionary)

    tick(0.35, "saving the component-analysis figures")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    net_vars, ctrl_vars = list(net_report["rotated_loadings"].index), list(ctrl_report["rotated_loadings"].index)
    plt.close("all")
    saver = FigureSaver(OUT_DIR, show=False)            # same file names and numbering as the terminal run
    for stage, label, variables, report in (("network", "street network", net_vars, net_report),
                                            ("controls", "socio-spatial controls", ctrl_vars, ctrl_report)):
        saver.stage = stage
        stats = overview_stats(gdf, variables)
        for make in (lambda: correlation_figure(stats, label), lambda: scree_figure(stats, label),
                     lambda: unrotated_loadings_figure(report), lambda: rotated_loadings_figure(report)):
            make()
            saver.save_open_figures()
            plt.close("all")

    tick(0.6, "saving the neighborhood map")
    popup_vars = [id_col] + net_vars + ctrl_vars
    with open(OUT_DIR / "neighborhood_map.html", "w", encoding="utf-8") as f:
        f.write(neighborhoods_map_html(gdf, popup_vars, id_col=id_col))

    if distribution_plots:
        cols = list(gdf[net_vars + ctrl_vars].select_dtypes(include="number").columns)
        for i, col in enumerate(cols):
            tick(0.6 + 0.4 * i / max(len(cols), 1), f"distribution figure {i + 1} of {len(cols)}: {col}")
            fig = distribution_figure(gdf, col)
            fig.savefig(OUT_DIR / f"dist_{FigureSaver.slug(col)}.png", dpi=110)
            plt.close(fig)

    tick(1.0, "done")
    return {"shp": shp, "csv": csv, "dictionary_path": dic, "dictionary": dictionary,
            "final": final_gdf, "selection_path": SELECTION_PATH, "figures_dir": str(OUT_DIR)}


def final_outputs():
    """The final-dataset files on disk (None for a missing one) and the saved selection."""
    paths = {"shp": FINAL_DIR / "final_dataset.shp", "csv": FINAL_DIR / "final_dataset.csv",
             "dictionary": FINAL_DIR / "final_dataset_dictionary.csv", "selection": Path(SELECTION_PATH).resolve()}
    present = {k: (p if p.exists() else None) for k, p in paths.items()}
    sel = None
    if present["selection"]:
        with open(present["selection"], encoding="utf-8") as f:
            sel = json.load(f)
    return present, sel


def shapefile_zip(stem="final_dataset"):
    """The shapefile and its sidecar files as one zip (bytes), for a download button."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(FINAL_DIR.glob(f"{stem}.*")):
            if p.suffix.lower() in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                z.write(p, p.name)
    return buf.getvalue()
