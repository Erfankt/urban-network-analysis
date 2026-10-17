"""Page 1: the two input layers (from the data folder or from this computer), the neighborhood identifier
column, the coordinate reference systems, a look at each dataset and a map of both."""
import html
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import pipeline as pl

SOURCES = ("Data folder", "This computer")
LAYER_TYPES = ["shp", "shx", "dbf", "prj", "cpg", "sbn", "sbx", "qix", "zip", "gpkg", "geojson"]
UPLOAD_HELP = ("Select the parts of one shapefile together (.shp, .shx, .dbf, .prj), or one .zip holding them, "
               "or one .gpkg / .geojson. Saving copies the files into the data folder of the repository "
               "(data/streetnetwork/ or data/neighborhoods/), where the pipeline reads them; the layer then "
               "appears in the Data folder list.")
BAR = "#2a78d6"            # the one hue of the two single-series charts
INK_2 = "#52514e"
TILE_CSS = """<style>
.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 16px;margin:4px 0 8px}
.tile .lbl{font-size:.82rem;opacity:.72;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .val{font-size:1.5rem;font-weight:600;line-height:1.25;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile[title]:not([title=""]) .lbl::after{content:" ⓘ";opacity:.6}
</style>"""


def _mtimes(*paths):
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)


@st.cache_resource(show_spinner="Reading the layers…")
def _layers(streets, nbhd, mtimes):
    return pl.read_layers(streets, nbhd)


@st.cache_data(show_spinner="Looking at the data…")
def _profile(streets, nbhd, mtimes, id_col):
    layers = _layers(streets, nbhd, mtimes)
    return pl.layer_profile(layers, pl.check_inputs(layers, id_col))


@st.cache_data(show_spinner="Drawing the map…")
def _map_html(streets, nbhd, mtimes, id_col):
    layers = _layers(streets, nbhd, mtimes)
    info = pl.check_inputs(layers, id_col)
    return pl.layers_map_html(info["neighborhoods"], edges=layers["edges"], id_col=id_col)


def _record(info):
    return {k: info[k] for k in ("streets_path", "neighborhoods_path", "id_col", "summary", "note",
                                 "n_neighborhoods", "n_segments", "columns", "crs")}


# ---- choosing where each layer comes from ---------------------------------------------------------------
def _save_upload(kind, key):
    """Button callback: save the uploaded files into data/ and select the layer (runs before the rerun)."""
    files = st.session_state.get(f"{key}_upload") or []
    try:
        path, message = pl.save_uploaded_layer(kind, [(f.name, f.getvalue()) for f in files])
    except ValueError as e:
        st.session_state[f"{key}_msg"] = ("error", str(e))
        return
    st.session_state.setdefault("paths", {})[kind] = path
    st.session_state[f"{key}_choice"] = path
    st.session_state[f"{key}_source"] = SOURCES[0]
    st.session_state[f"{key}_msg"] = ("success", message)
    st.session_state.run["inputs"] = None


def _layer_picker(label, kind, found, current):
    """Where the layer comes from: a layer of the data folder, or files uploaded from this computer (then saved there)."""
    key = kind
    folder = pl.LAYER_DIRS[kind]
    st.markdown(f"**{label}**")
    source = st.radio(f"{label}: source", SOURCES, horizontal=True, key=f"{key}_source",
                      label_visibility="collapsed")
    msg = st.session_state.pop(f"{key}_msg", None)
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])
    if source == SOURCES[0]:
        if not found:
            st.warning(f"No layer in `data/{folder}/` yet: choose *This computer* to add one.")
            return None
        choice_key = f"{key}_choice"
        if st.session_state.get(choice_key) not in found:
            st.session_state[choice_key] = current if current in found else found[0]
        return st.selectbox(f"{label}: layer", found, key=choice_key, label_visibility="collapsed",
                            format_func=lambda p: f"{Path(p).name}  ·  data/{folder}/")
    files = st.file_uploader("Files from this computer", accept_multiple_files=True, key=f"{key}_upload",
                             type=LAYER_TYPES, help=UPLOAD_HELP)
    st.button("Save into the data folder", key=f"{key}_save", disabled=not files, on_click=_save_upload,
              args=(kind, key), help=UPLOAD_HELP)
    return None


def _match_crs(streets, nbhd, mtimes):
    """Button callback: write the neighborhoods reprojected to the streets CRS and select that copy."""
    try:
        path = pl.write_matched_neighborhoods(_layers(streets, nbhd, mtimes))
    except Exception as e:
        st.session_state["neighborhoods_msg"] = ("error", f"Could not write the reprojected copy: {e}")
        return
    st.session_state.setdefault("paths", {})["neighborhoods"] = path
    st.session_state["neighborhoods_choice"] = path
    st.session_state["neighborhoods_source"] = SOURCES[0]
    st.session_state["neighborhoods_msg"] = ("success", f"Saved {Path(path).name} in data/neighborhoods/ and selected it.")
    st.session_state.run["inputs"] = None


# ---- the two dataset panels ----------------------------------------------------------------------------------
def _fmt(x, digits=0):
    return f"{x:,.{digits}f}"


def _hist_chart(h, x_title, y_title):
    """A single-series histogram from pre-binned counts: one hue, a surface gap between bars, values on hover."""
    if not h["count"]:
        return None
    width = (h["end"][0] - h["start"][0]) if len(h["start"]) else 1.0
    df = pd.DataFrame({"start": h["start"], "end": [e - 0.06 * width for e in h["end"]], "count": h["count"],
                       "to": h["end"]})
    return (alt.Chart(df)
            .mark_bar(color=BAR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(x=alt.X("start:Q", title=x_title, axis=alt.Axis(grid=False, format="~s", labelColor=INK_2, titleColor=INK_2)),
                    x2="end:Q",
                    y=alt.Y("count:Q", title=None, axis=alt.Axis(tickCount=4, format="~s", gridColor="#e9e9e6",
                                                                   domain=False, labelColor=INK_2, titleColor=INK_2)),
                    tooltip=[alt.Tooltip("start:Q", title="from", format=",.1f"),
                             alt.Tooltip("to:Q", title="to", format=",.1f"),
                             alt.Tooltip("count:Q", title=y_title, format=",")])
            .properties(height=150)
            .configure_view(strokeWidth=0))


def _tiles(items):
    """Stat tiles (label, value, hover note): a 2-column grid, the value semibold, the note on hover."""
    cells = "".join(f'<div class="tile" title="{html.escape(note or "")}"><div class="lbl">{html.escape(label)}</div>'
                    f'<div class="val">{html.escape(value)}</div></div>' for label, value, note in items)
    st.markdown(f'<div class="tiles">{cells}</div>', unsafe_allow_html=True)


def _panel(title, tiles, chart, clipped_note, notes):
    """One dataset card: four stat tiles in a 2x2 grid beside the histogram, then a line of notes."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        left, right = st.columns([1, 1.6], gap="medium")
        with left:
            _tiles(tiles)
        with right:
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
                if clipped_note:
                    st.caption(clipped_note)
        st.caption(" · ".join(notes))


def _plural(n, word):
    return f"{_fmt(n)} {word}{'' if n == 1 else 's'}"


def _streets_panel(p):
    h = p["hist"]
    tiles = [("Segments", _fmt(p["n"]), None),
             ("Total length", f"{_fmt(p['total_mi'], 1)} mi", "sum of the segment lengths"),
             ("Median segment", f"{_fmt(p['median_ft'])} ft", f"mean {_fmt(p['mean_ft'])} ft; 90% of the segments are shorter than {_fmt(p['p90_ft'])} ft"),
             ("Longest segment", f"{_fmt(p['max_ft'])} ft", None)]
    note = (f"Lengths up to the 99th percentile ({_fmt(h['upper'])} ft); {_plural(h['clipped'], 'longer segment')} not drawn."
            if h["clipped"] else None)
    notes = [f"CRS {p['crs']}", _plural(len(p["columns"]), "attribute column")]
    if p["multipart"]:
        notes.append(_plural(p["multipart"], "multi-part segment") + " (split at the nodes by the pipeline)")
    if p["empty"]:
        notes.append(_plural(p["empty"], "empty geometry") + " (ignored)")
    _panel("Street centerlines", tiles, _hist_chart(h, "segment length (ft)", "segments"), note, notes)


def _neighborhoods_panel(p):
    h = p["hist"]
    tiles = [("Polygons", _fmt(p["n"]), None),
             ("Total area", f"{_fmt(p['total_sqmi'], 1)} sq mi", None),
             ("Median area", f"{_fmt(p['median_acre'])} acres", f"smallest {_fmt(p['min_acre'])}, largest {_fmt(p['max_acre'])} acres"),
             ("Numeric columns", _fmt(len(p["numeric"])),
              "The columns the dependent variable and the controls are chosen from on the Stage 1 page: "
              + (", ".join(p["numeric"]) or "none"))]
    note = (f"Areas up to the 99th percentile ({_fmt(h['upper'])} acres); {_plural(h['clipped'], 'larger polygon')} not drawn."
            if h["clipped"] else None)
    notes = [f"identifier `{p['id_col']}`", f"CRS {p['crs']}"]
    if p["other"]:
        notes.append(_plural(len(p["other"]), "non-numeric column") + f" ({', '.join(p['other'][:6])}"
                     + (", …" if len(p["other"]) > 6 else "") + ")")
    if p["multipart"]:
        notes.append(_plural(p["multipart"], "multi-part polygon"))
    if p["missing_cells"]:
        notes.append(f"⚠️ {_plural(p['missing_cells'], 'missing value')} in the numeric columns (a variable with gaps "
                     "stops the component analysis in stage 2 unless they are handled first)")
    _panel("Neighborhoods", tiles, _hist_chart(h, "neighborhood area (acres)", "neighborhoods"), note, notes)


# ---- restoring the inputs after a browser refresh -----------------------------------------------------------
def _initial_paths():
    """The layers the pipeline last ran with (user_selection.json) when their files are still there, else the defaults."""
    try:
        sel = pl.load_selection()
        if os.path.exists(sel["streets_path"]) and os.path.exists(sel["neighborhoods_path"]):
            return dict(streets=sel["streets_path"], neighborhoods=sel["neighborhoods_path"], id_col=pl.id_column(sel))
    except Exception:
        pass
    return dict(streets=pl.DEFAULTS["streets_path"], neighborhoods=pl.DEFAULTS["neighborhoods_path"])


def restore_inputs():
    """
    After a browser refresh the session is new: take the inputs back from the saved selection
    (user_selection.json) when its files are still there and valid. Returns the inputs record or None.
    """
    run = st.session_state.run
    if run.get("inputs"):
        return run["inputs"]
    try:
        sel = pl.load_selection()
    except Exception:
        return None
    streets, nbhd, id_col = sel["streets_path"], sel["neighborhoods_path"], pl.id_column(sel)
    if not (os.path.exists(streets) and os.path.exists(nbhd)):
        return None
    try:
        info = pl.check_inputs(_layers(streets, nbhd, _mtimes(streets, nbhd)), id_col)
    except Exception:
        return None
    if info["problems"]:
        return None
    st.session_state.setdefault("paths", {}).update(streets=streets, neighborhoods=nbhd, id_col=id_col)
    run["inputs"] = _record(info)
    return run["inputs"]


# ---- the page -------------------------------------------------------------------------------------------------
def render():
    st.title("1 · Inputs")
    st.markdown(
        "The pipeline starts from a **cleaned street centerline layer** and **neighborhood polygons** with a "
        "column that identifies each neighborhood, a crime-count column and numeric control variables. The "
        "streets must be in a projected coordinate reference system in US feet (a State Plane zone, say). "
        "Pick each layer from the data folder of the repository, or add one from this computer.")

    found = pl.discover_shapefiles()
    if "paths" not in st.session_state:
        st.session_state.paths = _initial_paths()
    paths = st.session_state.paths
    c1, c2 = st.columns(2, gap="large")
    with c1:
        streets = _layer_picker("Street centerlines", "streets", found["streets"], paths["streets"])
    with c2:
        nbhd = _layer_picker("Neighborhoods", "neighborhoods", found["neighborhoods"], paths["neighborhoods"])
    run = st.session_state.run
    if not (streets and nbhd):
        run["inputs"] = None
        return
    paths.update(streets=streets, neighborhoods=nbhd)

    mtimes = _mtimes(streets, nbhd)
    try:
        layers = _layers(streets, nbhd, mtimes)
    except Exception as e:
        run["inputs"] = None
        st.error(f"Cannot read the layers: {e}")
        return

    # ---- the identifier column
    nb = layers["neighborhoods"]
    columns = [c for c in nb.columns if c != "geometry"]
    candidates = pl.id_candidates(nb)
    default = pl.default_id_column(nb)
    remembered = paths.get("id_col")
    if "id_choice" not in st.session_state or st.session_state.id_choice not in columns:
        st.session_state.id_choice = remembered if remembered in columns else (default or columns[0])
    k1, k2 = st.columns([1, 1.6], gap="large")
    with k1:
        id_col = st.selectbox("Neighborhood identifier", columns, key="id_choice",
                              help="The column with a distinct, non-missing value in every polygon. It keys the "
                                   "checkpoints of stage 1 and is kept, under its own name, in every output table.")
    paths["id_col"] = id_col
    info = pl.check_inputs(layers, id_col)
    if info["problems"]:
        run["inputs"] = None
        with k2:
            st.error("The layers cannot be processed as they are:\n\n" + "\n".join(f"- {p}" for p in info["problems"]))
            if candidates and id_col not in candidates:
                st.caption("Columns that identify every polygon uniquely: " + ", ".join(f"`{c}`" for c in candidates))
        return

    # ---- the coordinate reference systems
    crs = info["crs"]
    with k2:
        if crs["relation"] == "different":
            st.warning(f"The two layers use different coordinate reference systems: streets **{crs['streets']}**, "
                       f"neighborhoods **{crs['neighborhoods']}**. The measures are taken in the streets' system. "
                       "Match them now, or carry on: the run then reprojects the neighborhoods on the fly and the "
                       "files on disk stay as they are.")
            st.button("Reproject the neighborhoods to the streets' CRS (saves a copy in the data folder)",
                      on_click=_match_crs, args=(streets, nbhd, mtimes), key="match_crs")
        else:
            st.success(f"Both layers in {crs['streets']}.")
    run["inputs"] = _record(info)

    # ---- a look at each dataset
    profile = _profile(streets, nbhd, mtimes, id_col)
    st.markdown(TILE_CSS, unsafe_allow_html=True)
    _streets_panel(profile["streets"])
    _neighborhoods_panel(profile["neighborhoods"])

    # ---- the map
    st.markdown("#### Map")
    st.caption("Click a neighborhood or a street segment for its attributes. The layers and the basemap are chosen "
               "in the control at the top right; the button at the top left opens the map full screen. "
               "The streets are drawn simplified for speed.")
    components.html(_map_html(streets, nbhd, mtimes, id_col), height=620)
    st.page_link(st.session_state.pages["stage1"], label="Next: Stage 1, the street-network measures →")
