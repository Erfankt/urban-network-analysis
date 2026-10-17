"""Page 1: the two input layers, checked the way run_analysis.py checks them."""
import os
import streamlit as st
import streamlit.components.v1 as components

import pipeline as pl

OTHER = "Other path…"


def _mtimes(*paths):
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)


@st.cache_resource(show_spinner="Reading the layers…")
def _load(streets, nbhd, mtimes):
    return pl.load_inputs(streets, nbhd)


@st.cache_data(show_spinner="Drawing the map…")
def _map_html(streets, nbhd, mtimes):
    return pl.neighborhoods_map_html(_load(streets, nbhd, mtimes)["neighborhoods"])


def _path_picker(label, found, current, key):
    """A dropdown of the shapefiles found under data/, with a free path as the last option."""
    options = found + [OTHER]
    idx = found.index(current) if current in found else len(found)
    choice = st.selectbox(label, options, index=idx, key=f"{key}_choice")
    if choice == OTHER:
        return st.text_input(f"{label}: path", value=current if current not in found else "",
                             key=f"{key}_text", help="Relative to src/ (e.g. ../data/streetnetwork/streets.shp) or absolute.")
    return choice


def _save_upload(kind, key):
    """Button callback: save the uploaded files into data/ and select them in the picker (runs before the rerun)."""
    files = st.session_state.get(f"{key}_upload") or []
    try:
        path, message = pl.save_uploaded_layer(kind, [(f.name, f.getvalue()) for f in files])
    except ValueError as e:
        st.session_state["upload_msg"] = ("error", str(e))
        return
    st.session_state.setdefault("paths", {})[kind] = path
    st.session_state[f"{key}_choice"] = path
    st.session_state["upload_msg"] = ("success", message + " It is selected above: check the inputs to continue.")
    st.session_state.run["inputs"] = None


def _uploads():
    """Upload a layer from the computer; it is copied into data/ and then behaves like any other layer."""
    with st.expander("Use a layer from your computer"):
        st.markdown(
            "**Upload** the parts of a shapefile together (select the `.shp`, `.shx`, `.dbf` and `.prj` files at once), "
            "one `.zip` holding a shapefile, or a `.gpkg` / `.geojson`. The files are copied into `data/streetnetwork/` "
            "or `data/neighborhoods/` on this computer. For a big layer you can instead choose **Other path…** in the "
            "dropdown above and type its location, for example `C:\\Users\\you\\Documents\\streets.shp`; nothing is copied then.")
        c1, c2 = st.columns(2)
        for col, kind, key, label in ((c1, "streets", "streets", "Street centerlines"),
                                      (c2, "neighborhoods", "nbhd", "Neighborhoods")):
            with col:
                files = st.file_uploader(label, accept_multiple_files=True, key=f"{key}_upload",
                                         type=["shp", "shx", "dbf", "prj", "cpg", "sbn", "sbx", "qix", "zip", "gpkg", "geojson"])
                st.button(f"Save the {label.lower()} layer", key=f"{key}_save", disabled=not files,
                          on_click=_save_upload, args=(kind, key))
        msg = st.session_state.pop("upload_msg", None)
        if msg:
            (st.success if msg[0] == "success" else st.error)(msg[1])


def _record(streets, nbhd, info):
    return {"streets_path": streets, "neighborhoods_path": nbhd, "summary": info["summary"],
            "note": info["note"], "n_neighborhoods": info["n_neighborhoods"],
            "n_segments": info["n_segments"], "columns": info["columns"]}


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
    streets, nbhd = sel["streets_path"], sel["neighborhoods_path"]
    if not (os.path.exists(streets) and os.path.exists(nbhd)):
        return None
    try:
        info = _load(streets, nbhd, _mtimes(streets, nbhd))
    except Exception:
        return None
    if info["problems"]:
        return None
    st.session_state.setdefault("paths", {}).update(streets=streets, neighborhoods=nbhd)
    run["inputs"] = _record(streets, nbhd, info)
    return run["inputs"]


def render():
    st.title("1 · Inputs")
    st.markdown(
        "The pipeline starts from a **cleaned street centerline layer** and **neighborhood polygons** "
        "with a unique `id` column, a crime-count column and numeric control variables. The streets must be "
        "in a projected coordinate reference system in US feet (a State Plane zone, say); the neighborhoods "
        "are reprojected to it when they differ. Pick the two shapefiles and check them.")

    found = pl.discover_shapefiles()
    paths = st.session_state.setdefault("paths", dict(streets=pl.DEFAULTS["streets_path"],
                                                      neighborhoods=pl.DEFAULTS["neighborhoods_path"]))
    c1, c2 = st.columns(2)
    with c1:
        streets = _path_picker("Street centerlines (.shp)", found["streets"], paths["streets"], "streets")
    with c2:
        nbhd = _path_picker("Neighborhoods (.shp)", found["neighborhoods"], paths["neighborhoods"], "nbhd")
    paths.update(streets=streets, neighborhoods=nbhd)
    _uploads()

    run = st.session_state.run
    if st.button("Check the inputs", type="primary", disabled=not (streets and nbhd)):
        run["inputs"] = None
        try:
            info = _load(streets, nbhd, _mtimes(streets, nbhd))
        except Exception as e:
            st.error(f"Cannot read the layers: {e}")
            return
        if info["problems"]:
            st.error("The layers cannot be processed as they are:\n\n" + "\n".join(f"- {p}" for p in info["problems"]))
            return
        run["inputs"] = _record(streets, nbhd, info)

    inputs = run.get("inputs") or restore_inputs()
    if not inputs or inputs["streets_path"] != streets or inputs["neighborhoods_path"] != nbhd:
        if inputs:
            st.warning("The paths changed since the last check; check the inputs again.")
        return

    st.success(inputs["summary"])
    if inputs["note"]:
        st.info(f"Note: {inputs['note']}")

    info = _load(streets, nbhd, _mtimes(streets, nbhd))
    st.markdown(f"**Numeric columns** a variable can be chosen from ({len(info['columns'])}): "
                + (", ".join(f"`{c}`" for c in info["columns"]) or "none"))
    with st.expander("Summary statistics of the numeric columns"):
        st.dataframe(info["column_stats"].round(3), use_container_width=True)
    with st.expander("Map of the neighborhoods", expanded=True):
        components.html(_map_html(streets, nbhd, _mtimes(streets, nbhd)), height=440)
    st.page_link(st.session_state.pages["stage1"], label="Next: Stage 1, the street-network measures →")
