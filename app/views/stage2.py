"""Section 3: stage 2, the component analysis of both variable blocks and the final dataset."""
import streamlit as st

import pipeline as pl
from common import rules_of_thumb_text, solution_summary_text, candidate_text

BLOCKS = {"network": "street network", "controls": "socio-spatial controls"}


# ---------------------------------------------------------------------------
# cached computations (keyed by the measures file so a new stage-1 run invalidates them)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the measures…")
def _measures(mtime):
    return pl.load_measures()


@st.cache_resource(show_spinner=False)
def _overview(mtime, variables):
    return pl.block_overview(_measures(mtime)[0], list(variables))


@st.cache_data(show_spinner="Drawing the overview…")
def _overview_pngs(mtime, variables, label):
    return pl.overview_pngs(_overview(mtime, variables), label)


@st.cache_resource(show_spinner="Fitting the solution…")
def _solution(mtime, variables, n, method):
    return pl.block_solution(_measures(mtime)[0], list(variables), n, method)


@st.cache_data(show_spinner="Drawing the loadings…")
def _solution_pngs(mtime, variables, n, method):
    return pl.solution_pngs(_solution(mtime, variables, n, method))


@st.cache_data(show_spinner=False)
def _distribution_png(mtime, col):
    return pl.figure_png(pl.distribution_figure(_measures(mtime)[0], col))


def _fresh_block():
    return {"inspected": [], "view": None, "retained": None, "rep_picks": {}, "representatives": None}


# ---------------------------------------------------------------------------
# one variable block
# ---------------------------------------------------------------------------

def _report_view(report, pngs, label):
    var_tab = report["variance_table"]
    st.markdown(f"**Variance explained by the rotated components** — total retained by "
                f"{report['n_factors']} component(s): **{var_tab['Cumulative %'].iloc[-1]:.1f}%**")
    st.dataframe(var_tab.round(3), use_container_width=True)
    st.code(solution_summary_text(report), language=None)
    t1, t2, t3, t4 = st.tabs(["Rotated loadings (heatmap)", "Unrotated loadings (heatmap)", "Loadings (tables)", "Communalities"])
    with t1:
        st.image(pngs["rotated"], use_container_width=True)
    with t2:
        st.image(pngs["unrotated"], use_container_width=True)
    with t3:
        st.markdown(f"Rotated ({report['rotation']}):")
        st.dataframe(report["rotated_loadings"].round(3), use_container_width=True)
        st.markdown("Unrotated:")
        st.dataframe(report["unrotated_loadings"].round(3), use_container_width=True)
    with t4:
        st.dataframe(report["communalities"].set_index("Variable").round(3).T, use_container_width=True)


def _representatives(key, report):
    """One dropdown per retained component; a measure cannot stand for two components."""
    state = st.session_state.s2[key]
    st.subheader("Representative measures: one original measure per component")
    st.caption("Rule of thumb: the highest absolute loading; when two are close, prefer the one with the smaller "
               "cross-loading and the clearer theoretical meaning. Each line shows the loading on the component "
               "and the measure's largest loading on any other component.")
    rot = report["rotated_loadings"]
    show_all = st.toggle("Show every measure, not only the five top loaders", value=False, key=f"all_{key}")
    picks = []
    for comp in rot.columns:
        cands = pl.candidates_for(report, comp, exclude=picks, top_k=len(rot) if show_all else 5)
        names = [c["measure"] for c in cands]
        labels = [candidate_text(c) for c in cands]
        stored = state["rep_picks"].get(comp)
        idx = names.index(stored) if stored in names else 0
        choice = st.selectbox(comp, labels, index=idx)
        pick = names[labels.index(choice)]
        state["rep_picks"][comp] = pick
        picks.append(pick)
    state["representatives"] = picks
    st.markdown("**Selected representatives:** " + ", ".join(f"`{p}`" for p in picks))
    st.dataframe(rot.loc[picks].round(3), use_container_width=True)


def _block(key, variables, mtime, method, representatives=False):
    label = BLOCKS[key]
    state = st.session_state.s2.setdefault(key, _fresh_block())
    p = len(variables)
    stats = _overview(mtime, tuple(variables))
    pngs = _overview_pngs(mtime, tuple(variables), label)

    st.subheader("Overview, independent of the number of components")
    c1, c2 = st.columns(2)
    c1.image(pngs["correlation"], use_container_width=True)
    c2.image(pngs["scree"], use_container_width=True)
    with st.expander("Eigenvalue table", expanded=False):
        st.dataframe(stats["table"].round(3), use_container_width=True, height=min(40 + 35 * p, 420))
    st.info(f"Rules of thumb: {rules_of_thumb_text(stats['rules'])}.")

    st.subheader("Inspect a candidate solution")
    c1, c2 = st.columns([1, 3])
    default_n = state["view"] or stats["rules"]["kaiser"] or 1
    n = c1.number_input("Number of components", 1, p, value=min(default_n, p), key=f"n_{key}")
    if c2.button(f"Inspect the {n}-component solution", key=f"inspect_{key}"):
        if n not in state["inspected"]:
            state["inspected"].append(int(n))
        state["view"] = int(n)

    if state["inspected"]:
        options = sorted(state["inspected"])
        view = st.radio("Solutions inspected", options, index=options.index(state["view"]) if state["view"] in options else 0,
                        horizontal=True, key=f"view_{key}", format_func=lambda k: f"{k} components")
        state["view"] = view
        try:
            report = _solution(mtime, tuple(variables), view, method)
            rpngs = _solution_pngs(mtime, tuple(variables), view, method)
        except Exception as e:
            st.error(f"Could not fit a {view}-component solution with `{method}`: {e}")
            state["inspected"].remove(view)
            report = None
        if report is not None:
            _report_view(report, rpngs, label)
            retained = state["retained"]
            if retained != view and st.button(f"Retain {view} component(s)", type="primary", key=f"retain_{key}"):
                state["retained"], state["rep_picks"], state["representatives"] = view, {}, None
                st.rerun()

    if state["retained"]:
        st.success(f"Retained: {state['retained']} component(s) ({label}, {method}, varimax).")
        if representatives:
            _representatives(key, _solution(mtime, tuple(variables), state["retained"], method))


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def _chat_block_state(key, variables, mtime, method):
    """What the assistant is told about one block."""
    state = st.session_state.s2.get(key) or _fresh_block()
    stats = _overview(mtime, tuple(variables))
    info = {"variables": list(variables), "rules": rules_of_thumb_text(stats["rules"]), "table": stats["table"],
            "inspected": {}, "retained": state["retained"], "representatives": state["representatives"], "candidates": None}
    for n in state["inspected"]:
        try:
            r = _solution(mtime, tuple(variables), n, method)
        except Exception:
            continue
        info["inspected"][n] = {"retained": float(r["variance_table"]["Cumulative %"].iloc[-1]),
                                "summary": solution_summary_text(r)}
    if key == "network" and state["retained"]:
        r = _solution(mtime, tuple(variables), state["retained"], method)
        info["candidates"] = {comp: pl.candidates_for(r, comp, top_k=5) for comp in r["rotated_loadings"].columns}
    return info


def render():
    if not pl.measures_available():
        st.info("Stage 1 has not produced `extracted_measures.shp` for the saved selection yet: run stage 1 in section 2 first.")
        return
    mtime = pl.measures_mtime()
    gdf, sel, net = _measures(mtime)
    ctrl = sel["ctrl_vars"]
    st.markdown(f"**{len(gdf)} neighborhoods** · {len(net)} street-network measures "
                f"({len(sel['buffer_list_mile'])} buffers × 8 metrics) · {len(ctrl)} control variables · "
                f"dependent variable `{sel['crime_col']}`")

    # ---- missing values stop everything, as in the terminal
    m = pl.check_missing(gdf, net, ctrl, sel["id_col"])
    if not m["counts"].empty:
        st.error("The component analysis cannot run on missing values (the correlation matrix is undefined).")
        st.dataframe(m["counts"].rename("missing values").to_frame(), use_container_width=False)
        st.markdown("**Neighborhoods affected:** " + "; ".join(f"{sel['id_col']} {i}: {', '.join(v)}" for i, v in m["rows"]))
        for line in m["advice"]:
            st.markdown(f"- {line}")
        return

    # ---- settings
    s2 = st.session_state.setdefault("s2", {"method": "principal", "mtime": mtime})
    if s2.get("mtime") != mtime:                       # a new stage-1 run: start the choices over
        st.session_state.s2 = s2 = {"method": s2.get("method", "principal"), "mtime": mtime}
    methods = list(pl.PCA_METHODS)
    method = st.selectbox("Extraction method", methods, index=methods.index(s2["method"]),
                          format_func=lambda k: f"{k} — {pl.PCA_METHODS[k]}")
    if method != s2["method"]:
        st.session_state.s2 = s2 = {"method": method, "mtime": mtime}
        st.rerun()
    st.caption("Rotation: varimax, as in the paper.")

    tab_net, tab_ctrl, tab_dist, tab_build = st.tabs(
        ["Street-network measures", "Socio-spatial controls", "Distributions", "Build the final dataset"])
    with tab_net:
        _block("network", net, mtime, method, representatives=True)
    with tab_ctrl:
        _block("controls", ctrl, mtime, method)
    with tab_dist:
        col = st.selectbox("Variable", net + ctrl)
        st.image(_distribution_png(mtime, col), use_container_width=True)
        st.caption("Histogram with the skewness, boxplot, and boxplot with the observations; the same figures "
                   "the terminal run saves as `dist_<variable>.png`.")
    with tab_build:
        _build(gdf, sel, net, ctrl, mtime, method)

    # ---- what the assistant is told
    st.session_state.run["stage2"] = {
        "method": method, "rotation": "varimax",
        "network": _chat_block_state("network", net, mtime, method),
        "controls": _chat_block_state("controls", ctrl, mtime, method)}


def _build(gdf, sel, net, ctrl, mtime, method):
    s2 = st.session_state.s2
    b_net, b_ctrl = s2.get("network") or _fresh_block(), s2.get("controls") or _fresh_block()
    ready = bool(b_net["retained"] and b_ctrl["retained"] and b_net["representatives"])
    st.markdown("The final dataset is written exactly as the terminal run writes it: `final_dataset.shp` / `.csv`, "
                "`final_dataset_dictionary.csv`, the choices in `user_selection.json`, and the figures in "
                "`output/pca_diagnostics/`.")
    st.markdown(f"- Method `{method}`, rotation varimax\n"
                f"- Street-network block: {b_net['retained'] or '—'} component(s); representatives: "
                f"{', '.join(b_net['representatives']) if b_net['representatives'] else '—'}\n"
                f"- Socio-spatial controls: {b_ctrl['retained'] or '—'} component(s)")
    if not ready:
        st.info("Retain a solution in both blocks and choose the representative measures first.")
    dist = st.checkbox("Also save the per-variable distribution figures (as the terminal run does)", value=True)
    if st.button("Build the final dataset", type="primary", disabled=not ready):
        bar = st.progress(0.0, text="starting")
        try:
            net_report = _solution(mtime, tuple(net), b_net["retained"], method)
            ctrl_report = _solution(mtime, tuple(ctrl), b_ctrl["retained"], method)
            out = pl.build_final_dataset(gdf, sel, method, "varimax", net_report, ctrl_report,
                                         b_net["representatives"], distribution_plots=dist,
                                         progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg))
        except Exception as e:
            st.error(f"The final dataset could not be written: {e}")
            return
        bar.empty()
        st.session_state.run["final"] = {"rows": len(out["final"]), "columns": [c for c in out["final"].columns if c != "geometry"]}
        st.success(f"{len(out['final'])} rows × {out['final'].shape[1] - 1} columns written to `{out['csv']}` "
                   f"(and the shapefile and dictionary beside it).")
        st.caption("The dataset is shown in section 4 below.")
