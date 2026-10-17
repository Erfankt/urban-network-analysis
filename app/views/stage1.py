"""Page 2: stage 1, run_analysis.py in the background, with progress read from the checkpoints."""
import time
import streamlit as st

import pipeline as pl
from views.inputs import restore_inputs

STATE_LABEL = {"none": "not started", "running": "running", "finished": "finished",
               "failed": "failed", "stopped": "stopped before the end"}


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _buffer_state(prev):
    """The list of buffer distances being built (kept in the session, sorted and validated)."""
    if "buffers" not in st.session_state:
        st.session_state["buffers"] = list((prev or {}).get("buffer_list_mile", pl.DEFAULTS["buffer_list_mile"]))
    return st.session_state["buffers"]


def _add_buffer():
    """Callback: validate the distance typed in and append it to the list."""
    value = round(float(st.session_state["buffer_new"]), 2)
    current = st.session_state["buffers"]
    if value in current:
        st.session_state["buffer_msg"] = f"{value} mi is already in the list."
        return
    try:
        st.session_state["buffers"] = pl.parse_buffers(",".join(str(b) for b in current + [value]))
        st.session_state["buffer_msg"] = None
    except ValueError as e:
        st.session_state["buffer_msg"] = str(e)


def _remove_buffer(value):
    st.session_state["buffers"] = [b for b in st.session_state["buffers"] if b != value]
    st.session_state["buffer_msg"] = None


def _reset_buffers():
    st.session_state["buffers"] = list(pl.DEFAULTS["buffer_list_mile"])
    st.session_state["buffer_msg"] = None


def _set_controls(options, value):
    for c in options:
        st.session_state[f"ctrl_{c}"] = value


def _settings(inputs, prev):
    """The stage-1 choices (buffers, crime variable, controls); returns a selection dict or None."""
    # ---- buffer distances: type one, add it, the list grows
    buffers = _buffer_state(prev)
    st.markdown("**Buffer distances around each neighborhood (miles)**",
                help="Type a distance and press Add. Distances are positive and distinct; those of 1 mile or more "
                     "are multiples of 0.1. The paper's five bands are the default.")
    c1, c2, c3 = st.columns([2, 1, 2], vertical_alignment="bottom")
    c1.number_input("Distance to add", min_value=0.01, max_value=50.0, value=0.25, step=0.05, format="%.2f",
                    key="buffer_new")
    c2.button("Add", on_click=_add_buffer, use_container_width=True)
    c3.button("Reset to the paper's bands", on_click=_reset_buffers, type="tertiary")
    if st.session_state.get("buffer_msg"):
        st.error(st.session_state["buffer_msg"])
    if buffers:
        cols = st.columns(min(len(buffers), 6) + 1)
        for i, b in enumerate(buffers):
            cols[i % (len(cols) - 1)].button(f"{b} mi  ✕", key=f"buf_rm_{b}", on_click=_remove_buffer, args=(b,),
                                             help="Remove this distance")
    else:
        st.warning("Add at least one distance.")

    # ---- the crime (dependent) variable
    columns = inputs["columns"]
    prev_crime = (prev or {}).get("crime_col")
    crime = st.selectbox("Crime variable (dependent variable)", columns,
                         index=columns.index(prev_crime) if prev_crime in columns else 0)

    # ---- control variables: tick the columns
    options = [c for c in columns if c != crime]
    prev_ctrl = [c for c in (prev or {}).get("ctrl_vars", [])]
    for c in options:
        st.session_state.setdefault(f"ctrl_{c}", c in prev_ctrl)
    st.markdown("**Control variables**", help="The socio-spatial variables reduced in stage 2; the paper used 16.")
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Select all", on_click=_set_controls, args=(options, True), use_container_width=True)
    b2.button("Clear", on_click=_set_controls, args=(options, False), use_container_width=True)
    grid = st.columns(4)
    for i, c in enumerate(options):
        grid[i % 4].checkbox(c, key=f"ctrl_{c}")
    controls = [c for c in options if st.session_state.get(f"ctrl_{c}")]
    st.caption(f"{len(controls)} of {len(options)} selected.")
    if not buffers or not controls:
        return None
    return {"streets_path": inputs["streets_path"], "neighborhoods_path": inputs["neighborhoods_path"],
            "id_col": inputs["id_col"], "buffer_list_mile": list(buffers), "crime_col": crime, "ctrl_vars": controls}


def _progress_panel(status):
    run = status["run"]
    state = status["state"]
    sel = run["selection"]
    st.markdown(f"**Run:** {STATE_LABEL[state]} · elapsed {_elapsed(status['elapsed'])} · "
                f"buffers {', '.join(str(b) for b in sel['buffer_list_mile'])} · identifier `{pl.id_column(sel)}` · "
                f"dependent variable `{sel['crime_col']}` · {len(sel['ctrl_vars'])} controls")
    total_done = total_all = 0
    for b, (done, total) in status["progress"].items():
        total_done, total_all = total_done + done, total_all + total
        label = f"{b} mi: {done} of {total} neighborhoods" if total else f"{b} mi: {done} neighborhoods"
        st.progress(done / total if total else 0.0, text=label)
    if total_all:
        st.caption(f"Overall: {total_done} of {total_all} neighborhood-buffers measured "
                   f"({100 * total_done / total_all:.0f}%). The larger buffers take disproportionately longer.")
    with st.expander("Log", expanded=(state != "finished")):
        st.code("\n".join(status["log_tail"]) or "(empty)", language=None)


def render():
    st.title("2 · Stage 1: street-network measures")
    st.markdown(
        "Builds the street network (intersections and dead ends as nodes), buffers every neighborhood by the "
        "chosen distances, keeps the largest connected component inside each buffer and measures it with the "
        "eight metrics. The run is `run_analysis.py` started in the background with all answers given, so it "
        "leaves the same files as a terminal run and its checkpoints let it resume if it stops.")

    run = st.session_state.run
    inputs = run.get("inputs") or restore_inputs()
    status = pl.stage1_status()
    running = status["state"] == "running"
    prev = (status["run"] or {}).get("selection")
    if prev is None:                                   # no app run yet: offer the choices of the last terminal run
        try:
            prev = pl.load_selection()
        except Exception:
            prev = None

    if inputs is None and status["state"] == "none":
        st.info("Check the inputs on page 1 first.")
        return

    # ---- settings and the Run button
    if inputs is None:
        st.info("To start a new run, check the inputs on page 1 first. The previous run is shown below.")
        sel = None
    else:
        st.subheader("Settings")
        sel = _settings(inputs, prev)
        recompute = st.checkbox("Recompute: ignore the checkpoints of these inputs and measure everything again",
                                value=False, disabled=running)
        if st.button("Run stage 1", type="primary", disabled=running or sel is None):
            try:
                pl.start_stage1(sel, recompute=recompute, n_neighborhoods=inputs["n_neighborhoods"])
            except Exception as e:
                st.error(f"Could not start the run: {e}")
            else:
                time.sleep(1.0)
                st.rerun()

    # ---- progress, refreshed every 2 s while the process runs
    if status["state"] == "none":
        return
    st.subheader("Progress")

    @st.fragment(run_every=2 if running else None)
    def panel():
        status = pl.stage1_status()
        _progress_panel(status)
        state = status["state"]
        if state == "running":
            if st.button("Stop the run", help="The neighborhoods measured so far stay in the checkpoints."):
                pl.stop_stage1()
                st.rerun()
        elif state in ("stopped", "failed"):
            msg = ("The run stopped before the end." if state == "stopped"
                   else "The run failed; the log above says why.")
            st.warning(f"{msg} The neighborhoods measured so far are in the checkpoints: resume to continue from there.")
            if st.button("Resume the run", type="primary"):
                pl.start_stage1(status["run"]["selection"], recompute=False,
                                n_neighborhoods=status["run"].get("n_neighborhoods"))
                st.rerun()
        elif state == "finished":
            if pl.measures_available():
                st.success("Stage 1 complete: `data/final_dataset/extracted_measures.shp` holds the neighborhoods "
                           "with one column per measure and buffer.")
                st.page_link(st.session_state.pages["stage2"], label="Next: Stage 2, the component analysis →")
            else:
                st.warning("The run finished but extracted_measures.shp does not match the saved selection; run again.")
        if running and state != "running":
            st.rerun()                         # leave the auto-refresh once the process has ended

    panel()

    # ---- what the assistant is told
    status = pl.stage1_status()
    if status["run"]:
        run["stage1"] = {"selection": status["run"]["selection"], "status": STATE_LABEL[status["state"]],
                         "progress": {f"{b} mi": f"{d}/{t}" for b, (d, t) in status["progress"].items()}}
