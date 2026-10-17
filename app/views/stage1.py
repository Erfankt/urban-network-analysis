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


def _settings(inputs, prev):
    """The stage-1 choices (buffers, dependent variable, controls); returns a selection dict or None."""
    default_buffers = ", ".join(str(b) for b in (prev or {}).get("buffer_list_mile", pl.DEFAULTS["buffer_list_mile"]))
    text = st.text_input("Buffer distances around each neighborhood, in miles (comma-separated)", value=default_buffers,
                         help="Positive, distinct, at most two decimals; the paper's five bands are the default.")
    buffers = None
    try:
        buffers = pl.parse_buffers(text)
        codes = pl.buffer_codes(buffers)
        st.caption("Column suffixes: " + ", ".join(f"{b} mi → `{c}`" for b, c in codes.items()))
    except ValueError as e:
        st.error(str(e))

    columns = inputs["columns"]
    prev_crime = (prev or {}).get("crime_col")
    crime = st.selectbox("Dependent variable (crime count)", columns,
                         index=columns.index(prev_crime) if prev_crime in columns else 0)
    options = [c for c in columns if c != crime]
    prev_ctrl = [c for c in (prev or {}).get("ctrl_vars", []) if c in options]
    controls = st.multiselect("Control variables", options, default=prev_ctrl,
                              help="The socio-spatial variables reduced in stage 2; the paper used 16.")
    if buffers is None or not controls:
        return None
    return {"streets_path": inputs["streets_path"], "neighborhoods_path": inputs["neighborhoods_path"],
            "id_col": inputs["id_col"], "buffer_list_mile": buffers, "crime_col": crime, "ctrl_vars": controls}


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
