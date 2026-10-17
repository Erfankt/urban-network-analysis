"""
Web interface of the pipeline: the same three stages as the terminal scripts, with the
choices made by clicking, on one page of four sections, and an assistant beside it.

    streamlit run app/streamlit_app.py        (from the repository folder, in the activated venv)

The sections call app/pipeline.py, which calls the functions in src/; nothing is reimplemented.
The assistant (app/chat.py) answers with a local Ollama model by default or with a Gemini or
Groq key from .env, and the app works without it.
"""
import atexit
import sys
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Street network × property crime: pipeline", page_icon="🛣️", layout="wide")

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import theme                                           # noqa: E402
import setup_env                                       # noqa: E402  (standard library and Streamlit only)

# Stage 0: the environment. Only when it is complete are the heavy packages imported, so that a missing one
# shows the setup step instead of a traceback.
theme.inject()
env = setup_env.environment()
import_error = None
if setup_env.ready(env):
    try:
        import pipeline as pl                          # noqa: E402  (also switches the working directory to src/)
        import chat_panel                              # noqa: E402
        from views import inputs, stage1, stage2, final    # noqa: E402
    except Exception as e:                             # a package that is installed but does not load
        import_error = e
if import_error is not None or not setup_env.ready(env):
    with st.sidebar:
        st.markdown(theme.sidebar_head(), unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 3, 1])
    with middle:
        theme.title_bar()
        with st.container(key="sec-setup"):
            st.markdown(theme.section_head(0, "Setup", "setup", "warn", "Needs setup"), unsafe_allow_html=True)
            if import_error is not None:
                st.error(f"The packages are listed but the app could not load them: {import_error}. "
                         "Installing them again usually fixes it.")
                env["needs"] = [dict(r, status="reinstall") for r in env["rows"] if not r["optional"]]
            setup_env.render(env)
        theme.footer()
    st.stop()

if "run" not in st.session_state:
    st.session_state.run = {"inputs": None, "stage1": None, "stage2": None, "final": None}
    st.session_state.chat = []

STEPS = [("inputs", "Inputs", inputs), ("stage-1", "Stage 1: network measures", stage1),
         ("stage-2", "Stage 2: component analysis", stage2), ("final-dataset", "Final dataset", final)]
STAGE1_STATE = {"finished": ("done", "Finished"), "running": ("run", "Running…"), "failed": ("fail", "Failed"),
                "stopped": ("warn", "Stopped, can resume"), "none": ("wait", "Not started")}


def step_states():
    """(anchor, title, state, label) of the four steps, from what exists on disk and in the session."""
    run = st.session_state.run
    s1 = pl.stage1_status()["state"]
    measures = pl.measures_available()
    present, _ = pl.final_outputs()
    built = bool(present["csv"] and present["dictionary"])
    st1_state, st1_label = STAGE1_STATE[s1]
    fresh = measures and (not pl.LAUNCH_TIME or pl.measures_mtime() >= pl.LAUNCH_TIME or st.session_state.get("s2_started"))
    if s1 == "finished" and not measures:
        st1_state, st1_label = "warn", "Finished, measures do not match"
    return [
        ("setup", "Setup", "done", "Environment ready"),
        ("inputs", "Inputs", *(("done", "Layers ready") if run.get("inputs") else ("wait", "Choose the layers"))),
        ("stage-1", "Stage 1: network measures", st1_state, st1_label),
        ("stage-2", "Stage 2: component analysis",
         *(("done", "Dataset built") if built else ("ready", "Ready to analyze") if fresh
           else ("wait", "Not started"))),
        ("final-dataset", "Final dataset", *(("done", "Ready to download") if built else ("wait", "Not built yet"))),
    ]


def current_step(states):
    """The first step not finished: the assistant's context and the highlighted step of the sidebar."""
    return next((a for a, _, state, _ in states if state != "done"), states[-1][0])


@st.cache_resource
def _clear_output_at_exit():
    """Once per server process: empty output/ when the app shuts down (Ctrl+C in the terminal)."""
    atexit.register(pl.clear_output_dir)
    return True


_clear_output_at_exit()


@st.cache_resource
def _launch_time():
    """Once per server process: the moment the app started. Stage 2 and the final dataset only show what was
    built since then, and output/ is emptied now too (covers a previous run that could not clean up at exit)."""
    pl.clear_output_dir()
    return time.time()


pl.LAUNCH_TIME = _launch_time()

theme.inject()
with st.sidebar:
    st.markdown(theme.sidebar_head(), unsafe_allow_html=True)
    stepper_slot = st.empty()

# the chat panel stays in view while the long page scrolls
st.markdown("<style>[data-testid='stColumn']:has(.st-key-chat-panel) "
            "{ position: sticky; top: 3.5rem; align-self: flex-start; }</style>", unsafe_allow_html=True)

main, side = st.columns([2.4, 1], gap="large")
with main:
    theme.title_bar()
    heads = []
    with st.container(key="sec-setup-ready"):             # Stage 0, a one-line note once the environment is ready
        heads.append(st.empty())
        setup_env.render(env)
    for anchor, _, view in STEPS:
        with st.container(key=f"sec-{anchor}"):
            heads.append(st.empty())          # the header is filled in after the section ran, with its final status
            view.render()
    theme.footer()

states = step_states()
current = current_step(states)
for i, (head, (anchor, title, state, label)) in enumerate(zip(heads, states), start=0):
    head.markdown(theme.section_head(i, title, anchor, state, label), unsafe_allow_html=True)
stepper_slot.markdown(theme.stepper(states, current), unsafe_allow_html=True)
STEP_NAMES = {"inputs": "Inputs", "stage-1": "Stage 1", "stage-2": "Stage 2", "final-dataset": "Final dataset"}
with side, st.container(key="chat-panel"):
    chat_panel.render(STEP_NAMES[current])
