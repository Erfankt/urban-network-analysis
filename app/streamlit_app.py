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
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Street network × property crime: pipeline", page_icon="🛣️", layout="wide")

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pipeline as pl                                  # noqa: E402  (also switches the working directory to src/)
import chat_panel                                      # noqa: E402
from views import inputs, stage1, stage2, final        # noqa: E402

if "run" not in st.session_state:
    st.session_state.run = {"inputs": None, "stage1": None, "stage2": None, "final": None}
    st.session_state.chat = []

import theme                                           # noqa: E402

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
    if s1 == "none" and measures:
        st1_state, st1_label = "done", "Measures from an earlier run"
    if s1 == "finished" and not measures:
        st1_state, st1_label = "warn", "Finished, measures do not match"
    return [
        ("inputs", "Inputs", *(("done", "Layers ready") if run.get("inputs") else ("wait", "Choose the layers"))),
        ("stage-1", "Stage 1: network measures", st1_state, st1_label),
        ("stage-2", "Stage 2: component analysis",
         *(("done", "Dataset built") if built else ("ready", "Ready to analyze") if measures
           else ("wait", "Waiting for stage 1"))),
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
    for anchor, _, view in STEPS:
        with st.container(key=f"sec-{anchor}"):
            heads.append(st.empty())          # the header is filled in after the section ran, with its final status
            view.render()
    theme.footer()

states = step_states()
current = current_step(states)
for i, (head, (anchor, title, state, label)) in enumerate(zip(heads, states), start=1):
    head.markdown(theme.section_head(i, title, anchor, state, label), unsafe_allow_html=True)
stepper_slot.markdown(theme.stepper(states, current), unsafe_allow_html=True)
STEP_NAMES = {"inputs": "Inputs", "stage-1": "Stage 1", "stage-2": "Stage 2", "final-dataset": "Final dataset"}
with side, st.container(key="chat-panel"):
    chat_panel.render(STEP_NAMES[current])
