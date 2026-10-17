"""
Web interface of the pipeline: the same three stages as the terminal scripts, with the
choices made by clicking, on one page of four sections, and an assistant beside it.

    streamlit run app/streamlit_app.py        (from the repository folder, in the activated venv)

The sections call app/pipeline.py, which calls the functions in src/; nothing is reimplemented.
The assistant (app/chat.py) answers with a local Ollama model by default or with a Gemini or
Groq key from .env, and the app works without it.
"""
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

STEPS = [("inputs", "1 · Inputs", inputs), ("stage-1", "2 · Stage 1: network measures", stage1),
         ("stage-2", "3 · Stage 2: component analysis", stage2),
         ("final-dataset", "4 · Final dataset", final)]


def current_step():
    """The step the user is at (the first one not finished), for the assistant's suggestions and context."""
    if not st.session_state.run.get("inputs"):
        return "Inputs"
    if stage1_state() != "finished" or not pl.measures_available():
        return "Stage 1"
    present, _ = pl.final_outputs()
    return "Final dataset" if present["csv"] and present["dictionary"] else "Stage 2"


def stage1_state():
    return pl.stage1_status()["state"]


with st.sidebar:
    st.markdown("**Street Network Configuration and Property Crime** — data pipeline")
    st.caption("Kefayat & Thill (2025), ISPRS IJGI 14(5):200")
    st.markdown("\n".join(f"- [{title}](#{anchor})" for anchor, title, _ in STEPS))
    status = pl.stage1_status()
    st.caption(f"Stage 1: {stage1.STATE_LABEL[status['state']]}"
               + (" · measures ready" if pl.measures_available() else ""))
    st.caption(f"Repository: `{pl.REPO_DIR}`")

# the chat panel stays in view while the long page scrolls
st.markdown("<style>[data-testid='stColumn']:has(.st-key-chat-panel) "
            "{ position: sticky; top: 3.5rem; align-self: flex-start; }</style>", unsafe_allow_html=True)

main, side = st.columns([2.4, 1], gap="large")
with main:
    st.title("Street network configuration and property crime")
    st.caption("The data pipeline of Kefayat & Thill (2025), step by step: choose the inputs, measure the street "
               "networks, reduce the variables, and download the final dataset.")
    for i, (_, _, view) in enumerate(STEPS):
        if i:
            st.divider()
        view.render()
with side, st.container(key="chat-panel"):
    chat_panel.render(current_step())
