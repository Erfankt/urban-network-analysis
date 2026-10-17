"""
Web interface of the pipeline: the same three stages as the terminal scripts, with the
choices made by clicking and an assistant beside every page.

    streamlit run app/streamlit_app.py        (from the repository folder, in the activated venv)

The pages call app/pipeline.py, which calls the functions in src/; nothing is reimplemented.
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

pages = {
    "inputs": st.Page(inputs.render, title="1 · Inputs", icon="📂", url_path="inputs", default=True),
    "stage1": st.Page(stage1.render, title="2 · Stage 1: network measures", icon="🛣️", url_path="stage1"),
    "stage2": st.Page(stage2.render, title="3 · Stage 2: component analysis", icon="📊", url_path="stage2"),
    "final": st.Page(final.render, title="4 · Final dataset", icon="📦", url_path="final"),
}
st.session_state.pages = pages
current = st.navigation(list(pages.values()))

with st.sidebar:
    st.markdown("**Street Network Configuration and Property Crime** — data pipeline")
    st.caption("Kefayat & Thill (2025), ISPRS IJGI 14(5):200")
    status = pl.stage1_status()
    st.caption(f"Stage 1: {stage1.STATE_LABEL[status['state']]}"
               + (" · measures ready" if pl.measures_available() else ""))
    st.caption(f"Repository: `{pl.REPO_DIR}`")

PAGE_NAMES = {"inputs": "Inputs", "stage1": "Stage 1", "stage2": "Stage 2", "final": "Final dataset"}
page_name = next((PAGE_NAMES[k] for k, p in pages.items() if p is current), "Inputs")

# the chat panel stays in view while a long page scrolls
st.markdown("<style>[data-testid='stColumn']:has(.st-key-chat-panel) "
            "{ position: sticky; top: 3.5rem; align-self: flex-start; }</style>", unsafe_allow_html=True)

main, side = st.columns([2.4, 1], gap="large")
with main:
    current.run()
with side, st.container(key="chat-panel"):
    chat_panel.render(page_name)
