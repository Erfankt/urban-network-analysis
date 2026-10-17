"""The chat panel shown beside every page: history, suggested questions, the input box."""
import time
import streamlit as st

from chat import load_config, check_provider, stream_reply, system_message, SUGGESTIONS

SETUP_NOTE = """**No model is answering yet.** The app works without it: every step is done in the pages.
To turn the assistant on, either install [Ollama](https://ollama.com), run `ollama pull llama3.2`
once and reload, or put a Gemini or Groq API key in a `.env` file next to `README.md`
(see `.env.example`)."""


def provider_status(force=False):
    """The provider configuration and whether it answers, re-checked every 30 s or on demand."""
    cached = st.session_state.get("provider_status")
    if not force and cached and time.time() - cached["at"] < 30:
        return cached
    cfg = load_config()
    ok, message = check_provider(cfg)
    st.session_state.provider_status = {"cfg": cfg, "ok": ok, "message": message, "at": time.time()}
    return st.session_state.provider_status


def render(page):
    status = provider_status()
    cfg, ok = status["cfg"], status["ok"]

    head, tail = st.columns([3, 1])
    head.markdown("#### Assistant")
    if tail.button("Clear", help="forget this conversation", use_container_width=True):
        st.session_state.chat = []
        st.rerun()
    if ok:
        st.caption(f"✅ {status['message']}")
    else:
        st.caption("⚠️ no model connected")

    history = st.session_state.setdefault("chat", [])
    box = st.container(height=360)

    question = None
    if ok:
        for i, q in enumerate(SUGGESTIONS.get(page, [])):
            if st.button(q, key=f"sugg_{page}_{i}", use_container_width=True):
                question = q
        typed = st.chat_input("Ask about this step…", key="chat_input")
        if typed:
            question = typed
    else:
        st.info(SETUP_NOTE)
        if st.button("Check again", use_container_width=True):
            provider_status(force=True)
            st.rerun()

    if question:
        history.append({"role": "user", "content": question})

    with box:
        if not history:
            st.caption("The assistant sees the inputs, the stage-1 status and every table of the "
                       "component analysis as you go, and can explain or advise on each step. "
                       "It never changes anything: every choice stays yours, in the pages.")
        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if question:
            system = system_message(st.session_state.run, page)
            messages = [{"role": m["role"], "content": m["content"]} for m in history]
            with st.chat_message("assistant"):
                try:
                    reply = st.write_stream(stream_reply(cfg, system, messages))
                except Exception as e:                       # network, model or key problems
                    reply = f"The model could not answer: {e}"
                    st.error(reply)
            history.append({"role": "assistant", "content": reply if isinstance(reply, str) else str(reply)})
