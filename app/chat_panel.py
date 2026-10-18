"""The chat panel shown beside the page: history, suggested questions, the input box."""
import time
import streamlit as st

import pipeline as pl
from chat import (load_config, check_provider, agent_reply, system_message, SUGGESTIONS, save_settings, redact,
                  validate_key, discover_gemini_model, ollama_state, pull_model, KEY_PAGES, OLLAMA_DOWNLOAD, DEFAULT_MODELS)

def provider_status(force=False):
    """The provider configuration and whether it answers, re-checked every 30 s or on demand."""
    cached = st.session_state.get("provider_status")
    if not force and cached and time.time() - cached["at"] < 30:
        return cached
    cfg = load_config()
    ok, message = check_provider(cfg)
    st.session_state.provider_status = {"cfg": cfg, "ok": ok, "message": message, "at": time.time()}
    return st.session_state.provider_status


# ---- turning the assistant on: pick a provider and click, no files to edit -------------------------------------
def _connected():
    provider_status(force=True)
    st.session_state.show_setup = False
    st.rerun()


def _key_tab(provider):
    company, url = KEY_PAGES[provider]
    if provider == "gemini":
        st.markdown("**Connect Google Gemini with your own, free API key** — the recommended way. "
                    "Press the first button to open Google AI Studio, sign in with any Google account, create "
                    "a key (about a minute, no payment details), then paste it below. *(Your key is saved only "
                    "on this computer and is not shared anywhere: it is used solely to send your own questions "
                    "to Google, and git never uploads it.)*")
        st.caption(f"When you ask the assistant something, your question and a short summary of your run "
                   f"are sent to {company}; your layer files are not.")
    else:
        st.caption("A free key, about a minute, very fast answers. "
                   f"Your questions and a short summary of your run are sent to {company}.")
    st.link_button("1 · Get a free key", url, use_container_width=True)
    key = st.text_input("2 · Paste the key here", type="password", key=f"{provider}_key_in",
                        placeholder="paste, then press Enter")
    if st.button("3 · Save and connect", key=f"{provider}_save", type="primary", use_container_width=True,
                 disabled=not key.strip()):
        with st.spinner("Checking the key…"):
            ok, message = validate_key(provider, key)
        if ok:
            updates = {f"{provider.upper()}_API_KEY": key, "LLM_PROVIDER": provider}
            if provider == "gemini":                         # models get retired: use one this key can call now
                try:
                    found = discover_gemini_model(key)
                except Exception:
                    found = None
                if found:
                    updates["GEMINI_MODEL"] = found
            save_settings(updates)
            _connected()
        else:
            st.error(message)
    st.caption("The key is kept in the `.env` file of the repository folder on this computer; git ignores that file.")


def _ollama_tab():
    cfg = load_config()
    cfg = dict(cfg, provider="ollama", model=cfg["model"] if cfg["provider"] == "ollama" else DEFAULT_MODELS["ollama"])
    state = ollama_state(cfg)
    st.caption("Runs on this computer: free, private, no account. It needs about 2 GB of disk and 4 GB of free memory, "
               "and answers more slowly than the online options.")
    if state == "no_server":
        st.link_button("1 · Download Ollama", OLLAMA_DOWNLOAD, use_container_width=True)
        st.caption("2 · Install it and open it once, so that it runs in the background. 3 · Come back and press the button.")
        if st.button("Check again", key="ollama_check", use_container_width=True):
            _connected()
    elif state == "no_model":
        st.success("Ollama is running. One more step: download the model.")
        if st.button(f"Download the model (`{cfg['model']}`, about 2 GB)", key="ollama_pull", type="primary",
                     use_container_width=True):
            bar, note = st.progress(0.0), st.empty()
            try:
                for status, frac in pull_model(cfg):
                    note.caption(status)
                    if frac is not None:
                        bar.progress(min(frac, 1.0))
            except Exception as e:
                st.error(f"The download stopped: {e}")
            else:
                save_settings({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": cfg["model"]})
                _connected()
    else:
        st.success(f"Ollama is running with `{cfg['model']}`.")
        if st.button("Use Ollama", key="ollama_use", type="primary", use_container_width=True):
            save_settings({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": cfg["model"]})
            _connected()


def _setup_panel(can_close):
    st.markdown("**Turn on the assistant** · optional")
    st.caption("The assistant explains each step and can look up values in your run; everything else in the "
               "app works without it. Connecting takes one key, saved on this computer — this panel then "
               "does not come back.")
    _key_tab("gemini")
    with st.expander("Rather not use Google? Two alternatives"):
        alt = st.tabs(["Groq — another free key", "This computer — no account, slower"])
        with alt[0]:
            _key_tab("groq")
        with alt[1]:
            _ollama_tab()
    if can_close and st.button("Keep the current model", key="setup_close", type="tertiary"):
        st.session_state.show_setup = False
        st.rerun()



def _data_context():
    """Where the assistant's tools find the data of the current run."""
    inp = st.session_state.run.get("inputs") or {}
    present, _ = pl.final_outputs()
    return {"neighborhoods_path": inp.get("neighborhoods_path"), "streets_path": inp.get("streets_path"),
            "id_col": inp.get("id_col", "id"),
            "measures_path": str(pl.MEASURES_PATH) if pl.measures_available() else None,
            "final_csv": str(present["csv"]) if present.get("csv") else None}


def _tool_box(ev, key):
    """A collapsed note of what the assistant looked up, with the table it produced (and a download)."""
    args = ", ".join(f"{k}={v}" for k, v in (ev.get("args") or {}).items())
    with st.expander(f"Looked up: {ev['name']}({args})", expanded=ev.get("table") is not None):
        if ev.get("table") is not None:
            st.dataframe(ev["table"], use_container_width=True, hide_index=True, height=220)
            st.download_button("Download CSV", ev["table"].to_csv(index=False).encode("utf-8"),
                               file_name=f"{ev['name']}.csv", mime="text/csv", key=key)
        else:
            st.caption(ev["text"])


def _show(message, idx):
    """One chat message: its text and the lookups in the order they happened."""
    for j, seg in enumerate(message.get("segments") or [{"type": "text", "content": message["content"]}]):
        if seg["type"] == "text":
            if seg["content"].strip():
                st.markdown(seg["content"])
        else:
            _tool_box(seg, key=f"dl_{idx}_{j}")


def render(page):
    status = provider_status()
    cfg, ok = status["cfg"], status["ok"]

    head, tail = st.columns([3, 1])
    head.markdown("#### Assistant")
    if tail.button("Clear", help="forget this conversation", use_container_width=True):
        st.session_state.chat = []
        st.rerun()
    if ok:
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.caption(f"✅ {status['message']}")
        if c2.button("Change", key="setup_open", type="tertiary", help="connect a different model"):
            st.session_state.show_setup = True
            st.rerun()
    else:
        st.caption("⚠️ no model connected")

    history = st.session_state.setdefault("chat", [])
    if st.session_state.get("show_setup") or not ok:
        _setup_panel(can_close=ok)
    box = st.container(height=360) if ok else None

    question = None
    if ok:
        for i, q in enumerate(SUGGESTIONS.get(page, [])):
            if st.button(q, key=f"sugg_{page}_{i}", use_container_width=True):
                question = q
        typed = st.chat_input("Ask about this step…", key="chat_input")
        if typed:
            question = typed

    if question:
        history.append({"role": "user", "content": question})

    if box is not None:
        with box:
            if not history:
                st.caption("The assistant sees the inputs, the stage-1 status and every table of the "
                           "component analysis as you go, and can look things up in your data (for example the area of "
                           "every neighborhood). It never changes anything: every choice stays yours, in the app.")
            for i, m in enumerate(history):
                with st.chat_message(m["role"]):
                    _show(m, i)
            if question:
                system = system_message(st.session_state.run, page)
                messages = [{"role": m["role"], "content": m["content"]} for m in history]
                segments, text = [], ""
                with st.chat_message("assistant"):
                    slot = st.empty()
                    try:
                        for ev in agent_reply(cfg, system, messages, ctx=_data_context()):
                            if ev[0] == "text":
                                text += ev[1]
                                slot.markdown(text + "▌")
                            else:
                                slot.markdown(text)
                                if text.strip():
                                    segments.append({"type": "text", "content": text})
                                text = ""
                                tool = dict(ev[1], type="tool")
                                segments.append(tool)
                                _tool_box(tool, key=f"dl_new_{len(history)}_{len(segments)}")
                                slot = st.empty()
                        slot.markdown(text)
                    except Exception as e:                       # network, model or key problems
                        text += f"\n\nThe model could not answer: {redact(e)}"
                        slot.error(text)
                if text.strip():
                    segments.append({"type": "text", "content": text})
                reply = "\n\n".join(s["content"] for s in segments if s["type"] == "text")
                history.append({"role": "assistant", "content": reply, "segments": segments})
