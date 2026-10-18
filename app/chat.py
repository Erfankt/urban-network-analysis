"""
The chatbot behind the chat panel: which model answers, how it is called, and what it is
told about the run in progress.

Provider and model come from a .env file in the repository folder (see .env.example) or from
environment variables. Nothing is required: without any setting the app looks for a local
Ollama server and its llama3.2 model, and works without the chat when there is none.

    LLM_PROVIDER   ollama | gemini | groq   (unset: gemini or groq when its key is set, else ollama)
    OLLAMA_URL     http://localhost:11434   OLLAMA_MODEL  llama3.2
    GEMINI_API_KEY                           GEMINI_MODEL  gemini-flash-latest
    GROQ_API_KEY                             GROQ_MODEL    llama-3.3-70b-versatile

The three providers are called over plain HTTP (no SDKs), streaming the reply. Every call
carries a system message made of fixed method notes (what the pipeline computes and what the
paper chose) and a description of the current run (inputs, stage-1 status, the eigenvalues,
the solutions inspected, the choices made), so the answers refer to the user's own numbers.
"""
import json, os, re
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
load_dotenv(REPO_DIR / ".env")

DEFAULT_MODELS = {"ollama": "llama3.2", "gemini": "gemini-flash-latest", "groq": "llama-3.3-70b-versatile"}
GROQ_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GEMINI_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")


def redact(text):
    """Error text safe to show or keep: any key that an URL or a provider echo might carry is masked."""
    out = re.sub(r"(key=)[A-Za-z0-9_\-]{8,}", r"\1[hidden]", str(text))
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        value = os.getenv(name, "").strip()
        if len(value) >= 8:
            out = out.replace(value, "[hidden]")
    return out
HISTORY_TURNS = 8          # how many past messages go with each question
TIMEOUT = (10, 180)        # connect, read (seconds); a local model on a CPU can be slow


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config():
    """The provider to use and how to reach it, from the environment (.env already loaded)."""
    gemini_key, groq_key = os.getenv("GEMINI_API_KEY", "").strip(), os.getenv("GROQ_API_KEY", "").strip()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider not in ("ollama", "gemini", "groq"):
        provider = "gemini" if gemini_key else "groq" if groq_key else "ollama"
    cfg = {"provider": provider,
           "model": os.getenv(f"{provider.upper()}_MODEL", "").strip() or DEFAULT_MODELS[provider],
           "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
           "api_key": gemini_key if provider == "gemini" else groq_key if provider == "groq" else ""}
    return cfg


def check_provider(cfg):
    """
    Whether the configured model can answer now: (ok, message). For Ollama this asks the local
    server for its models; for the cloud providers it only checks that a key is set (the first
    real call reports anything else).
    """
    p = cfg["provider"]
    if p == "ollama":
        try:
            r = requests.get(f"{cfg['ollama_url']}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except requests.RequestException:
            return False, (f"No Ollama server at {cfg['ollama_url']}. Install it from ollama.com, start it, "
                           f"run `ollama pull {cfg['model']}` once, then reload this page.")
        wanted = cfg["model"] if ":" in cfg["model"] else f"{cfg['model']}:latest"
        if wanted not in names and cfg["model"] not in names:
            have = ", ".join(n for n in names) or "none"
            return False, (f"Ollama is running but has no model `{cfg['model']}` (models present: {have}). "
                           f"Run `ollama pull {cfg['model']}`, or set OLLAMA_MODEL in .env to one you have.")
        return True, f"Ollama · {cfg['model']} (local, free)"
    if not cfg["api_key"]:
        return False, f"{p.capitalize()} is selected but {p.upper()}_API_KEY is not set in .env."
    return True, f"{p.capitalize()} · {cfg['model']} (your API key)"


# ---------------------------------------------------------------------------
# Setting the assistant up from the app (no file editing)
# ---------------------------------------------------------------------------

ENV_PATH = REPO_DIR / ".env"
KEY_PAGES = {"gemini": ("Google", "https://aistudio.google.com/apikey"),
             "groq": ("Groq", "https://console.groq.com/keys")}
OLLAMA_DOWNLOAD = "https://ollama.com/download"


def save_settings(updates):
    """
    Write KEY=value pairs into the .env file of the repository (created when missing, other lines
    kept) and apply them to the running process, so the new setting works at once. The file is
    ignored by git; a key typed in the app lives only there.
    """
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    for key, value in updates.items():
        value = str(value).strip().replace("\n", "").replace("\r", "")
        for i, line in enumerate(lines):
            if line.split("=", 1)[0].strip() == key:
                lines[i] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")
        os.environ[key] = value
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_key(provider, key):
    """Ask the provider whether the key is accepted (a list-models call, nothing is generated): (ok, message)."""
    key = key.strip()
    try:
        if provider == "gemini":
            r = requests.get(f"{GEMINI_URL}/models", params={"pageSize": 1},
                             headers={"x-goog-api-key": key}, timeout=15)
        else:
            r = requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except requests.RequestException:
        return False, "Could not reach the service. Check the internet connection and try again."
    if r.status_code == 200:
        return True, "Key accepted."
    if r.status_code in (400, 401, 403):
        return False, "The service did not accept this key. Copy it again from the key page, without spaces."
    if r.status_code == 429:
        return True, "Key accepted (the service is rate-limiting right now, it will answer shortly)."
    return False, f"The service answered {r.status_code}. Try again in a moment."


def discover_gemini_model(key, avoid=""):
    """
    The newest stable Flash model this key can call (models are retired from time to time, so the
    name is looked up, not assumed). Returns its name, e.g. 'gemini-3-flash', or None.
    """
    names, token = [], None
    for _ in range(6):
        params = {"pageSize": 100}
        if token:
            params["pageToken"] = token
        r = requests.get(f"{GEMINI_URL}/models", params=params,
                         headers={"x-goog-api-key": key.strip()}, timeout=15)
        r.raise_for_status()
        data = r.json()
        for m in data.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                names.append(m["name"].split("/", 1)[-1])
        token = data.get("nextPageToken")
        if not token:
            break
    versioned = []
    for n in names:
        m = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", n)
        if m and n != avoid:
            versioned.append((float(m.group(1)), n))
    if versioned:
        return max(versioned)[1]
    plain = [n for n in names if "flash" in n and n != avoid
             and not re.search(r"lite|preview|exp|image|tts|live|thinking|audio|8b", n)]
    return sorted(plain)[-1] if plain else None


def ollama_state(cfg):
    """'no_server' (Ollama not installed or not running), 'no_model' (running, model not downloaded) or 'ready'."""
    try:
        r = requests.get(f"{cfg['ollama_url']}/api/tags", timeout=3)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except requests.RequestException:
        return "no_server"
    wanted = cfg["model"] if ":" in cfg["model"] else f"{cfg['model']}:latest"
    return "ready" if wanted in names or cfg["model"] in names else "no_model"


def pull_model(cfg):
    """Download the model through the local Ollama server; yields (status text, fraction done or None)."""
    body = {"model": cfg["model"], "stream": True}
    with requests.post(f"{cfg['ollama_url']}/api/pull", json=body, stream=True, timeout=(10, 600)) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            chunk = json.loads(raw)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            total, done = chunk.get("total"), chunk.get("completed")
            yield chunk.get("status", ""), (done / total if total and done is not None else None)


# ---------------------------------------------------------------------------
# Calling the model (streaming, with tools)
# ---------------------------------------------------------------------------
# Each provider function runs ONE round and yields events: ("text", piece) while the reply streams, and at
# the end ("calls", [{"name", "args", "id"}], raw) when the model asks for tools (raw = what to send back as
# the model's turn). agent_reply() loops: round, run the tools, round again, until the model answers.

MAX_ROUNDS = 5             # tool rounds per question


def _sse_json(resp):
    """Yield the JSON objects of a server-sent-events stream (lines starting with 'data: ')."""
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _round_ollama(cfg, system, messages, tools):
    body = {"model": cfg["model"], "stream": True,
            "messages": [{"role": "system", "content": system}] + messages,
            "options": {"temperature": 0.2, "num_ctx": 8192}}
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
    calls = []
    with requests.post(f"{cfg['ollama_url']}/api/chat", json=body, stream=True, timeout=TIMEOUT) as r:
        if r.status_code == 400 and tools:                      # this model does not take tools: answer without
            yield from _round_ollama(cfg, system, messages, None)
            return
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            chunk = json.loads(raw)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            msg = chunk.get("message", {})
            if msg.get("content"):
                yield ("text", msg["content"])
            for c in msg.get("tool_calls", []) or []:
                fn = c.get("function", {})
                calls.append({"name": fn.get("name"), "args": fn.get("arguments") or {}, "id": None})
            if chunk.get("done"):
                break
    if calls:
        raw_turn = {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": c["name"], "arguments": c["args"]}} for c in calls]}
        yield ("calls", calls, raw_turn)


def _round_groq(cfg, system, messages, tools):
    body = {"model": cfg["model"], "stream": True, "temperature": 0.2,
            "messages": [{"role": "system", "content": system}] + messages}
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    partial = {}
    with requests.post(f"{GROQ_URL}/chat/completions", json=body, headers=headers,
                       stream=True, timeout=TIMEOUT) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Groq answered {r.status_code}: {r.text[:300]}")
        for chunk in _sse_json(r):
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    yield ("text", delta["content"])
                for tc in delta.get("tool_calls", []) or []:
                    p = partial.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    p["id"] = tc.get("id") or p["id"]
                    fn = tc.get("function", {})
                    p["name"] += fn.get("name") or ""
                    p["arguments"] += fn.get("arguments") or ""
    if partial:
        calls, raw_calls = [], []
        for i, p in sorted(partial.items()):
            try:
                args = json.loads(p["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            cid = p["id"] or f"call_{i}"
            calls.append({"name": p["name"], "args": args, "id": cid})
            raw_calls.append({"id": cid, "type": "function",
                              "function": {"name": p["name"], "arguments": json.dumps(args)}})
        yield ("calls", calls, {"role": "assistant", "content": None, "tool_calls": raw_calls})


def _round_gemini(cfg, system, contents, tools):
    body = {"system_instruction": {"parts": [{"text": system}]}, "contents": contents,
            "generationConfig": {"temperature": 0.2}}
    if tools:
        body["tools"] = [{"functionDeclarations": tools}]
    for attempt in (1, 2):
        # the key travels in a header, never in the URL: exception messages embed the URL
        url = f"{GEMINI_URL}/models/{cfg['model']}:streamGenerateContent?alt=sse"
        with requests.post(url, json=body, stream=True, timeout=TIMEOUT,
                           headers={"x-goog-api-key": cfg["api_key"]}) as r:
            if r.status_code == 404 and attempt == 1:
                # the model was retired or renamed: look up one this key can use, remember it, and retry
                try:
                    found = discover_gemini_model(cfg["api_key"], avoid=cfg["model"])
                except requests.RequestException:
                    found = None
                if found:
                    cfg["model"] = found
                    save_settings({"GEMINI_MODEL": found})
                    continue
            if r.status_code != 200:
                raise RuntimeError(f"Gemini answered {r.status_code}: {r.text[:300]}")
            calls, parts_out = [], []
            for chunk in _sse_json(r):
                for cand in chunk.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("text") and not part.get("thought"):
                            yield ("text", part["text"])
                        if "functionCall" in part:
                            fc = part["functionCall"]
                            calls.append({"name": fc.get("name"), "args": fc.get("args") or {}, "id": None})
                            parts_out.append(part)           # kept whole: newer models require their signature back
            if calls:
                yield ("calls", calls, {"role": "model", "parts": parts_out})
            return


def _to_gemini(messages):
    return [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages]


def _result_messages(provider, calls, results):
    """The tool answers, in the shape each provider expects them after the model's own turn."""
    if provider == "gemini":
        return [{"role": "user", "parts": [{"functionResponse": {"name": c["name"], "response": {"result": t}}}
                                           for c, t in zip(calls, results)]}]
    if provider == "groq":
        return [{"role": "tool", "tool_call_id": c["id"], "content": t} for c, t in zip(calls, results)]
    return [{"role": "tool", "tool_name": c["name"], "content": t} for c, t in zip(calls, results)]


def agent_reply(cfg, system, messages, ctx=None, tools=None):
    """
    Answer the last question in `messages`, letting the model look things up with the tools.
    Yields ("text", piece) as the reply streams and ("tool", {"name", "args", "text", "table"}) after
    each tool the model used, so the panel can show both.
    """
    import tools as tool_mod
    tools = TOOLS_ON and (tool_mod.TOOLS if tools is None else tools) or None
    provider = cfg["provider"]
    rnd = {"ollama": _round_ollama, "gemini": _round_gemini, "groq": _round_groq}[provider]
    history = messages[-HISTORY_TURNS:]
    convo = _to_gemini(history) if provider == "gemini" else list(history)
    for step in range(MAX_ROUNDS + 1):
        offer = tools if step < MAX_ROUNDS else None           # the last round must answer in words
        calls = raw = None
        for ev in rnd(cfg, system, convo, offer):
            if ev[0] == "text":
                yield ev
            else:
                calls, raw = ev[1], ev[2]
        if not calls:
            return
        results = []
        for c in calls:
            out = tool_mod.call_tool(c["name"], c["args"], ctx or {})
            results.append(out["text"])
            yield ("tool", {"name": c["name"], "args": c["args"], "text": out["text"], "table": out["table"]})
        convo.append(raw)
        convo.extend(_result_messages(provider, calls, results))


TOOLS_ON = os.getenv("ASSISTANT_TOOLS", "1").strip() != "0"     # ASSISTANT_TOOLS=0 turns the lookups off


def stream_reply(cfg, system, messages):
    """The reply as plain text pieces, without tools (kept for simple uses)."""
    for ev in agent_reply(cfg, system, messages, tools=[]):
        if ev[0] == "text":
            yield ev[1]


# ---------------------------------------------------------------------------
# What the model is told
# ---------------------------------------------------------------------------

ROLE = """You are the assistant built into a web app that runs a data pipeline for research on urban \
street networks and property crime (Kefayat & Thill, 2025, ISPRS IJGI 14(5):200). The user moves \
through four sections of one page: Inputs, Stage 1 (street-network measures), Stage 2 (component analysis) and \
Final dataset. You explain the steps, the measures and the diagnostics, and you interpret the numbers \
of the current run, suggesting choices with your reasons; the user makes every choice by clicking in the app. \
You have read-only tools to look things up in the user's data: list the columns, compute the area of every \
neighborhood, summarize or correlate columns, fetch the values of given neighborhoods. When the user asks for a \
number or a calculation on the data, call a tool instead of guessing, then quote the result; when the \
tool says the data does not exist yet, tell the user which section produces it. A table a tool produces is \
shown to the user in the chat automatically, so say that it is there rather than retyping it. You cannot run \
code, change files or choose for the user. Answer briefly and concretely, in plain language."""

METHOD_NOTES = """METHOD NOTES
Inputs: a cleaned street centerline layer (any projected CRS, in feet or metres, or latitude/longitude, which is moved to its UTM zone; it defines the working CRS and its unit is converted to miles, acres and feet) and \
neighborhood polygons with an identifier column (unique, chosen in the Inputs section; `id` in the paper's \
data), a crime-count column and numeric control variables. The neighborhoods are reprojected to the \
streets CRS when they differ.
Stage 1 builds the street network (nodes at segment endpoints; degree-2 nodes removed so nodes are \
intersections and dead ends), buffers each neighborhood by the chosen distances (paper: 0.1, 0.25, \
0.5, 0.75 and 1.0 miles), keeps the largest connected component of the segments inside each buffer, \
and measures it with eight metrics: int_den intersection density (nodes of degree >= 3 per acre), \
st_den street density (street length per acre), nd_deg average node degree, cl_cof average clustering \
coefficient, shtpth average length-weighted shortest path (miles), bt_cnt average length-weighted \
betweenness centrality, cl_cnt average closeness centrality, pr_cnt average PageRank. Column names are \
metric + buffer code: 1 (0.1 mi), 025, 05, 075, 10 (1.0 mi), e.g. int_den075. The measures are \
checkpointed per neighborhood, so an interrupted run resumes.
Stage 2 reduces the measures with a component analysis (varimax rotation). Extraction methods: \
principal (principal components, as the paper describes), minres (minimum-residual factor analysis, \
what the original computation ran) and ml (maximum likelihood). The overview shows the correlation \
matrix, the scree plot and all eigenvalues with the cumulative % of variance; rules of thumb: Kaiser \
(eigenvalues > 1), the elbow of the scree plot, 70/80/90% of variance; interpretability decides in the \
end. A candidate solution shows rotated loadings; salient loadings are |loading| >= 0.5; a variable \
salient on two components cross-loads; communality is the share of a variable's variance the solution \
keeps. One original measure represents each retained network component (highest absolute loading, \
preferring a smaller cross-loading and a clear meaning); the representatives, not the scores, enter \
the models after a log(x + 1) transform. The socio-spatial controls get the same treatment and their \
component scores are the controls.
The paper (Charlotte, NC, 371 neighborhoods, 16 controls): 4 network components (74.1% of variance \
with minres) represented by int_den075, cl_cof075, shtpth075 and bt_cnt05; 9 control components. \
Final dataset columns: the identifier, cent_x/cent_y (centroid, for GWR), the crime count, area_acre (log as the \
offset of the count models), the representative measures, PC*_net scores (robustness check) and \
PC*_ctrl scores (controls). The models themselves (NB GLM, SAR/CAR-NB with INLA, GWR-NB) are fitted \
outside the app."""


def _table_text(df, max_rows=12):
    return df.head(max_rows).round(3).to_string()


def describe_run(run, page):
    """The current run as text for the system message. `run` is the app's run-state dict."""
    out = [f"CURRENT RUN (the user is at the step: {page})"]
    inp = run.get("inputs")
    if inp:
        out.append(f"Inputs checked: {inp.get('summary', '')}. Identifier column: {inp.get('id_col', 'id')}. "
                   f"Numeric columns available: {', '.join(inp.get('columns', [])) or 'none'}."
                   + (f" Note: {inp['note']}." if inp.get("note") else ""))
    else:
        out.append("Inputs: not chosen yet (Inputs section).")
    s1 = run.get("stage1")
    if s1:
        sel = s1.get("selection", {})
        out.append(f"Stage 1: status {s1.get('status', 'unknown')}; buffers {sel.get('buffer_list_mile')}; "
                   f"dependent variable {sel.get('crime_col')}; controls {sel.get('ctrl_vars')}."
                   + (f" Progress: {s1['progress']}." if s1.get("progress") else ""))
    else:
        out.append("Stage 1: not started.")
    s2 = run.get("stage2")
    if s2:
        out.append(f"Stage 2: extraction method {s2.get('method')}, rotation {s2.get('rotation')}.")
        for key, name in (("network", "Street-network block"), ("controls", "Socio-spatial controls block")):
            b = s2.get(key)
            if not b:
                continue
            out.append(f"{name}: {len(b.get('variables', []))} variables.")
            if b.get("rules"):
                out.append(f"  Rules of thumb: {b['rules']}")
            if b.get("table") is not None:
                out.append("  Eigenvalues (first rows):\n" + _table_text(b["table"]))
            for n, info in sorted(b.get("inspected", {}).items()):
                out.append(f"  Inspected {n} components: {info['retained']:.1f}% of variance retained.\n"
                           + "\n".join("    " + line for line in info["summary"].splitlines()))
            if b.get("retained"):
                out.append(f"  RETAINED: {b['retained']} components.")
            if b.get("candidates"):
                out.append("  Representative candidates per component (loading; largest loading elsewhere):")
                for comp, cands in b["candidates"].items():
                    out.append(f"    {comp}: " + "; ".join(
                        f"{c['measure']} ({c['loading']:+.2f}; {c['max_elsewhere']:.2f} on {c['where']})"
                        if c["max_elsewhere"] is not None else f"{c['measure']} ({c['loading']:+.2f})" for c in cands))
            if b.get("representatives"):
                out.append(f"  Representatives chosen: {b['representatives']}.")
    else:
        out.append("Stage 2: not started.")
    fin = run.get("final")
    if fin:
        out.append(f"Final dataset written: {fin.get('rows')} rows, columns {fin.get('columns')}.")
    return "\n".join(out)


def system_message(run, page):
    return f"{ROLE}\n\n{METHOD_NOTES}\n\n{describe_run(run, page)}"


SUGGESTIONS = {
    "Inputs": ["Compute the area in acres of all neighborhoods.",
               "What must the two input layers contain?",
               "Does the streets layer have to be in feet?",
               "What does the CRS warning mean?"],
    "Stage 1": ["Compute the area in acres of all neighborhoods.",
                "What do the buffer distances mean, and why five of them?",
                "Explain the eight measures in one line each.",
                "Why do the larger buffers take so much longer?"],
    "Stage 2": ["How many components would you keep for the street-network block, and why?",
                "Which measure should represent each component?",
                "What is a cross-loading and does it matter here?",
                "principal or minres: what is the difference?"],
    "Final dataset": ["Compute the area in acres of all neighborhoods.",
                      "How do I use each column in the models?",
                      "Why log(x + 1) for the representatives but not for the scores?",
                      "What is area_acre for?"],
}
