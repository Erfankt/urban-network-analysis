"""
Stage 0 of the app: is the Python environment ready, and if not, make it ready.

It follows the setup steps of the README: (1) the virtual environment `.venv` in the repository folder
exists, (2) the app runs inside it, (3) the dependencies of requirements.txt are installed, with the
versions it pins (scikit-learn < 1.6 for factor_analyzer). When everything is in place the page shows a
one-line "ready" note and goes on; when something is missing it offers to do the missing steps.

This module imports only the standard library, `packaging` (a dependency of Streamlit) and Streamlit,
so it works when the heavy packages (geopandas and the rest) are not installed yet, which is exactly
when it is needed. The app imports the rest only after this check passes.
"""
import html
import importlib
import importlib.metadata as md
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from packaging.requirements import Requirement

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
REQUIREMENTS = REPO_DIR / "requirements.txt"
VENV_DIR = REPO_DIR / ".venv"
RECOMMENDED_PYTHON = (3, 11)
OPTIONAL_MARKER = "Notebook"            # the requirements.txt group that is only for the notebook version of stage 2


# ---------------------------------------------------------------------------
# What is required
# ---------------------------------------------------------------------------

def parse_requirements(path=REQUIREMENTS):
    """requirements.txt as a list of {group, name, spec, line, optional}; the group is the comment above a block."""
    reqs, group = [], ""
    if not Path(path).exists():
        return reqs
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        if text.startswith("#"):
            group = text.lstrip("# ").strip()
            continue
        line = text.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            r = Requirement(line)
        except Exception:
            continue
        reqs.append({"group": group, "name": r.name, "spec": str(r.specifier), "line": line,
                     "optional": group.startswith(OPTIONAL_MARKER)})
    return reqs


def check_packages(reqs=None):
    """Each requirement against the Python that is running this app: rows of name, wanted, installed, status."""
    rows = []
    for r in (parse_requirements() if reqs is None else reqs):
        spec = Requirement(r["line"]).specifier
        try:
            have = md.version(r["name"])
        except md.PackageNotFoundError:
            have = None
        if have is None:
            status = "missing"
        elif have and spec and not spec.contains(have, prereleases=True):
            status = "wrong version"
        else:
            status = "ok"
        rows.append({**r, "installed": have, "status": status})
    return rows


def problems(rows, include_optional=False):
    """The rows that need installing (the notebook group only when asked for)."""
    return [r for r in rows if r["status"] != "ok" and (include_optional or not r["optional"])]


# ---------------------------------------------------------------------------
# The interpreters
# ---------------------------------------------------------------------------

def venv_python(venv=VENV_DIR):
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return exe


def venv_exists(venv=VENV_DIR):
    return venv_python(venv).exists()


def running_in_venv(venv=VENV_DIR):
    """True when this app was started by the python of the repository's .venv."""
    try:
        return Path(sys.prefix).resolve() == venv.resolve()
    except OSError:
        return False


def restart_command(venv=VENV_DIR):
    exe = venv_python(venv)
    rel = os.path.relpath(exe, REPO_DIR)
    return f'"{rel}" -m streamlit run app/streamlit_app.py' if " " in rel else f"{rel} -m streamlit run app/streamlit_app.py"


def python_note():
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) != RECOMMENDED_PYTHON:
        text += f" (the paper's computations used {RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]}; others usually work)"
    return text


# ---------------------------------------------------------------------------
# Doing the steps (each yields the output lines as they come)
# ---------------------------------------------------------------------------

def _stream(cmd, cwd=REPO_DIR):
    """Run a command and yield its output lines; the last item is ('exit', code)."""
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        yield ("line", line.rstrip())
    yield ("exit", p.wait())


def create_venv():
    """Step 1: python -m venv .venv with the Python that runs this app."""
    yield from _stream([sys.executable, "-m", "venv", str(VENV_DIR)])


def install(python, reqs):
    """Step 3: pip install of the given requirements into the given interpreter."""
    lines = [r["line"] for r in reqs]
    yield from _stream([str(python), "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *lines])


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def environment():
    """The state of the environment, from the running Python (cheap: metadata only)."""
    reqs = parse_requirements()
    rows = check_packages(reqs)
    return {"rows": rows, "venv_exists": venv_exists(), "in_venv": running_in_venv(),
            "python": python_note(), "needs": problems(rows, include_optional=False)}


def ready(env):
    return not env["needs"]


def summary_line(env):
    n = sum(1 for r in env["rows"] if r["status"] == "ok")
    where = "virtual environment .venv" if env["in_venv"] else "this Python"
    return f"{env['python']} · {n} packages in place · running in {where}"


TIPS = {"scikit-learn": "<b>Why <code>scikit-learn &lt; 1.6</code>?</b><br><code>factor_analyzer</code> 0.5.x fails with newer "
                        "scikit-learn releases (an error such as <code>check_array() got an unexpected keyword argument "
                        "'force_all_finite'</code>), so <code>requirements.txt</code> pins it."}

TABLE_CSS = """<style>
.pk { width: 100%; border-collapse: collapse; font-size: .88rem; }
.pk th { text-align: left; font-weight: 600; padding: .35rem .6rem; border-bottom: 1px solid rgba(127,127,127,.35); opacity: .75; }
.pk td { padding: .32rem .6rem; border-bottom: 1px solid rgba(127,127,127,.15); }
.pk .bad { color: #d9830b; font-weight: 600; }
.pk .ok { color: #0e8a7f; }
.tipword { position: relative; display: inline-block; cursor: help; border-bottom: 1px dotted currentColor; }
.tipword .tipbox { display: none; position: absolute; left: 0; bottom: 130%; z-index: 50; width: 22rem; max-width: 70vw;
                   padding: .7rem .85rem; border-radius: 10px; font-size: .82rem; line-height: 1.45; font-weight: 400;
                   background: #1f2933; color: #f3f6f8; box-shadow: 0 8px 24px rgba(0,0,0,.3); text-align: left; }
.tipword .tipbox code { background: rgba(255,255,255,.14); color: inherit; padding: 0 .25rem; border-radius: 4px; }
.tipword:hover .tipbox, .tipword:focus .tipbox { display: block; }
</style>"""


def _table(rows):
    """The packages as an HTML table; a package with an explanation (TIPS) shows it in a pop-up on hover."""
    body = []
    for r in rows:
        name = html.escape(r["name"])
        if r["name"] in TIPS:
            name = f'<span class="tipword" tabindex="0">{name}<span class="tipbox">{TIPS[r["name"]]}</span></span>'
        status = r["status"] + (" (optional)" if r["optional"] and r["status"] != "ok" else "")
        body.append(f"<tr><td>{name}</td><td>{html.escape(r['spec'] or 'any')}</td>"
                    f"<td>{html.escape(r['installed'] or '—')}</td>"
                    f"<td class=\"{'ok' if r['status'] == 'ok' else 'bad'}\">{html.escape(status)}</td>"
                    f"<td>{html.escape(r['group'])}</td></tr>")
    return (TABLE_CSS + '<table class="pk"><tr><th>package</th><th>required</th><th>installed</th><th>status</th>'
            '<th>used for</th></tr>' + "".join(body) + "</table>")


def _run_steps(steps, label):
    """Run generator steps with a live log; returns True when every step exited with 0."""
    box = st.empty()
    tail = []
    ok = True
    with st.status(label, expanded=True) as status:
        for title, gen in steps:
            st.write(f"**{title}**")
            slot = st.empty()
            for kind, value in gen:
                if kind == "line":
                    tail.append(value)
                    slot.code("\n".join(tail[-14:]), language=None)
                elif value != 0:
                    ok = False
            if not ok:
                break
        status.update(label=label + (" — done" if ok else " — failed"), state="complete" if ok else "error")
    box.empty()
    if not ok:
        st.error("A step failed; the last lines of its output are above. A common cause is no internet connection "
                 "(the packages are downloaded from pypi.org). Fix it and press the button again.")
    return ok


def render(env):
    """
    The Stage 0 card body. Returns True when the app may go on (everything installed in the running Python).
    When ready it is a compact note with the details collapsed; otherwise the steps with the buttons.
    """
    if ready(env):
        v = sys.version_info
        n = sum(1 for r in env["rows"] if r["status"] == "ok")
        where = "in the virtual environment .venv" if env["in_venv"] else "in this Python"
        st.markdown(f"Python {v.major}.{v.minor}.{v.micro} · {n} packages in place · running {where}.")
        with st.expander("Details"):
            st.caption(env["python"] + ".")
            if not env["in_venv"]:
                st.caption("The app is not running inside the repository's `.venv`; that is fine as long as the "
                           "packages are in place, though a virtual environment keeps them apart from your other projects.")
            st.markdown(_table(env["rows"]), unsafe_allow_html=True)
        return True

    st.markdown("Before the pipeline can run, the Python environment needs a few packages. "
                "This step does it for you; it needs an internet connection and takes a few minutes the first time.")
    missing = env["needs"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**1 · Virtual environment**  \n{'✅ `.venv` exists' if env['venv_exists'] else '⬜ not created yet'}")
    c2.markdown(f"**2 · App runs in it**  \n{'✅ yes' if env['in_venv'] else '⬜ no (' + env['python'].split(' (')[0] + ' outside .venv)'}")
    c3.markdown(f"**3 · Packages**  \n⬜ {len(missing)} to install or fix")
    st.markdown(_table(missing), unsafe_allow_html=True)

    notebook = [r for r in env["rows"] if r["optional"] and r["status"] != "ok"]
    with_nb = False
    if notebook:
        with_nb = st.checkbox("Also install Jupyter (only for the notebook version of stage 2)", value=False)
    todo = problems(env["rows"], include_optional=with_nb)

    if env["in_venv"]:
        place, python = "the virtual environment this app runs in", sys.executable
        choices = ["here"]
    else:
        if env["venv_exists"]:
            choices = ["venv", "here"]
            labels = {"venv": "The existing `.venv` of the repository (recommended; the app then needs a restart)",
                      "here": f"The Python running this app ({sys.executable})"}
        else:
            choices = ["new", "here"]
            labels = {"new": "A new virtual environment, `.venv`, in the repository (recommended; the app then needs a restart)",
                      "here": f"The Python running this app ({sys.executable})"}
    if not env["in_venv"]:
        choice = st.radio("Where should the packages go?", choices, format_func=lambda k: labels[k], key="setup_where")
    else:
        choice = "here"

    button = {"here": "Install the packages", "venv": "Install into .venv", "new": "Create .venv and install"}[choice]
    if st.button(button, type="primary", key="setup_go"):
        steps = []
        target = sys.executable
        if choice == "new":
            steps.append(("Create the virtual environment", create_venv()))
        if choice in ("new", "venv"):
            target = str(venv_python())
            todo = [r for r in parse_requirements() if with_nb or not r["optional"]]   # a fresh .venv has nothing yet
        steps.append(("Install the packages", install(target, todo)))
        if _run_steps(steps, "Setting up the environment"):
            importlib.invalidate_caches()
            if choice == "here":
                st.success("Done. Loading the app…")
                st.session_state["setup_done"] = True
                st.rerun()
            else:
                st.session_state["setup_restart"] = True
        # fall through: on failure the log stays visible
    if st.session_state.get("setup_restart"):
        st.success("The environment is ready. One last step: restart the app inside it.")
        st.markdown("Stop the app (Ctrl+C in the terminal), then from the repository folder run:")
        st.code(restart_command(), language="bash")
    return False
