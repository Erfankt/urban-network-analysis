"""
The look of the app: one stylesheet, the title bar, the section headers with their status chips,
the stepper in the sidebar and the footer.

The palette itself (light and dark) lives in .streamlit/config.toml, where Streamlit applies it to
every widget and follows the system setting. The rules here only add structure on top of it and use
neutral translucent greys plus one teal accent, so they read the same in both modes.
"""
import html

import streamlit as st

TEAL = "#0e8a7f"          # the accent: step badges, the title rule, dots
AMBER = "#d9830b"
RED = "#d64545"

REPO_URL = "https://github.com/Erfankt/urban-network-analysis"
CITATION = ("Kefayat, E., & Thill, J.-C. (2025). Urban Street Network Configuration and Property Crime: "
            "An Empirical Multivariate Case Study. <i>ISPRS International Journal of Geo-Information</i>, 14(5), 200.")

CSS = f"""
<style>
/* ---- page frame ------------------------------------------------------------------------------ */
.block-container {{ max-width: 1320px; padding-top: 2.2rem; padding-bottom: 3rem; }}
h1, h2, h3, h4 {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
                   letter-spacing: -0.01em; }}
h4 {{ font-size: 1.12rem; margin-top: .6rem; }}
[data-testid="stSidebar"] {{ border-right: 1px solid rgba(127,127,127,.18); }}

/* ---- title bar ------------------------------------------------------------------------------- */
.topbar {{ padding: 0 0 .9rem; margin-bottom: 1.4rem; border-bottom: 1px solid rgba(127,127,127,.22);
           position: relative; }}
.topbar::after {{ content: ""; position: absolute; left: 0; bottom: -2px; width: 5.5rem; height: 3px;
                  border-radius: 3px; background: {TEAL}; }}
.topbar .eyebrow {{ font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; opacity: .62; }}
.topbar .title {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
                  font-size: 1.95rem; line-height: 1.2; font-weight: 600; margin: .15rem 0 .1rem; }}
.topbar .sub {{ font-size: .95rem; opacity: .72; }}

/* ---- section cards --------------------------------------------------------------------------- */
[class*="st-key-sec-"] {{ border: 1px solid rgba(127,127,127,.24); border-radius: 18px;
                          padding: 1.5rem 1.8rem 1.7rem; margin-bottom: 1.6rem;
                          background: rgba(127,127,127,.045);
                          box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 14px 30px -18px rgba(0,0,0,.28); }}
[class*="st-key-sec-setup-ready"] {{ padding: .85rem 1.4rem; margin-bottom: 1.1rem; }}
[class*="st-key-sec-setup-ready"] .sec-head {{ margin-bottom: .35rem; }}
[class*="st-key-sec-setup-ready"] .sec-title {{ font-size: 1.15rem; }}
[class*="st-key-sec-setup-ready"] .sec-badge {{ width: 1.7rem; height: 1.7rem; font-size: .85rem; }}
.sec-anchor {{ scroll-margin-top: 4.5rem; height: 0; }}
.sec-head {{ display: flex; align-items: center; gap: .85rem; margin: -.2rem 0 .9rem; flex-wrap: wrap; }}
.sec-badge {{ flex: none; width: 2.15rem; height: 2.15rem; border-radius: 50%; background: {TEAL}; color: #fff;
              display: grid; place-items: center; font-weight: 700; font-size: 1.02rem;
              font-family: system-ui, sans-serif; box-shadow: 0 0 0 4px rgba(14,138,127,.18); }}
.sec-title {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
              font-size: 1.6rem; font-weight: 600; line-height: 1.2; margin: 0; }}
.chip {{ display: inline-flex; align-items: center; gap: .45rem; font-size: .78rem; font-weight: 600;
         padding: .2rem .65rem; border-radius: 999px; border: 1px solid rgba(127,127,127,.3);
         background: rgba(127,127,127,.1); margin-left: auto; white-space: nowrap; }}
.chip i, .step i {{ width: .55rem; height: .55rem; border-radius: 50%; background: rgba(127,127,127,.7); display: inline-block; }}
.chip.done i, .step.done i {{ background: {TEAL}; }}
.chip.ready i, .step.ready i {{ background: transparent; box-shadow: inset 0 0 0 2px {TEAL}; }}
.chip.run i, .step.run i {{ background: {AMBER}; animation: pulse 1.3s ease-in-out infinite; }}
.chip.warn i, .step.warn i {{ background: {AMBER}; }}
.chip.fail i, .step.fail i {{ background: {RED}; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .3; }} }}

/* ---- sidebar stepper ------------------------------------------------------------------------- */
.brand {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; font-size: 1.12rem;
          font-weight: 600; line-height: 1.3; margin-bottom: .2rem; }}
.stepper {{ margin: 1.1rem 0 1rem; position: relative; }}
.stepper::before {{ content: ""; position: absolute; left: .66rem; top: .9rem; bottom: .9rem; width: 2px;
                    background: rgba(127,127,127,.28); }}
.step {{ position: relative; padding: .45rem .5rem .45rem 2.2rem; border-radius: 10px; margin-bottom: .1rem; }}
.step a {{ color: inherit !important; text-decoration: none !important; display: block; }}
.step:hover {{ background: rgba(127,127,127,.12); }}
.step i {{ position: absolute; left: .3rem; top: .72rem; width: .78rem; height: .78rem;
           box-shadow: 0 0 0 4px var(--st-bg, transparent); }}
.step .t {{ font-weight: 600; font-size: .92rem; line-height: 1.25; }}
.step .s {{ font-size: .78rem; opacity: .65; }}
.step.current {{ background: rgba(14,138,127,.13); }}
.facts {{ font-size: .8rem; opacity: .72; line-height: 1.5; }}

/* ---- details ---------------------------------------------------------------------------------- */
[data-testid="stImage"] img, iframe {{ border-radius: 12px; }}
iframe {{ border: 1px solid rgba(127,127,127,.28) !important; }}
[data-testid="stMetric"] {{ background: rgba(127,127,127,.07); padding: .6rem .9rem; border-radius: 12px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; }}
.footer {{ margin: 2.2rem 0 .5rem; padding-top: 1rem; border-top: 1px solid rgba(127,127,127,.22);
           font-size: .82rem; line-height: 1.55; opacity: .75; }}
.footer a {{ color: inherit; text-decoration: underline; text-underline-offset: 3px; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def title_bar():
    st.markdown(
        '<div class="topbar"><div class="eyebrow">Kefayat &amp; Thill (2025) · ISPRS IJGI 14(5):200</div>'
        '<div class="title">Street network configuration and property crime</div>'
        '<div class="sub">The data pipeline of the paper, step by step: choose the inputs, measure the street '
        'networks, reduce the variables, download the final dataset.</div></div>', unsafe_allow_html=True)


def section_head(number, title, anchor, state, label):
    """The header of a section card: step badge, title and a status chip."""
    return (f'<div class="sec-anchor" id="{anchor}"></div><div class="sec-head"><span class="sec-badge">{number}</span>'
            f'<h2 class="sec-title">{html.escape(title)}</h2>'
            f'<span class="chip {state}"><i></i>{html.escape(label)}</span></div>')


def stepper(steps, current):
    """The sidebar list of the four steps with their status; steps = [(anchor, title, state, label)]."""
    items = "".join(
        f'<div class="step {state}{" current" if anchor == current else ""}"><i></i>'
        f'<a href="#{anchor}"><div class="t">{i}  ·  {html.escape(title)}</div>'
        f'<div class="s">{html.escape(label)}</div></a></div>'
        for i, (anchor, title, state, label) in enumerate(steps, start=0))
    return f'<div class="stepper">{items}</div>'


def sidebar_head():
    return '<div class="brand">Street Network Configuration and Property Crime</div>'


def footer():
    st.markdown(
        f'<div class="footer"><b>Cite as</b> {CITATION}<br>'
        f'Code and data construction: <a href="{REPO_URL}" target="_blank" rel="noopener">github.com/Erfankt/'
        f'urban-network-analysis</a> · The app and the pipeline run on your computer; your layers stay here. '
        f'Questions to the assistant (and the small data summaries sent with them) go to the model provider you configure.</div>',
        unsafe_allow_html=True)
