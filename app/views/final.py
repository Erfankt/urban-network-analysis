"""Page 4: the final dataset, its dictionary, the choices made, and downloads."""
import json
import pandas as pd
import streamlit as st

import pipeline as pl


def render():
    st.title("4 · Final dataset")
    present, sel = pl.final_outputs()
    if not present["csv"] or not present["dictionary"]:
        st.info("Not built yet: finish the component analysis on page 3 and build the final dataset there.")
        return

    final = pd.read_csv(present["csv"])
    dictionary = pd.read_csv(present["dictionary"])
    st.markdown(f"**{len(final)} rows × {final.shape[1]} columns.** The repository's end product: one row per "
                "neighborhood, ready for the spatial count models of the paper (fitted outside this app). "
                "The shapefile carries the geometry for the spatial weights.")

    st.subheader("Data dictionary")
    st.dataframe(dictionary, use_container_width=True, hide_index=True, height=min(60 + 35 * len(dictionary), 600))

    st.subheader("Preview")
    st.dataframe(final.round(4), use_container_width=True, height=320)

    st.subheader("Download")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("final_dataset.csv", present["csv"].read_bytes(), "final_dataset.csv", "text/csv",
                       use_container_width=True)
    c2.download_button("Dictionary (CSV)", present["dictionary"].read_bytes(), "final_dataset_dictionary.csv",
                       "text/csv", use_container_width=True)
    c3.download_button("Shapefile (zip)", pl.shapefile_zip(), "final_dataset_shapefile.zip", "application/zip",
                       use_container_width=True)
    if present["selection"]:
        c4.download_button("Choices (JSON)", present["selection"].read_bytes(), "user_selection.json",
                           "application/json", use_container_width=True)
    st.caption(f"The files are in `{pl.FINAL_DIR}`; the figures in `{pl.OUT_DIR}`.")

    if sel:
        with st.expander("Choices recorded in user_selection.json"):
            st.code(json.dumps(sel, indent=2), language="json")

    pngs = sorted(p for p in pl.OUT_DIR.glob("0*.png")) if pl.OUT_DIR.exists() else []
    if pngs:
        with st.expander(f"Component-analysis figures ({len(pngs)})"):
            for p in pngs:
                st.image(str(p), caption=p.name, use_container_width=True)

    st.session_state.run["final"] = {"rows": len(final), "columns": list(final.columns)}
