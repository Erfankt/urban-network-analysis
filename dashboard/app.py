"""
dashboard/app.py
Urban Street Network & Crime — Streamlit Dashboard
Pulls live data from the FastAPI service.
No geopandas/GDAL dependency — folium accepts raw GeoJSON dicts.
"""

import os

import requests
import pandas as pd
import folium
from folium import Choropleth, GeoJsonTooltip
from streamlit_folium import st_folium
import plotly.express as px
import streamlit as st

# When running inside Docker, API_URL is set to http://api:8000 (internal DNS).
# When running locally, defaults to http://localhost:8000.
API = os.getenv("API_URL", "http://localhost:8000")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Charlotte NPA Crime & Street Network",
    page_icon="🏙️",
    layout="wide",
)

st.title("🏙️ Charlotte NPA — Crime & Street Network Dashboard")
st.caption("Data: Kefayat & Thill (2025), ISPRS Int. J. Geo-Information 14(5), 200 · API: " + API)

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_npa_list() -> pd.DataFrame:
    r = requests.get(f"{API}/npa", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=300)
def fetch_npa_geojson() -> dict:
    """Fetch all 371 NPAs as a raw GeoJSON FeatureCollection dict (WGS84)."""
    r = requests.get(f"{API}/npa/geojson", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fetch_top(metric: str, buffer_mi: float, n: int = 371) -> pd.DataFrame:
    r = requests.get(f"{API}/metrics/top",
                     params={"metric": metric, "buffer_mi": buffer_mi, "n": n},
                     timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def fetch_near(lon: float, lat: float, radius_mi: float) -> pd.DataFrame:
    r = requests.get(f"{API}/spatial/near",
                     params={"lon": lon, "lat": lat, "radius_mi": radius_mi},
                     timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


METRIC_LABELS = {
    "int_den": "Intersection Density",
    "st_den":  "Street Density",
    "nd_deg":  "Node Degree",
    "cl_cof":  "Clustering Coefficient",
    "shtpth":  "Shortest Path Length",
    "bt_cnt":  "Betweenness Centrality",
    "cl_cnt":  "Closeness Centrality",
    "pr_cnt":  "PageRank Centrality",
}
BUFFER_OPTIONS = {
    "0.10 mi (528 ft)": 0.10,
    "0.25 mi (¼ mile)": 0.25,
    "0.50 mi (½ mile)": 0.50,
    "0.75 mi (¾ mile)": 0.75,
    "1.00 mi (1 mile)":  1.00,
}

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🗺️ Crime Map",
    "📊 Network Metrics Explorer",
    "📍 Spatial Proximity Tool",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Crime Choropleth Map (all 371 NPAs, one API call)
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Property Crime Rate by Neighborhood Planning Area")

    with st.spinner("Loading NPA data from API…"):
        try:
            df = fetch_npa_list()
            fc = fetch_npa_geojson()   # raw dict — no geopandas needed
        except Exception as e:
            st.error(f"Could not reach API at {API}: {e}")
            st.stop()

    col_a, col_b = st.columns([3, 1])

    with col_b:
        st.metric("Total NPAs", len(df))
        st.metric("Max crime/1k",    f"{df['propcrim_1k'].max():.1f}")
        st.metric("Median crime/1k", f"{df['propcrim_1k'].median():.1f}")
        st.metric("Min crime/1k",    f"{df['propcrim_1k'].min():.1f}")

        st.markdown("---")
        st.markdown("**Top 10 highest-crime NPAs**")
        top10 = (df.nlargest(10, "propcrim_1k")
                   [["npa_id", "propcrim_1k"]]
                   .reset_index(drop=True))
        top10.columns = ["NPA", "Crime/1k"]
        st.dataframe(top10, use_container_width=True, hide_index=True)

    with col_a:
        m = folium.Map(location=[35.23, -80.84], zoom_start=11,
                       tiles="OpenStreetMap")

        # Choropleth uses the raw GeoJSON dict and the summary DataFrame
        Choropleth(
            geo_data=fc,                         # raw GeoJSON dict
            data=df,
            columns=["npa_id", "propcrim_1k"],
            key_on="feature.properties.npa_id",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name="Property Crimes per 1,000 Residents",
            nan_fill_color="lightgray",
        ).add_to(m)

        # Tooltip layer — zero-opacity fill so only the hover label shows
        folium.GeoJson(
            fc,
            tooltip=GeoJsonTooltip(
                fields=["npa_id", "propcrim_1k", "propcrim_ct"],
                aliases=["NPA ID", "Crime / 1k residents", "Crime count"],
                localize=True,
            ),
            style_function=lambda f: {"fillOpacity": 0, "weight": 0},
        ).add_to(m)

        st_folium(m, width="100%", height=520)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Network Metrics Explorer
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Street Network Metrics vs. Property Crime Rate")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.selectbox(
            "Network metric",
            list(METRIC_LABELS.keys()),
            format_func=lambda k: METRIC_LABELS[k],
        )
    with c2:
        buf_label = st.selectbox("Buffer distance", list(BUFFER_OPTIONS.keys()), index=1)
        buffer_mi = BUFFER_OPTIONS[buf_label]
    with c3:
        chart_type = st.radio("Chart type", ["Scatter", "Bar (top 30)"], horizontal=True)

    with st.spinner("Fetching metric rankings…"):
        try:
            mdf = fetch_top(metric, buffer_mi)
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    if chart_type == "Scatter":
        fig = px.scatter(
            mdf,
            x="value",
            y="propcrim_1k",
            hover_name="npa_id",
            labels={
                "value": METRIC_LABELS[metric],
                "propcrim_1k": "Property Crimes / 1k Residents",
            },
            title=f"{METRIC_LABELS[metric]} ({buf_label}) vs. Property Crime Rate",
            color="propcrim_1k",
            color_continuous_scale="YlOrRd",
            trendline="ols",
        )
        fig.update_layout(height=520, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        top30 = mdf.head(30).sort_values("value", ascending=True)
        fig = px.bar(
            top30,
            x="value",
            y="npa_id",
            orientation="h",
            color="propcrim_1k",
            color_continuous_scale="YlOrRd",
            labels={
                "value": METRIC_LABELS[metric],
                "npa_id": "NPA",
                "propcrim_1k": "Crime/1k",
            },
            title=f"Top 30 NPAs by {METRIC_LABELS[metric]} ({buf_label})",
        )
        fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data table"):
        st.dataframe(mdf, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Spatial Proximity Tool
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Find NPAs Near a Point")
    st.markdown("Enter a WGS84 coordinate (decimal degrees) to retrieve nearby NPAs ranked by crime rate.")

    col1, col2, col3 = st.columns(3)
    with col1:
        lon = st.number_input("Longitude", value=-80.8431, format="%.4f",
                              help="Charlotte uptown ≈ -80.8431")
    with col2:
        lat = st.number_input("Latitude", value=35.2271, format="%.4f",
                              help="Charlotte uptown ≈ 35.2271")
    with col3:
        radius = st.slider("Radius (miles)", min_value=0.25, max_value=5.0,
                           value=1.0, step=0.25)

    if st.button("🔍 Search", type="primary"):
        with st.spinner("Querying PostGIS…"):
            try:
                near_df = fetch_near(lon, lat, radius)
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        if near_df.empty:
            st.warning("No NPAs found within that radius.")
        else:
            near_df["dist_mi"] = near_df["dist_ft"] / 5280
            st.success(
                f"Found **{len(near_df)} NPAs** within {radius} mi "
                f"of ({lat:.4f}, {lon:.4f})"
            )

            # Filter the cached full GeoJSON to just the nearby NPAs
            try:
                full_fc = fetch_npa_geojson()
                nearby_ids = set(near_df["npa_id"].astype(str))
                nearby_fc = {
                    "type": "FeatureCollection",
                    "features": [
                        f for f in full_fc["features"]
                        if str(f["properties"]["npa_id"]) in nearby_ids
                    ],
                }
            except Exception:
                nearby_fc = None

            m2 = folium.Map(location=[lat, lon], zoom_start=13,
                            tiles="OpenStreetMap")
            folium.Marker(
                [lat, lon],
                tooltip="Query point",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m2)

            if nearby_fc and nearby_fc["features"]:
                folium.GeoJson(
                    nearby_fc,
                    tooltip=GeoJsonTooltip(
                        fields=["npa_id", "propcrim_1k"],
                        aliases=["NPA ID", "Crime / 1k"],
                        localize=True,
                    ),
                    style_function=lambda f: {
                        "fillColor": "#d62728",
                        "color": "#7f0000",
                        "weight": 1.5,
                        "fillOpacity": 0.45,
                    },
                ).add_to(m2)

            st_folium(m2, width="100%", height=440)

            display_df = near_df[["npa_id", "dist_mi", "propcrim_1k"]].copy()
            display_df.columns = ["NPA ID", "Distance (mi)", "Crime / 1k"]
            display_df["Distance (mi)"] = display_df["Distance (mi)"].round(3)
            st.dataframe(
                display_df.style.background_gradient(
                    subset=["Crime / 1k"], cmap="YlOrRd"
                ),
                use_container_width=True,
                hide_index=True,
            )
