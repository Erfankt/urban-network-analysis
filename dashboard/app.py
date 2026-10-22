"""
dashboard/app.py
Urban Street Network & Crime — Streamlit Dashboard
Pulls live data from the FastAPI service at http://localhost:8000
"""

import requests
import pandas as pd
import geopandas as gpd
import folium
from folium import Choropleth, GeoJsonTooltip
from streamlit_folium import st_folium
import plotly.express as px
import streamlit as st

API = "http://localhost:8000"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Charlotte NPA Crime & Street Network",
    page_icon="🏙️",
    layout="wide",
)

st.title("🏙️ Charlotte NPA — Crime & Street Network Dashboard")
st.caption("Data: Kefayat & Thill (2025), ISPRS Int. J. Geo-Information 14(5), 200 · API: localhost:8000")

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_npa_list():
    r = requests.get(f"{API}/npa", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=300)
def fetch_npa_detail(npa_id: str):
    r = requests.get(f"{API}/npa/{npa_id}", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fetch_top(metric: str, buffer_mi: float, n: int = 371):
    r = requests.get(f"{API}/metrics/top",
                     params={"metric": metric, "buffer_mi": buffer_mi, "n": n},
                     timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def fetch_near(lon: float, lat: float, radius_mi: float):
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
# TAB 1 — Crime Choropleth Map
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Property Crime Rate by Neighborhood Planning Area")

    with st.spinner("Loading NPA data from API…"):
        try:
            df = fetch_npa_list()
        except Exception as e:
            st.error(f"Could not reach API: {e}")
            st.stop()

    col_a, col_b = st.columns([3, 1])

    with col_b:
        st.metric("Total NPAs", len(df))
        st.metric("Max crime/1k", f"{df['propcrim_1k'].max():.1f}")
        st.metric("Median crime/1k", f"{df['propcrim_1k'].median():.1f}")
        st.metric("Min crime/1k", f"{df['propcrim_1k'].min():.1f}")

        st.markdown("---")
        st.markdown("**Top 10 highest-crime NPAs**")
        top10 = df.nlargest(10, "propcrim_1k")[["npa_id", "propcrim_1k"]].reset_index(drop=True)
        top10.columns = ["NPA", "Crime/1k"]
        st.dataframe(top10, use_container_width=True, hide_index=True)

    with col_a:
        progress = st.progress(0, text="Fetching NPA geometries…")
        features = []
        npa_ids = df["npa_id"].tolist()
        for i, npa_id in enumerate(npa_ids[:50]):
            try:
                detail = fetch_npa_detail(str(npa_id))
                features.append({
                    "npa_id": npa_id,
                    "propcrim_1k": detail.get("propcrim_1k"),
                    "geometry": detail["geometry"],
                })
            except Exception:
                pass
            progress.progress((i + 1) / min(len(npa_ids), 50),
                              text=f"Loaded {i+1}/50 NPAs…")
        progress.empty()

        if features:
            from shapely.geometry import shape

            gdf = gpd.GeoDataFrame(
                [{**f, "geometry": shape(f["geometry"])} for f in features],
                crs="EPSG:2264",
            ).to_crs("EPSG:4326")

            m = folium.Map(location=[35.23, -80.84], zoom_start=11,
                           tiles="OpenStreetMap")
            Choropleth(
                geo_data=gdf.__geo_interface__,
                data=gdf,
                columns=["npa_id", "propcrim_1k"],
                key_on="feature.properties.npa_id",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.3,
                legend_name="Property Crimes per 1,000 Residents",
                nan_fill_color="lightgray",
            ).add_to(m)

            folium.GeoJson(
                gdf.__geo_interface__,
                tooltip=GeoJsonTooltip(
                    fields=["npa_id", "propcrim_1k"],
                    aliases=["NPA ID", "Crime / 1k residents"],
                    localize=True,
                ),
                style_function=lambda f: {"fillOpacity": 0, "weight": 0},
            ).add_to(m)

            st_folium(m, width="100%", height=520)
            st.caption("⚠️ Showing first 50 NPAs for demo speed. Full dataset: 371 NPAs.")
        else:
            st.warning("No geometry data returned from API.")

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
            st.success(f"Found **{len(near_df)} NPAs** within {radius} mi of ({lat:.4f}, {lon:.4f})")

            m2 = folium.Map(location=[lat, lon], zoom_start=13,
                            tiles="OpenStreetMap")
            folium.Marker(
                [lat, lon],
                tooltip="Query point",
                icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
            ).add_to(m2)

            for _, row in near_df.iterrows():
                try:
                    detail = fetch_npa_detail(str(row["npa_id"]))
                    from shapely.geometry import shape
                    s = shape(detail["geometry"])
                    cy, cx = s.centroid.y, s.centroid.x
                    folium.CircleMarker(
                        [cy, cx],
                        radius=8,
                        color="#d62728",
                        fill=True,
                        fill_opacity=0.6,
                        tooltip=f"NPA {row['npa_id']} | {row['dist_mi']:.2f} mi | crime/1k: {row['propcrim_1k']}",
                    ).add_to(m2)
                except Exception:
                    pass

            st_folium(m2, width="100%", height=440)

            display_df = near_df[["npa_id", "dist_mi", "propcrim_1k"]].copy()
            display_df.columns = ["NPA ID", "Distance (mi)", "Crime / 1k"]
            display_df["Distance (mi)"] = display_df["Distance (mi)"].round(3)
            st.dataframe(
                display_df.style.background_gradient(subset=["Crime / 1k"], cmap="YlOrRd"),
                use_container_width=True,
                hide_index=True,
            )
