import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pune Rice Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace !important;
}
.stApp { background-color: #0a0f1e; color: #e2e8f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027 0%, #0d1b2a 100%);
    border-right: 1px solid #1e3a2f;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #4ade80 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #0d1b2a;
    border: 1px solid #1e3a2f;
    border-radius: 10px;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #4ade80 !important; font-size: 11px !important; letter-spacing: 0.15em; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 28px !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px;
    letter-spacing: 0.1em;
    color: #64748b !important;
    background: transparent;
    border: 1px solid #1e3a2f !important;
    border-radius: 6px !important;
    margin-right: 6px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4ade80 !important;
    border-color: #4ade80 !important;
    background: #4ade8011 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #4ade80) !important;
    color: #0a0f1e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    border: none !important;
    border-radius: 8px !important;
    width: 100%;
    padding: 14px !important;
    font-size: 14px !important;
    box-shadow: 0 0 20px #4ade8033;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

/* Sliders */
[data-testid="stSlider"] > div > div { background: #ffffff !important; }

/* Inputs */
[data-testid="stNumberInput"] input, .stTextInput input {
    background: #060d18 !important;
    border: 1px solid #1e3a2f !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 6px !important;
}

/* Select boxes */
[data-testid="stMultiSelect"] div, [data-testid="stSelectbox"] div {
    background: #060d18 !important;
    border-color: #1e3a2f !important;
    color: #e2e8f0 !important;
}

/* Divider */
hr { border-color: #1e3a2f !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e3a2f !important; border-radius: 8px; }

/* Headers */
h1, h2, h3 { color: #4ade80 !important; font-family: 'IBM Plex Mono', monospace !important; }

/* Result box */
.yield-result {
    background: linear-gradient(135deg, #0d1b2a, #0f2a1e);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
}

/* Info box override */
[data-testid="stInfo"] { background: #0d1b2a !important; border-color: #4ade80 !important; color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model & data ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("yield_model.pkl")
    return bundle["model"], bundle["scaler"], bundle["features"]

@st.cache_data
def load_district_data():
    return pd.read_csv("district_yields.csv")

model, scaler, feature_cols = load_model()
df = load_district_data()

YEARS = sorted(df["year"].unique().tolist())
DISTRICTS = sorted(df["district"].unique().tolist())

# ── Helpers ────────────────────────────────────────────────────────────────────
def yield_color(v):
    if v >= 3200: return "#4ade80"
    if v >= 2900: return "#86efac"
    if v >= 2700: return "#fbbf24"
    if v >= 2500: return "#fb923c"
    return "#f87171"

def yield_label(v):
    if v >= 3200: return "🟢 Excellent"
    if v >= 2900: return "🟡 Good"
    if v >= 2700: return "🟠 Average"
    if v >= 2500: return "🔴 Below Average"
    return "⛔ Poor"

def predict_yield(features_dict, lat, lon):
    row = [features_dict[c] if c in features_dict else (lat if c == "lat" else lon)
           for c in feature_cols]
    X = np.array(row).reshape(1, -1)
    X_sc = scaler.transform(X)
    return float(model.predict(X_sc)[0])

FEATURE_DEFAULTS = dict(
    B2=0.069, B3=0.084, B4=0.091, B5=0.126, B8=0.229, B11=0.153,
    EVI=0.151, GCVI=1.433, NDRE=0.216, NDVI=0.432, NDWI=-0.028, SAVI=0.184
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 PUNE YIELD INTEL")
    st.markdown("---")

    st.markdown("### 📅 Year Filter")
    selected_year = st.select_slider("", options=YEARS, value=2023)

    st.markdown("### 🗺️ Districts")
    selected_districts = st.multiselect(
        "Select up to 6 districts",
        DISTRICTS,
        default=["Pune City", "Baramati", "Junnar"],
        max_selections=6
    )
    if not selected_districts:
        selected_districts = ["Pune City"]

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.markdown("""
    <div style='font-size:11px;color:#475569;line-height:1.9'>
    Algorithm: Gradient Boosting<br>
    R² Score: <span style='color:#4ade80'>0.924</span><br>
    MAE: <span style='color:#4ade80'>98 kg/ha</span><br>
    CV Folds: <span style='color:#4ade80'>5-fold</span><br>
    Samples: <span style='color:#4ade80'>500</span><br>
    Features: <span style='color:#4ade80'>14</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#1e3a2f;letter-spacing:0.1em'>
    SENTINEL-2 · HYPERSPECTRAL<br>
    PUNE AGRI INTELLIGENCE
    </div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f2027,#1a3a2a,#0f2027);
    border:1px solid #1e3a2f;border-radius:12px;padding:20px 28px;
    margin-bottom:24px;display:flex;align-items:center;gap:16px'>
  <span style='font-size:40px'>🌾</span>
  <div>
    <div style='font-size:22px;font-weight:700;color:#4ade80;letter-spacing:0.05em'>
      PUNE RICE YIELD INTELLIGENCE
    </div>
    <div style='font-size:11px;color:#64748b;letter-spacing:0.15em'>
      HYPERSPECTRAL · SENTINEL-2 · GBM MODEL · R²=0.924 · 14 DISTRICTS · 2019–2024
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI strip ──────────────────────────────────────────────────────────────────
year_data = df[df["year"] == selected_year]
best_row  = year_data.loc[year_data["yield_kg_ha"].idxmax()]
worst_row = year_data.loc[year_data["yield_kg_ha"].idxmin()]
avg_yield = year_data["yield_kg_ha"].mean()

prev_year_data = df[df["year"] == max(YEARS[0], selected_year - 1)]
prev_avg = prev_year_data["yield_kg_ha"].mean() if not prev_year_data.empty else avg_yield

c1, c2, c3, c4 = st.columns(4)
c1.metric("DISTRICTS", len(DISTRICTS), "Pune Division")
c2.metric(f"BEST {selected_year}", f"{best_row['yield_kg_ha']:,.0f}", best_row["district"])
c3.metric(f"PUNE AVG {selected_year}", f"{avg_yield:,.0f} kg/ha", f"{avg_yield - prev_avg:+.0f} vs prev yr")
c4.metric("MODEL MAE", "±98 kg/ha", "GBM · 5-fold CV")

st.markdown("")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_map, tab_trend, tab_compare, tab_predict = st.tabs([
    "📍  MAP VIEW", "📈  YEAR TRENDS", "📊  DISTRICT COMPARE", "🔮  PREDICT YIELD"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ════════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown(f"### District Yield Map — {selected_year}")

    map_df = year_data.copy()
    map_df["label"] = map_df["yield_kg_ha"].apply(yield_label)
    map_df["color"] = map_df["yield_kg_ha"].apply(yield_color)
    map_df["size"]  = ((map_df["yield_kg_ha"] - map_df["yield_kg_ha"].min()) /
                       (map_df["yield_kg_ha"].max() - map_df["yield_kg_ha"].min()) * 25 + 15)

    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat", lon="lon",
        size="size",
        color="yield_kg_ha",
        color_continuous_scale=["#f87171","#fb923c","#fbbf24","#86efac","#4ade80"],
        range_color=[map_df["yield_kg_ha"].min()-100, map_df["yield_kg_ha"].max()+100],
        hover_name="district",
        hover_data={"yield_kg_ha":":.0f", "lat":False, "lon":False, "size":False, "label":True},
        zoom=8.2,
        center={"lat": 18.55, "lon": 74.1},
        mapbox_style="carto-darkmatter",
        labels={"yield_kg_ha": "Yield (kg/ha)", "label": "Grade"},
        height=540
    )
    fig_map.update_layout(
        paper_bgcolor="#0a0f1e",
        font_color="#94a3b8",
        coloraxis_colorbar=dict(
            title=dict(text="kg/ha", font=dict(color="#4ade80", family="IBM Plex Mono")),
            tickfont=dict(color="#94a3b8", family="IBM Plex Mono")
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Table below map
    st.markdown(f"#### Rankings for {selected_year}")
    display_df = (year_data[["district","yield_kg_ha"]]
                  .sort_values("yield_kg_ha", ascending=False)
                  .reset_index(drop=True))
    display_df.index += 1
    display_df.columns = ["District", "Yield (kg/ha)"]
    display_df["Grade"] = display_df["Yield (kg/ha)"].apply(yield_label)
    display_df["Yield (kg/ha)"] = display_df["Yield (kg/ha)"].round(0).astype(int)
    st.dataframe(display_df, use_container_width=True, height=420)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ════════════════════════════════════════════════════════════════════════════════
with tab_trend:
    st.markdown("### Year-wise Rice Yield Trend")

    trend_df = df[df["district"].isin(selected_districts)]

    colors = ["#4ade80","#60a5fa","#f59e0b","#f472b6","#a78bfa","#34d399"]

    fig_line = go.Figure()
    for i, dist in enumerate(selected_districts):
        d = trend_df[trend_df["district"] == dist].sort_values("year")
        fig_line.add_trace(go.Scatter(
            x=d["year"], y=d["yield_kg_ha"],
            mode="lines+markers",
            name=dist,
            line=dict(color=colors[i % len(colors)], width=2.5),
            marker=dict(size=8, symbol="circle"),
            hovertemplate=f"<b>{dist}</b><br>Year: %{{x}}<br>Yield: %{{y:.0f}} kg/ha<extra></extra>"
        ))

    fig_line.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#060d18",
        font=dict(family="IBM Plex Mono", color="#94a3b8"),
        xaxis=dict(gridcolor="#1e3a2f", tickfont=dict(color="#94a3b8"), title="Year", title_font_color="#4ade80"),
        yaxis=dict(gridcolor="#1e3a2f", tickfont=dict(color="#94a3b8"), title="Yield (kg/ha)", title_font_color="#4ade80"),
        legend=dict(bgcolor="#0d1b2a", bordercolor="#1e3a2f", font=dict(color="#94a3b8")),
        height=420,
        margin=dict(l=60, r=20, t=20, b=60),
        hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Summary stats
    st.markdown("#### District Summary")
    cols = st.columns(min(len(selected_districts), 3))
    for i, dist in enumerate(selected_districts):
        d = df[df["district"] == dist]
        avg = d["yield_kg_ha"].mean()
        chg = ((d[d["year"]==2024]["yield_kg_ha"].values[0] -
                d[d["year"]==2019]["yield_kg_ha"].values[0]) /
               d[d["year"]==2019]["yield_kg_ha"].values[0] * 100)
        best_yr = d.loc[d["yield_kg_ha"].idxmax(), "year"]
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background:#0d1b2a;border:1px solid {colors[i]}33;border-radius:10px;padding:14px;margin-bottom:8px'>
              <div style='color:{colors[i]};font-size:12px;font-weight:700'>{dist}</div>
              <div style='font-size:22px;font-weight:700;color:#e2e8f0;margin:4px 0'>{avg:.0f} <span style='font-size:11px;color:#475569'>avg kg/ha</span></div>
              <div style='font-size:11px;color:{"#4ade80" if chg>=0 else "#f87171"}'>
                {"▲" if chg>=0 else "▼"} {abs(chg):.1f}% (2019→2024)
              </div>
              <div style='font-size:10px;color:#475569;margin-top:4px'>Best year: {best_yr}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE
# ════════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown(f"### District Comparison — {selected_year}")

    col_bar, col_heat = st.columns([1, 1])

    with col_bar:
        bar_df = year_data.sort_values("yield_kg_ha", ascending=True)
        bar_colors = [yield_color(v) for v in bar_df["yield_kg_ha"]]

        fig_bar = go.Figure(go.Bar(
            x=bar_df["yield_kg_ha"], y=bar_df["district"],
            orientation="h",
            marker_color=bar_colors,
            text=bar_df["yield_kg_ha"].round(0).astype(int),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
            hovertemplate="<b>%{y}</b><br>%{x:.0f} kg/ha<extra></extra>"
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0a0f1e", plot_bgcolor="#060d18",
            font=dict(family="IBM Plex Mono", color="#94a3b8"),
            xaxis=dict(gridcolor="#1e3a2f", range=[2200, 3700], title="kg/ha", title_font_color="#4ade80"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=480, margin=dict(l=10, r=60, t=20, b=40),
            title=dict(text=f"Ranking {selected_year}", font=dict(color="#4ade80", size=13))
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_heat:
        pivot = df.pivot(index="district", columns="year", values="yield_kg_ha").round(0)

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(y) for y in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[0,"#f87171"],[0.25,"#fb923c"],[0.5,"#fbbf24"],[0.75,"#86efac"],[1,"#4ade80"]],
            text=pivot.values.round(0).astype(int),
            texttemplate="%{text}",
            textfont=dict(size=10, color="white", family="IBM Plex Mono"),
            hovertemplate="<b>%{y}</b> · %{x}<br>%{z:.0f} kg/ha<extra></extra>",
            showscale=True,
            colorbar=dict(tickfont=dict(color="#94a3b8"), title=dict(text="kg/ha", font=dict(color="#4ade80")))
        ))
        fig_heat.update_layout(
            paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e",
            font=dict(family="IBM Plex Mono", color="#94a3b8"),
            height=480, margin=dict(l=10, r=20, t=20, b=40),
            xaxis=dict(tickfont=dict(color="#94a3b8")),
            yaxis=dict(tickfont=dict(color="#94a3b8")),
            title=dict(text="Heatmap All Years", font=dict(color="#4ade80", size=13))
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Full table
    st.markdown("#### Full Data Table")
    full_pivot = df.pivot(index="district", columns="year", values="yield_kg_ha").round(0).astype(int)
    full_pivot["Average"] = full_pivot.mean(axis=1).round(0).astype(int)
    full_pivot = full_pivot.sort_values("Average", ascending=False)
    st.dataframe(full_pivot, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### 🔮 Location-based Yield Predictor")
    st.markdown("Enter GPS coordinates and spectral indices from Google Earth Engine to predict rice yield at any point in Pune.")

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.markdown("#### 📍 Coordinates")
        ci1, ci2 = st.columns(2)
        with ci1:
            lat = st.number_input("Latitude", value=18.52, min_value=17.5, max_value=20.0, step=0.001, format="%.4f")
        with ci2:
            lon = st.number_input("Longitude", value=73.86, min_value=72.5, max_value=76.0, step=0.001, format="%.4f")

        st.markdown("#### 🛰️ Spectral Indices (from GEE)")

        # Group sliders visually
        st.markdown("**Vegetation Indices**")
        sv1, sv2 = st.columns(2)
        with sv1:
            ndvi = st.slider("NDVI", -0.2, 0.9, 0.432, 0.001, help="Normalized Difference Vegetation Index — most important feature")
            evi  = st.slider("EVI",  -0.2, 0.8, 0.151, 0.001, help="Enhanced Vegetation Index")
            savi = st.slider("SAVI", -0.1, 0.8, 0.184, 0.001, help="Soil Adjusted Vegetation Index")
        with sv2:
            ndre = st.slider("NDRE", -0.3, 0.7, 0.216, 0.001, help="Red-Edge NDVI — sensitive to chlorophyll")
            gcvi = st.slider("GCVI", -0.3, 5.5, 1.433, 0.001, help="Green Chlorophyll Vegetation Index")
            ndwi = st.slider("NDWI", -0.5, 0.5, -0.028, 0.001, help="Water Index — negative = dry, positive = wet")

        st.markdown("**Spectral Bands (Reflectance)**")
        sb1, sb2, sb3 = st.columns(3)
        with sb1:
            b2  = st.slider("B2 (Blue)",  0.01, 0.3, 0.069, 0.001)
            b3  = st.slider("B3 (Green)", 0.01, 0.3, 0.084, 0.001)
        with sb2:
            b4  = st.slider("B4 (Red)",   0.01, 0.3, 0.091, 0.001)
            b5  = st.slider("B5 (RedEdge)", 0.01, 0.4, 0.126, 0.001)
        with sb3:
            b8  = st.slider("B8 (NIR)",   0.01, 0.6, 0.229, 0.001)
            b11 = st.slider("B11 (SWIR)", 0.01, 0.4, 0.153, 0.001)

        st.markdown("")
        predict_btn = st.button("⚡ PREDICT RICE YIELD")

    with col_result:
        st.markdown("#### 📊 Prediction Result")

        if predict_btn:
            feat = dict(B2=b2,B3=b3,B4=b4,B5=b5,B8=b8,B11=b11,
                        EVI=evi,GCVI=gcvi,NDRE=ndre,NDVI=ndvi,NDWI=ndwi,SAVI=savi,
                        lat=lat, lon=lon)
            raw = predict_yield(feat, lat, lon)
            result = max(500.0, raw)
            color  = yield_color(result)
            grade  = yield_label(result)

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0d1b2a,#0f2a1e);
                border:2px solid {color};border-radius:14px;padding:36px;
                text-align:center;box-shadow:0 0 40px {color}22;margin-bottom:16px'>
              <div style='font-size:12px;color:#64748b;letter-spacing:0.2em;margin-bottom:10px'>
                PREDICTED RICE YIELD
              </div>
              <div style='font-size:72px;font-weight:700;color:{color};line-height:1'>
                {result:,.0f}
              </div>
              <div style='font-size:18px;color:#64748b;margin-top:6px'>kg / hectare</div>
              <div style='display:inline-block;margin-top:16px;padding:6px 24px;
                background:{color}22;border:1px solid {color};border-radius:20px;
                font-size:14px;color:{color};font-weight:700'>
                {grade}
              </div>
              <div style='margin-top:20px;font-size:11px;color:#475569'>
                📍 {lat:.4f}°N, {lon:.4f}°E &nbsp;|&nbsp; NDVI: {ndvi:.3f} &nbsp;|&nbsp; Confidence: ±98 kg/ha
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result,
                number=dict(suffix=" kg/ha", font=dict(color=color, family="IBM Plex Mono", size=24)),
                gauge=dict(
                    axis=dict(range=[1500, 3800], tickfont=dict(color="#94a3b8")),
                    bar=dict(color=color),
                    bgcolor="#060d18",
                    bordercolor="#1e3a2f",
                    steps=[
                        dict(range=[1500,2500], color="#1a0a0a"),
                        dict(range=[2500,2700], color="#1a1208"),
                        dict(range=[2700,2900], color="#1a1608"),
                        dict(range=[2900,3200], color="#0a1a10"),
                        dict(range=[3200,3800], color="#0a2010"),
                    ],
                    threshold=dict(line=dict(color="#fff",width=2), thickness=0.8, value=result)
                ),
                domain=dict(x=[0,1], y=[0,1])
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0a0f1e",
                font=dict(family="IBM Plex Mono", color="#94a3b8"),
                height=240, margin=dict(l=20,r=20,t=20,b=0)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.markdown("""
            <div style='background:#0d1b2a;border:1px dashed #1e3a2f;border-radius:12px;
                padding:40px;text-align:center;color:#475569'>
              <div style='font-size:48px;margin-bottom:12px'>🛰️</div>
              <div style='font-size:13px'>Adjust the spectral indices on the left<br>and click <b style="color:#4ade80">PREDICT YIELD</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Feature importance chart (always shown)
        st.markdown("#### Feature Importance")
        imp_data = {
            "Feature": ["NDVI","SAVI","GCVI","NDWI","NDRE","EVI","B8","Others"],
            "Importance": [77.6, 7.4, 4.5, 3.3, 1.8, 1.4, 1.2, 2.8]
        }
        fig_imp = go.Figure(go.Bar(
            x=imp_data["Importance"], y=imp_data["Feature"],
            orientation="h",
            marker=dict(
                color=imp_data["Importance"],
                colorscale=[[0,"#1e3a2f"],[1,"#4ade80"]]
            ),
            text=[f"{v}%" for v in imp_data["Importance"]],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=10)
        ))
        fig_imp.update_layout(
            paper_bgcolor="#0a0f1e", plot_bgcolor="#060d18",
            font=dict(family="IBM Plex Mono", color="#94a3b8"),
            xaxis=dict(gridcolor="#1e3a2f", title="%", title_font_color="#4ade80", range=[0,90]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=260, margin=dict(l=10, r=50, t=10, b=30)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.info("""
**How to get spectral indices from GEE:**
1. Open GEE → load Sentinel-2 SR collection
2. Filter by date (kharif season: Jun–Oct)
3. Compute NDVI = (B8-B4)/(B8+B4)
4. Sample pixel at your lat/lon
5. Paste values into the sliders
        """)
