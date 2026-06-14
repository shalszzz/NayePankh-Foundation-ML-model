import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Expansion Heatmap · NayePankh", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .page-title { font-size:2rem; font-weight:700; color:#1a3a2a; margin-bottom:4px; }
  .page-sub   { font-size:0.95rem; color:#557; margin-bottom:28px; }
  .metric-card {
    background:#f0f7f4; border-radius:12px; padding:20px; text-align:center;
    border:1px solid #d0e8da;
  }
  .metric-card .val { font-size:1.9rem; font-weight:700; color:#2d6a4f; }
  .metric-card .lbl { font-size:0.78rem; color:#667; text-transform:uppercase; letter-spacing:.05em; }
  .section-label {
    font-size:0.72rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#2d6a4f; margin:24px 0 10px 0;
  }
  .priority-card {
    background:white; border:1px solid #e8f0ec; border-left:5px solid #2d6a4f;
    border-radius:8px; padding:14px 16px; margin:8px 0;
  }
  .priority-card h4 { margin:0 0 4px 0; color:#1a3a2a; font-size:1rem; }
  .priority-card p  { margin:0; color:#556; font-size:0.84rem; }
  .cluster-0 { border-left-color: #dc3545 !important; }
  .cluster-1 { border-left-color: #ffc107 !important; }
  .cluster-2 { border-left-color: #28a745 !important; }
  .cluster-3 { border-left-color: #007bff !important; }
  [data-testid="stSidebar"] { background:#0f2419; }
  [data-testid="stSidebar"] * { color:#d4e8de !important; }
  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🗺️ District Expansion Heatmap</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">K-Means clustering on 75 UP districts to identify where NayePankh\'s next expansion would save the most lives.</div>', unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/districts.csv')
    except FileNotFoundError:
        import subprocess, sys
        subprocess.run([sys.executable, 'data/generate_data.py'], check=True)
        return pd.read_csv('data/districts.csv')

@st.cache_data
def run_clustering(df, k=4):
    features = ['poverty_rate_pct','literacy_rate_pct','sanitation_access_pct',
                'food_insecurity_pct','child_malnutrition_pct','female_literacy_pct']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = km.fit_predict(X_scaled)

    # Need score: high poverty + low literacy + low sanitation + high food insecurity
    df['need_score'] = (
        df['poverty_rate_pct'] * 0.3 +
        (100 - df['literacy_rate_pct']) * 0.2 +
        (100 - df['sanitation_access_pct']) * 0.25 +
        df['food_insecurity_pct'] * 0.25
    )

    # PCA for 2D viz
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df['pca_x'] = coords[:, 0]
    df['pca_y'] = coords[:, 1]

    # Label clusters by avg need score
    cluster_need = df.groupby('cluster')['need_score'].mean().sort_values(ascending=False)
    label_map = {}
    labels = ['🔴 Critical Need', '🟠 High Need', '🟡 Moderate Need', '🟢 Lower Need']
    for i, (c, _) in enumerate(cluster_need.items()):
        label_map[c] = labels[i] if i < len(labels) else f'Cluster {c}'
    df['cluster_label'] = df['cluster'].map(label_map)

    return df, features, cluster_need, label_map

with st.spinner("Running K-Means on 75 UP districts..."):
    df = load_data()
    df_c, features, cluster_need, label_map = run_clustering(df, k=4)

# ── KPI Row ──────────────────────────────────────────────────────────────────
critical = df_c[df_c['cluster_label'].str.contains('Critical')]
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="metric-card"><div class="val">75</div><div class="lbl">UP Districts Analysed</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="val">{len(critical)}</div><div class="lbl">Critical Need Districts</div></div>', unsafe_allow_html=True)
with k3:
    active = df_c['nayepankh_active'].sum()
    st.markdown(f'<div class="metric-card"><div class="val">{active}</div><div class="lbl">Currently Active Districts</div></div>', unsafe_allow_html=True)
with k4:
    gap = len(critical) - critical['nayepankh_active'].sum()
    st.markdown(f'<div class="metric-card"><div class="val">{max(gap,0)}</div><div class="lbl">Untapped Critical Districts</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Map ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">District Need Map — Uttar Pradesh</div>', unsafe_allow_html=True)

color_map = {
    '🔴 Critical Need': '#dc3545',
    '🟠 High Need':     '#fd7e14',
    '🟡 Moderate Need': '#ffc107',
    '🟢 Lower Need':    '#28a745',
}

fig_map = px.scatter_mapbox(
    df_c,
    lat='lat', lon='lon',
    color='cluster_label',
    color_discrete_map=color_map,
    size='need_score',
    size_max=20,
    hover_name='district',
    hover_data={
        'poverty_rate_pct': True,
        'literacy_rate_pct': True,
        'sanitation_access_pct': True,
        'food_insecurity_pct': True,
        'nayepankh_active': True,
        'lat': False, 'lon': False
    },
    zoom=6.2,
    center={"lat": 26.8, "lon": 80.9},
    mapbox_style='carto-positron',
    labels={
        'cluster_label': 'Need Level',
        'poverty_rate_pct': 'Poverty (%)',
        'literacy_rate_pct': 'Literacy (%)',
        'sanitation_access_pct': 'Sanitation (%)',
        'food_insecurity_pct': 'Food Insecurity (%)',
        'nayepankh_active': 'NayePankh Active'
    }
)
fig_map.update_layout(
    height=500,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation='v', x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)', bordercolor='#ccc', borderwidth=1)
)
st.plotly_chart(fig_map, use_container_width=True)
st.caption("Bubble size = Need Score (composite of poverty, literacy, sanitation, food insecurity). Red = Highest need.")

st.markdown("---")

# ── Top Priority Districts ───────────────────────────────────────────────────
st.markdown('<div class="section-label">Top 5 Priority Districts for Expansion</div>', unsafe_allow_html=True)
top5 = df_c[df_c['nayepankh_active']==0].nlargest(5, 'need_score')[
    ['district','need_score','poverty_rate_pct','literacy_rate_pct','sanitation_access_pct','food_insecurity_pct','cluster_label']
]

for i, row in enumerate(top5.itertuples(), 1):
    css_n = i - 1
    st.markdown(f"""
    <div class="priority-card cluster-{min(css_n,3)}">
      <h4>#{i} {row.district} — Need Score: {row.need_score:.1f}</h4>
      <p>
        Poverty: {row.poverty_rate_pct}% &nbsp;|&nbsp;
        Literacy: {row.literacy_rate_pct}% &nbsp;|&nbsp;
        Sanitation: {row.sanitation_access_pct}% &nbsp;|&nbsp;
        Food Insecurity: {row.food_insecurity_pct}%
        &nbsp;| Cluster: {row.cluster_label}
      </p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Cluster Profiles ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Cluster Profiles — What Each Group Looks Like</div>', unsafe_allow_html=True)
profile = df_c.groupby('cluster_label')[['poverty_rate_pct','literacy_rate_pct','sanitation_access_pct','food_insecurity_pct','child_malnutrition_pct']].mean().round(1).reset_index()
profile.columns = ['Cluster','Poverty (%)','Literacy (%)','Sanitation (%)','Food Insecurity (%)','Child Malnutrition (%)']
st.dataframe(profile, use_container_width=True, hide_index=True)

# ── PCA Scatter ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">PCA Projection — How Districts Cluster</div>', unsafe_allow_html=True)
fig_pca = px.scatter(
    df_c, x='pca_x', y='pca_y',
    color='cluster_label',
    color_discrete_map=color_map,
    text='district',
    hover_data={'need_score': True, 'poverty_rate_pct': True, 'pca_x': False, 'pca_y': False},
    labels={'pca_x': 'Principal Component 1', 'pca_y': 'Principal Component 2'}
)
fig_pca.update_traces(textposition='top center', textfont_size=7, marker_size=9)
fig_pca.update_layout(
    height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(gridcolor='#eee'), yaxis=dict(gridcolor='#eee'),
    margin=dict(l=0,r=0,t=10,b=10)
)
st.plotly_chart(fig_pca, use_container_width=True)

# ── Feature Radar ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Radar — Critical vs Lower Need Districts</div>', unsafe_allow_html=True)
critical_avg = df_c[df_c['cluster_label'].str.contains('Critical')][features].mean()
lower_avg    = df_c[df_c['cluster_label'].str.contains('Lower')][features].mean()
feat_labels  = ['Poverty','Literacy','Sanitation','Food\nInsecurity','Child\nMalnutrition','Female\nLiteracy']

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(r=critical_avg.values, theta=feat_labels, fill='toself',
                                    name='Critical Need', line_color='#dc3545'))
fig_radar.add_trace(go.Scatterpolar(r=lower_avg.values, theta=feat_labels, fill='toself',
                                    name='Lower Need', line_color='#28a745'))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,100])),
    height=380, paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20,r=20,t=20,b=20)
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("Algorithm: K-Means Clustering (k=4) · Features: Poverty Rate, Literacy, Female Literacy, Sanitation Access, Food Insecurity, Child Malnutrition · Data modelled after NITI Aayog SDG India Index & Census India 2021")
