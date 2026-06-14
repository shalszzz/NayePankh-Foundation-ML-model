import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Donor Predictor · NayePankh", page_icon="💰", layout="wide")

# ── Shared CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .page-title { font-size:2rem; font-weight:700; color:#1a3a2a; margin-bottom:4px; }
  .page-sub   { font-size:0.95rem; color:#557; margin-bottom:28px; }
  .metric-card {
    background:#f0f7f4; border-radius:12px; padding:20px; text-align:center;
    border: 1px solid #d0e8da;
  }
  .metric-card .val { font-size:1.9rem; font-weight:700; color:#2d6a4f; }
  .metric-card .lbl { font-size:0.78rem; color:#667; text-transform:uppercase; letter-spacing:.05em; }
  .segment-chip {
    display:inline-block; border-radius:20px; padding:4px 14px;
    font-size:0.82rem; font-weight:600; margin:3px;
  }
  .champion  { background:#d4edda; color:#155724; }
  .loyal     { background:#cce5ff; color:#004085; }
  .at-risk   { background:#fff3cd; color:#856404; }
  .lost      { background:#f8d7da; color:#721c24; }
  .section-label {
    font-size:0.72rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#2d6a4f; margin:24px 0 10px 0;
  }
  [data-testid="stSidebar"] { background:#0f2419; }
  [data-testid="stSidebar"] * { color:#d4e8de !important; }
  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">💰 Donor Retention Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">RFM analysis + ML models to predict which donors will give again — and how much.</div>', unsafe_allow_html=True)

# ── Load / Generate Data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/donors.csv')
    except FileNotFoundError:
        import subprocess, sys
        subprocess.run([sys.executable, 'data/generate_data.py'], check=True)
        df = pd.read_csv('data/donors.csv')
    return df

@st.cache_data
def train_models(df):
    le_type = LabelEncoder()
    le_camp = LabelEncoder()
    le_city = LabelEncoder()

    X = df[['recency_days', 'frequency', 'avg_donation_inr']].copy()
    X['donor_type_enc'] = le_type.fit_transform(df['donor_type'])
    X['campaign_enc']   = le_camp.fit_transform(df['acquisition_campaign'])
    X['city_enc']       = le_city.fit_transform(df['city'])
    y = df['will_donate_again']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':             xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric='logloss'),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, m in models.items():
        if name == 'Logistic Regression':
            m.fit(X_train_s, y_train)
            preds = m.predict(X_test_s)
        else:
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
        results[name] = {
            'model': m,
            'accuracy': round(accuracy_score(y_test, preds) * 100, 1),
            'preds': preds,
            'y_test': y_test
        }
    return results, X_test, X_test_s, y_test, scaler, le_type, le_camp, le_city

def rfm_segment(row):
    r, f, m = row['recency_days'], row['frequency'], row['avg_donation_inr']
    if r < 30 and f >= 5 and m > 1000:  return 'Champion'
    if r < 90 and f >= 3:               return 'Loyal'
    if r > 180 and f >= 2:              return 'At-Risk'
    return 'Lost'

with st.spinner("Training 4 ML models..."):
    df = load_data()
    df['segment'] = df.apply(rfm_segment, axis=1)
    results, X_test, X_test_s, y_test, scaler, le_type, le_camp, le_city = train_models(df)

best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_acc = results[best_model_name]['accuracy']

# ── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
seg_counts = df['segment'].value_counts()

with k1:
    st.markdown(f'<div class="metric-card"><div class="val">{best_acc}%</div><div class="lbl">Best Model Accuracy</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="val">{seg_counts.get("Champion",0)}</div><div class="lbl">Champion Donors</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="val">{seg_counts.get("At-Risk",0)}</div><div class="lbl">At-Risk Donors</div></div>', unsafe_allow_html=True)
with k4:
    champ_value = int(df[df['segment']=='Champion']['avg_donation_inr'].mean())
    st.markdown(f'<div class="metric-card"><div class="val">₹{champ_value:,}</div><div class="lbl">Avg Champion Donation</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Model Comparison ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
acc_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy (%)': [results[k]['accuracy'] for k in results]
}).sort_values('Accuracy (%)', ascending=True)

fig_acc = px.bar(
    acc_df, x='Accuracy (%)', y='Model', orientation='h',
    color='Accuracy (%)', color_continuous_scale=['#d4edda','#2d6a4f'],
    text='Accuracy (%)'
)
fig_acc.update_traces(texttemplate='%{text}%', textposition='outside')
fig_acc.update_layout(
    height=280, showlegend=False, coloraxis_showscale=False,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0,r=60,t=10,b=10),
    xaxis=dict(range=[max(0, acc_df['Accuracy (%)'].min() - 10), 100], gridcolor='#eee'),
    yaxis=dict(gridcolor='#eee')
)
st.plotly_chart(fig_acc, use_container_width=True)
st.caption(f"Best model: **{best_model_name}** at **{best_acc}%** accuracy")

st.markdown("---")

# ── Donor Segments ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Donor Segmentation (RFM)</div>', unsafe_allow_html=True)
col_pie, col_seg = st.columns([1, 1])

with col_pie:
    seg_df = df['segment'].value_counts().reset_index()
    seg_df.columns = ['Segment', 'Count']
    color_map = {'Champion':'#2d6a4f','Loyal':'#52b788','At-Risk':'#ffc107','Lost':'#dc3545'}
    fig_pie = px.pie(seg_df, values='Count', names='Segment',
                     color='Segment', color_discrete_map=color_map, hole=0.45)
    fig_pie.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=10),
                          paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_seg:
    seg_stats = df.groupby('segment').agg(
        Count=('donor_id','count'),
        Avg_Donation=('avg_donation_inr','mean'),
        Avg_Frequency=('frequency','mean')
    ).round(0).reset_index()
    seg_stats.columns = ['Segment','Count','Avg Donation (₹)','Avg Frequency']
    st.dataframe(seg_stats, use_container_width=True, hide_index=True)

    st.markdown("**What to do with each segment:**")
    st.markdown("""
    <span class="segment-chip champion">Champion</span> Send personal thank-you + impact report<br>
    <span class="segment-chip loyal">Loyal</span> Offer recurring donation option<br>
    <span class="segment-chip at-risk">At-Risk</span> Re-engage within 60 days with campaign update<br>
    <span class="segment-chip lost">Lost</span> Annual newsletter only — low-cost outreach
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Scatter: RFM 3D feel ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">Recency vs Frequency vs Donation Size</div>', unsafe_allow_html=True)
fig_scatter = px.scatter(
    df, x='recency_days', y='frequency', size='avg_donation_inr',
    color='segment', color_discrete_map=color_map,
    hover_data=['avg_donation_inr','city','acquisition_campaign'],
    labels={'recency_days':'Days Since Last Donation','frequency':'Number of Donations'},
    size_max=30
)
fig_scatter.update_layout(
    height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(gridcolor='#eee'), yaxis=dict(gridcolor='#eee')
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ── Live Prediction Widget ───────────────────────────────────────────────────
st.markdown('<div class="section-label">🔮 Live Donor Prediction</div>', unsafe_allow_html=True)
st.write("Enter a donor's details below to predict if they'll donate again:")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    p_recency  = st.slider("Days since last donation", 1, 500, 45)
    p_freq     = st.slider("Total donations made", 1, 20, 3)
with pc2:
    p_monetary = st.number_input("Average donation (₹)", min_value=50, max_value=50000, value=500, step=50)
    p_type     = st.selectbox("Donor type", ['Individual','Corporate','Student','NRI'])
with pc3:
    p_campaign = st.selectbox("How they found NayePankh", ['Food Drive','Education Fund','Sanitary Pad Drive','Clothes Donation','Social Media','Word of Mouth'])
    p_city     = st.selectbox("City", ['Kanpur','Ghaziabad','Lucknow','Delhi','Mumbai','Other'])

if st.button("🔮 Predict", type="primary"):
    xgb_model = results['XGBoost']['model']
    try:
        t_enc = le_type.transform([p_type])[0]
    except: t_enc = 0
    try:
        c_enc = le_camp.transform([p_campaign])[0]
    except: c_enc = 0
    try:
        ci_enc = le_city.transform([p_city])[0]
    except: ci_enc = 0

    sample = np.array([[p_recency, p_freq, p_monetary, t_enc, c_enc, ci_enc]])
    prob   = xgb_model.predict_proba(sample)[0][1]
    pred   = xgb_model.predict(sample)[0]

    seg = rfm_segment({'recency_days': p_recency, 'frequency': p_freq, 'avg_donation_inr': p_monetary})
    next_amt = int(p_monetary * (0.9 + prob * 0.5)) if pred == 1 else 0

    r1, r2, r3 = st.columns(3)
    with r1:
        color = "🟢" if pred == 1 else "🔴"
        st.metric("Will Donate Again?", f"{color} {'YES' if pred==1 else 'NO'}")
    with r2:
        st.metric("Confidence", f"{prob*100:.1f}%")
    with r3:
        st.metric("Predicted Next Donation", f"₹{next_amt:,}" if next_amt > 0 else "—")

    css_class = seg.lower().replace('-','').replace(' ','')
    st.markdown(f'Donor Segment: <span class="segment-chip {css_class}">{seg}</span>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Model: XGBoost Classifier · Features: Recency, Frequency, Monetary (RFM) + Donor Type + City + Acquisition Channel")
