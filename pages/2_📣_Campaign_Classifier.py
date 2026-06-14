import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Campaign Classifier · NayePankh", page_icon="📣", layout="wide")

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
  .pred-high   { background:#d4edda; border:1px solid #28a745; border-radius:12px; padding:16px; }
  .pred-medium { background:#fff3cd; border:1px solid #ffc107; border-radius:12px; padding:16px; }
  .pred-low    { background:#f8d7da; border:1px solid #dc3545; border-radius:12px; padding:16px; }
  [data-testid="stSidebar"] { background:#0f2419; }
  [data-testid="stSidebar"] * { color:#d4e8de !important; }
  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📣 Campaign Impact Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Predict whether a campaign will be High / Medium / Low impact — before you launch it.</div>', unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/campaigns.csv')
    except FileNotFoundError:
        import subprocess, sys
        subprocess.run([sys.executable, 'data/generate_data.py'], check=True)
        return pd.read_csv('data/campaigns.csv')

@st.cache_data
def train_models(df):
    le_type = LabelEncoder()
    le_plat = LabelEncoder()
    le_mnth = LabelEncoder()
    le_targ = LabelEncoder()

    feature_cols = ['campaign_type','platform','month','has_image','has_video']
    X = df[feature_cols].copy()
    X['campaign_type_enc'] = le_type.fit_transform(X['campaign_type'])
    X['platform_enc']      = le_plat.fit_transform(X['platform'])
    X['month_enc']         = le_mnth.fit_transform(X['month'])
    X_num = X[['campaign_type_enc','platform_enc','month_enc','has_image','has_video']]
    y = le_targ.fit_transform(df['impact_level'])   # High=0, Low=1, Medium=2 (sorted)

    X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':             xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric='mlogloss'),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        cv    = cross_val_score(m, X_num, y, cv=5, scoring='accuracy')
        results[name] = {
            'model': m, 'accuracy': round(accuracy_score(y_test, preds)*100, 1),
            'cv_mean': round(cv.mean()*100, 1), 'cv_std': round(cv.std()*100, 1),
            'preds': preds, 'y_test': y_test
        }
    return results, X_num, y, le_type, le_plat, le_mnth, le_targ

with st.spinner("Training campaign models..."):
    df = load_data()
    results, X_num, y, le_type, le_plat, le_mnth, le_targ = train_models(df)

best_name = max(results, key=lambda k: results[k]['accuracy'])

# ── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
high_pct = round((df['impact_level']=='High').mean()*100, 0)
best_platform = df.groupby('platform')['engagement_rate'].mean().idxmax()
best_type = df.groupby('campaign_type')['engagement_rate'].mean().idxmax()

with k1:
    st.markdown(f'<div class="metric-card"><div class="val">{results[best_name]["accuracy"]}%</div><div class="lbl">Best Accuracy</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="val">{int(high_pct)}%</div><div class="lbl">Campaigns Hit High Impact</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="val">{best_platform}</div><div class="lbl">Best Platform</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="val">{best_type.split()[0]}</div><div class="lbl">Best Campaign Type</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Engagement by Type & Platform ────────────────────────────────────────────
st.markdown('<div class="section-label">Engagement Rate by Campaign Type</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    eng_type = df.groupby('campaign_type')['engagement_rate'].mean().reset_index().sort_values('engagement_rate', ascending=True)
    fig1 = px.bar(eng_type, x='engagement_rate', y='campaign_type', orientation='h',
                  color='engagement_rate', color_continuous_scale=['#d4edda','#1a3a2a'],
                  labels={'engagement_rate':'Avg Engagement Rate','campaign_type':'Type'})
    fig1.update_layout(height=300, showlegend=False, coloraxis_showscale=False,
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       margin=dict(l=0,r=20,t=10,b=10), xaxis=dict(gridcolor='#eee'))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    eng_plat = df.groupby('platform')['engagement_rate'].mean().reset_index().sort_values('engagement_rate', ascending=True)
    fig2 = px.bar(eng_plat, x='engagement_rate', y='platform', orientation='h',
                  color='engagement_rate', color_continuous_scale=['#cce5ff','#004085'],
                  labels={'engagement_rate':'Avg Engagement Rate','platform':'Platform'})
    fig2.update_layout(height=300, showlegend=False, coloraxis_showscale=False,
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       margin=dict(l=0,r=20,t=10,b=10), xaxis=dict(gridcolor='#eee'))
    st.plotly_chart(fig2, use_container_width=True)

# ── Model Accuracy + CV ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">Model Comparison (Accuracy + Cross-Validation)</div>', unsafe_allow_html=True)
acc_data = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Accuracy (%)': [results[k]['accuracy'] for k in results],
    'CV Mean (%)': [results[k]['cv_mean'] for k in results],
    'CV Std (%)': [results[k]['cv_std'] for k in results],
})
st.dataframe(acc_data.sort_values('Test Accuracy (%)', ascending=False), use_container_width=True, hide_index=True)

# ── Impact by month heatmap ───────────────────────────────────────────────────
st.markdown('<div class="section-label">Impact Level by Month & Platform</div>', unsafe_allow_html=True)
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = df.groupby(['month','platform'])['engagement_rate'].mean().unstack(fill_value=0)
pivot = pivot.reindex([m for m in month_order if m in pivot.index])

fig_heat = px.imshow(pivot, color_continuous_scale='Greens',
                     labels=dict(color="Engagement Rate"),
                     aspect='auto')
fig_heat.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0,r=0,t=10,b=10))
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ── Live Prediction ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🔮 Predict Your Next Campaign</div>', unsafe_allow_html=True)
st.write("Fill in your planned campaign details to get an impact prediction before launch:")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    p_type  = st.selectbox("Campaign Type", ['Food Drive','Education Fund','Sanitary Pad Drive','Clothes Donation','Awareness Post','Fundraiser Event'])
    p_plat  = st.selectbox("Platform",      ['Instagram','Facebook','WhatsApp','LinkedIn','Twitter'])
with pc2:
    p_month = st.selectbox("Month",         ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    p_image = st.radio("Will you use an image?", ['Yes','No'], horizontal=True)
with pc3:
    p_video = st.radio("Will you use a video?", ['Yes','No'], horizontal=True)
    st.write("")

if st.button("🔮 Predict Impact", type="primary"):
    rf_model = results['Random Forest']['model']
    try: t_e = le_type.transform([p_type])[0]
    except: t_e = 0
    try: pl_e = le_plat.transform([p_plat])[0]
    except: pl_e = 0
    try: m_e = le_mnth.transform([p_month])[0]
    except: m_e = 0

    sample = np.array([[t_e, pl_e, m_e, 1 if p_image=='Yes' else 0, 1 if p_video=='Yes' else 0]])
    pred_enc = rf_model.predict(sample)[0]
    probs    = rf_model.predict_proba(sample)[0]
    classes  = le_targ.classes_   # ['High','Low','Medium']

    pred_label = classes[pred_enc]
    prob_dict  = {cls: round(prob*100,1) for cls, prob in zip(classes, probs)}

    css = 'pred-high' if pred_label=='High' else ('pred-medium' if pred_label=='Medium' else 'pred-low')
    emoji = '🟢' if pred_label=='High' else ('🟡' if pred_label=='Medium' else '🔴')

    st.markdown(f"""
    <div class="{css}">
      <strong>{emoji} Predicted Impact: {pred_label}</strong><br>
      Confidence — High: {prob_dict.get('High',0)}% &nbsp;|&nbsp;
      Medium: {prob_dict.get('Medium',0)}% &nbsp;|&nbsp;
      Low: {prob_dict.get('Low',0)}%
    </div>""", unsafe_allow_html=True)

    # Tips
    tips = {
        'Low': "💡 Tips: Try switching to Instagram or adding a video. Food Drive & Sanitary Pad campaigns tend to perform best. Consider launching in Aug–Oct for higher engagement.",
        'Medium': "💡 Tips: Adding a video could push this to High impact. Make sure to use a compelling image and post in the evening.",
        'High': "✅ Great choice! This campaign has strong predicted engagement. Boost it with a WhatsApp broadcast to your volunteer network."
    }
    st.info(tips[pred_label])

st.markdown("---")
st.caption("Model: Random Forest Classifier · Features: Campaign Type, Platform, Month, Image Used, Video Used")
