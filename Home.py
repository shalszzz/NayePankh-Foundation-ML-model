import streamlit as st

st.set_page_config(
    page_title="NayePankh ML Intelligence",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d6a4f 60%, #52b788 100%);
    border-radius: 16px;
    padding: 48px 40px 40px 40px;
    color: white;
    margin-bottom: 32px;
  }
  .hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    margin: 0 0 8px 0;
    line-height: 1.2;
  }
  .hero p  { font-size: 1.05rem; opacity: 0.88; max-width: 620px; margin-top: 12px; }
  .hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  .card {
    background: white;
    border: 1px solid #e8f0ec;
    border-radius: 14px;
    padding: 28px 24px;
    margin-bottom: 16px;
    transition: box-shadow .2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  .card:hover { box-shadow: 0 6px 24px rgba(45,106,79,0.13); }
  .card h3 { font-size: 1.1rem; font-weight: 700; color: #1a3a2a; margin: 0 0 6px 0; }
  .card p  { font-size: 0.88rem; color: #556; margin: 0; line-height: 1.55; }
  .card .icon { font-size: 1.8rem; margin-bottom: 10px; }

  .stat-row { display: flex; gap: 16px; margin: 28px 0; }
  .stat-box {
    flex: 1;
    background: #f0f7f4;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }
  .stat-box .num { font-size: 2rem; font-weight: 700; color: #2d6a4f; }
  .stat-box .lbl { font-size: 0.78rem; color: #667; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

  .gap-tag {
    display: inline-block;
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.8rem;
    color: #7a5000;
    margin: 4px 4px 4px 0;
  }
  .section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #2d6a4f;
    margin-bottom: 12px;
  }
  .insight-row {
    border-left: 4px solid #2d6a4f;
    padding-left: 16px;
    margin: 12px 0;
    font-size: 0.92rem;
    color: #334;
    line-height: 1.6;
  }

  [data-testid="stSidebar"] { background: #0f2419; }
  [data-testid="stSidebar"] * { color: #d4e8de !important; }
  [data-testid="stSidebar"] .stRadio label { color: #d4e8de !important; }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">🕊️ NayePankh Foundation · ML Intelligence Suite</div>
  <h1>Data-Driven Wings<br>for Social Impact</h1>
  <p>Three machine learning models built specifically for NayePankh — to retain more donors,
     predict campaign impact before launch, and identify where India needs help most.</p>
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-box"><div class="num">2L+</div><div class="lbl">Lives Impacted</div></div>
  <div class="stat-box"><div class="num">3</div><div class="lbl">ML Models Built</div></div>
  <div class="stat-box"><div class="num">75</div><div class="lbl">UP Districts Analysed</div></div>
  <div class="stat-box"><div class="num">500</div><div class="lbl">Donor Records Modelled</div></div>
</div>
""", unsafe_allow_html=True)

# ── MODULE CARDS ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">The Three Modules</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
      <div class="icon">💰</div>
      <h3>Module 1 — Donor Predictor</h3>
      <p>Uses RFM analysis + XGBoost to predict which donors will give again and how much.
         Segments all donors into Champion / Loyal / At-Risk / Lost buckets.</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
      <div class="icon">📣</div>
      <h3>Module 2 — Campaign Classifier</h3>
      <p>Predicts campaign impact (High / Medium / Low) before you launch it —
         based on type, platform, content format, and timing.</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
      <div class="icon">🗺️</div>
      <h3>Module 3 — Expansion Heatmap</h3>
      <p>K-Means clustering on 75 UP districts using poverty, literacy, sanitation,
         and food insecurity data to find where NayePankh should expand next.</p>
    </div>""", unsafe_allow_html=True)

# ── GAPS IDENTIFIED ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Current Gaps in NayePankh Operations</div>', unsafe_allow_html=True)

gaps = [
    "No donor retention model — lapsed donors receive the same outreach as active ones",
    "Campaigns are launched without any pre-launch performance prediction",
    "Expansion decisions are instinct-based, not driven by district-level need data",
    "No data collection pipeline on the website (no email capture, no behaviour tracking)",
    "Donation page has no urgency signal or personalisation — same CTA for all visitors",
    "Social media performance is not aggregated or compared across platforms"
]

for g in gaps:
    st.markdown(f'<span class="gap-tag">⚠ {g}</span>', unsafe_allow_html=True)

# ── HOW ML HELPS ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">How Machine Learning Closes These Gaps</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="insight-row">
      <strong>Donor Churn Recovery:</strong> Identifying At-Risk donors 60 days before they lapse
      and sending personalised re-engagement can recover an estimated 20–30% of lost revenue —
      something no Indian NGO of this size currently does.
    </div>
    <div class="insight-row">
      <strong>Campaign ROI before spending:</strong> A classifier trained on past performance
      tells you which content format + platform combination will drive the most volunteers
      and donations — before you spend a rupee on promotion.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="insight-row">
      <strong>Data-backed expansion:</strong> Instead of expanding to cities where volunteers
      happen to be, NayePankh can use district clustering to find where poverty + poor sanitation
      + food insecurity converge — the highest-impact entry points.
    </div>
    <div class="insight-row">
      <strong>Website improvement:</strong> Adding a real-time impact counter, donation progress bar,
      and city-based personalisation on the donate page could lift conversion by 15–25% based
      on NGO sector benchmarks.
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER NOTE ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built for NayePankh Foundation ML Intern Selection Task · All models trained on realistic synthetic data modelled after public NGO and Census India statistics · Ready to plug into real NayePankh data")
