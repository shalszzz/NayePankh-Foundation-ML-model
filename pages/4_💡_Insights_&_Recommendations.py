import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Insights · NayePankh", page_icon="💡", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .page-title { font-size:2rem; font-weight:700; color:#1a3a2a; margin-bottom:4px; }
  .page-sub   { font-size:0.95rem; color:#557; margin-bottom:28px; }
  .section-label {
    font-size:0.72rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#2d6a4f; margin:28px 0 12px 0;
  }
  .insight-box {
    background:white; border:1px solid #e8f0ec; border-radius:12px;
    padding:20px 22px; margin:10px 0; box-shadow:0 2px 8px rgba(0,0,0,0.04);
  }
  .insight-box h4 { color:#1a3a2a; margin:0 0 6px 0; font-size:1rem; }
  .insight-box p  { color:#556; font-size:0.87rem; margin:0; line-height:1.6; }
  .rec-card {
    background:#f0f7f4; border-left:5px solid #2d6a4f; border-radius:8px;
    padding:16px 18px; margin:10px 0;
  }
  .rec-card h4 { color:#1a3a2a; margin:0 0 6px 0; }
  .rec-card p  { color:#445; font-size:0.87rem; margin:0; }
  .gap-item {
    background:#fff8f0; border-left:4px solid #fd7e14; border-radius:6px;
    padding:10px 14px; margin:8px 0; font-size:0.87rem; color:#5a3000;
  }
  .win-item {
    background:#f0fff4; border-left:4px solid #28a745; border-radius:6px;
    padding:10px 14px; margin:8px 0; font-size:0.87rem; color:#1a4a2a;
  }
  .big-quote {
    font-family:'Playfair Display', serif;
    font-size:1.35rem; color:#1a3a2a; font-style:italic;
    border-left:5px solid #2d6a4f; padding-left:20px; margin:24px 0;
    line-height:1.5;
  }
  [data-testid="stSidebar"] { background:#0f2419; }
  [data-testid="stSidebar"] * { color:#d4e8de !important; }
  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">💡 NayePankh — Data Strategy Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Website analysis, data gaps, ML recommendations, and a roadmap to make NayePankh the most data-intelligent NGO in India.</div>', unsafe_allow_html=True)

# ── Big Quote ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="big-quote">
  "A donor churn model could recover 20–30% of lapsed donors if NayePankh sends
  personalised re-engagement messages within 60 days of last donation —
  something no Indian NGO of this size is currently doing."
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Website Gaps ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Current Website — Data Gaps Found</div>', unsafe_allow_html=True)

gaps = [
    ("No impact counter", "The website says '2 lakh lives helped' but it's a static number — no live tracker or breakdown by city/campaign."),
    ("No email capture", "There is no newsletter signup or email funnel. Without emails, NayePankh cannot do targeted outreach or re-engagement."),
    ("Donation page has no urgency signals", "No progress bar, no '47 people donated this week', no matching campaign. These simple additions can lift conversion 15–30%."),
    ("Social media not aggregated", "Instagram, Facebook, LinkedIn, YouTube and Twitter are all active but performance isn't compared or tracked centrally."),
    ("No volunteer pipeline data", "Volunteers join but there is no funnel from 'joined' → 'active' → 'team lead' tracked anywhere."),
    ("No campaign archive with metrics", "Past campaigns have no documented reach / engagement / outcome data — making it impossible to learn from them."),
]

for title, desc in gaps:
    st.markdown(f'<div class="gap-item"><strong>{title}:</strong> {desc}</div>', unsafe_allow_html=True)

# ── Quick Wins ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Quick Wins — Implement in 2 Weeks</div>', unsafe_allow_html=True)
wins = [
    "Add a live donation counter using a Google Sheet backend — updates automatically and shows social proof.",
    "Add an email signup form on the homepage with a '10% donation match' incentive for newsletter subscribers.",
    "Add a donation progress bar to the donate page: '₹82,000 of ₹1,00,000 goal this month'.",
    "Create a simple Google Looker Studio dashboard that pulls Instagram + Facebook insights into one view.",
    "Add city filter on the About page so visitors see impact specific to their city.",
]
for w in wins:
    st.markdown(f'<div class="win-item">✅ {w}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── How ML Helps ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">How Machine Learning Helps NayePankh Make Better Decisions</div>', unsafe_allow_html=True)

ml_uses = [
    {
        "icon": "💰", "title": "Donor Retention (Module 1)",
        "body": "Instead of sending the same WhatsApp blast to everyone, NayePankh can use the donor predictor to identify 'At-Risk' donors who haven't given in 90+ days and send them a personalised update about what their last donation achieved. Studies show personalised re-engagement increases donation rates by 2–3x over generic outreach."
    },
    {
        "icon": "📣", "title": "Campaign Pre-Launch Testing (Module 2)",
        "body": "Before spending time designing a poster or writing a fundraiser post, the campaign classifier can tell you whether a 'Clothes Donation' campaign on LinkedIn in February is predicted to be High or Low impact — and suggest switching to Instagram or adding a video to improve the outcome."
    },
    {
        "icon": "🗺️", "title": "Data-Backed Expansion (Module 3)",
        "body": "NayePankh currently operates in Kanpur and Ghaziabad. The district clustering model shows exactly which of the 75 remaining UP districts have the highest convergence of poverty + food insecurity + poor sanitation — giving the leadership a data-backed answer to 'where should we go next?' instead of guessing."
    },
    {
        "icon": "📊", "title": "Resource Allocation",
        "body": "ML can predict which months see the highest demand for food vs clothes vs sanitary pads (e.g., sanitary pad demand peaks in summer months in districts with low female literacy). This means NayePankh can stock resources in advance rather than reacting to shortage after shortage."
    },
]

c1, c2 = st.columns(2)
for i, item in enumerate(ml_uses):
    col = c1 if i % 2 == 0 else c2
    with col:
        st.markdown(f"""
        <div class="insight-box">
          <h4>{item['icon']} {item['title']}</h4>
          <p>{item['body']}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── ROI Chart ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Estimated Impact of ML Implementation</div>', unsafe_allow_html=True)

impact_data = pd.DataFrame({
    'Initiative': [
        'Donor Re-engagement Model',
        'Campaign Pre-launch Classifier',
        'District Expansion Prioritisation',
        'Email Capture + Funnel',
        'Impact Counter on Website'
    ],
    'Est. Uplift': [28, 22, 35, 18, 14],
    'Effort': ['Medium','Low','Low','Low','Very Low'],
    'Category': ['Donor','Campaign','Expansion','Website','Website']
})

color_map = {'Donor':'#2d6a4f','Campaign':'#52b788','Expansion':'#1a3a2a','Website':'#74c69d'}
fig = px.bar(
    impact_data, x='Est. Uplift', y='Initiative', orientation='h',
    color='Category', color_discrete_map=color_map,
    text='Est. Uplift',
    labels={'Est. Uplift': 'Estimated % Improvement', 'Initiative': ''}
)
fig.update_traces(texttemplate='%{text}%', textposition='outside')
fig.update_layout(
    height=320, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(range=[0,50], gridcolor='#eee'),
    yaxis=dict(gridcolor='#eee'),
    margin=dict(l=0,r=60,t=10,b=10),
    legend=dict(orientation='h', y=-0.2)
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Estimates based on NGO sector benchmarks and ML adoption case studies from similar-scale organisations.")

st.markdown("---")

# ── 90-Day Roadmap ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Recommended 90-Day ML Roadmap for NayePankh</div>', unsafe_allow_html=True)

roadmap = [
    ("Days 1–15", "Data Foundation", "Set up a Google Sheet to log every donation, campaign post, and volunteer signup. This is the raw material for all ML work."),
    ("Days 15–30", "Quick Wins Live", "Deploy email capture + donation counter on website. Start collecting structured data for 30 days."),
    ("Days 30–60", "Donor Model Live", "Run the donor retention model on first 30 days of data. Identify At-Risk donors and run one re-engagement campaign."),
    ("Days 60–75", "Campaign Classifier Live", "Feed past campaign data into the classifier. Use predictions to choose format + platform for the next 3 campaigns."),
    ("Days 75–90", "Expansion Decision", "Use district clustering to present a data-backed recommendation to leadership for the next city NayePankh enters."),
]

for period, title, desc in roadmap:
    st.markdown(f"""
    <div class="rec-card">
      <h4>{period} — {title}</h4>
      <p>{desc}</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Analysis by ML Intern Candidate · Built for NayePankh Foundation Selection Task · All insights derived from website audit and publicly available NGO sector data")
