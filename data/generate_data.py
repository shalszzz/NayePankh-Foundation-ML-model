"""
Generates realistic synthetic datasets for NayePankh ML project.
Based on real NGO donation patterns and India district census statistics.
Run this once before starting the app.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
DATA_DIR = Path(__file__).parent

# ─────────────────────────────────────────────
# 1. DONOR DATASET  (500 donors, RFM features)
# ─────────────────────────────────────────────
def generate_donor_data():
    n = 500
    # Recency: days since last donation (lower = better)
    recency    = np.random.exponential(scale=90, size=n).astype(int) + 1
    # Frequency: number of donations ever
    frequency  = np.random.negative_binomial(2, 0.4, size=n) + 1
    # Monetary: avg donation in ₹
    monetary   = np.random.lognormal(mean=6.5, sigma=0.9, size=n)  # ~₹665 avg
    monetary   = np.round(monetary, -1)  # round to nearest 10

    # Donor type
    donor_type = np.random.choice(
        ['Individual', 'Corporate', 'Student', 'NRI'],
        size=n, p=[0.55, 0.15, 0.25, 0.05]
    )

    # Campaign that brought them in
    campaign = np.random.choice(
        ['Food Drive', 'Education Fund', 'Sanitary Pad Drive', 'Clothes Donation', 'Social Media', 'Word of Mouth'],
        size=n, p=[0.22, 0.18, 0.20, 0.15, 0.15, 0.10]
    )

    # City
    city = np.random.choice(
        ['Kanpur', 'Ghaziabad', 'Lucknow', 'Delhi', 'Mumbai', 'Other'],
        size=n, p=[0.28, 0.20, 0.18, 0.14, 0.10, 0.10]
    )

    # Will donate again? (target) — derived from RFM with noise
    rfm_score = (1/recency * 100) * 0.4 + (frequency * 5) * 0.3 + (np.log(monetary+1)) * 0.3
    prob_retain = 1 / (1 + np.exp(-0.15 * (rfm_score - 10)))
    will_donate_again = (np.random.rand(n) < prob_retain).astype(int)

    # Predicted next donation (only meaningful if will_donate_again=1)
    next_donation = np.where(
        will_donate_again == 1,
        monetary * np.random.uniform(0.8, 1.4, size=n),
        0
    )
    next_donation = np.round(next_donation, -1)

    df = pd.DataFrame({
        'donor_id': [f'D{str(i).zfill(4)}' for i in range(1, n+1)],
        'recency_days': recency,
        'frequency': frequency,
        'avg_donation_inr': monetary.astype(int),
        'donor_type': donor_type,
        'acquisition_campaign': campaign,
        'city': city,
        'will_donate_again': will_donate_again,
        'predicted_next_donation': next_donation.astype(int)
    })
    df.to_csv(DATA_DIR / 'donors.csv', index=False)
    print(f"✅ donors.csv — {n} rows")

# ─────────────────────────────────────────────
# 2. CAMPAIGN DATASET  (120 campaigns)
# ─────────────────────────────────────────────
def generate_campaign_data():
    n = 120
    campaign_types = ['Food Drive', 'Education Fund', 'Sanitary Pad Drive',
                      'Clothes Donation', 'Awareness Post', 'Fundraiser Event']
    platforms = ['Instagram', 'Facebook', 'WhatsApp', 'LinkedIn', 'Twitter']
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    camp_type  = np.random.choice(campaign_types, size=n)
    platform   = np.random.choice(platforms, size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    month      = np.random.choice(months, size=n)

    # Engagement metrics
    reach      = np.random.lognormal(mean=8, sigma=1.2, size=n).astype(int)
    likes      = (reach * np.random.uniform(0.03, 0.18, size=n)).astype(int)
    shares     = (likes  * np.random.uniform(0.05, 0.30, size=n)).astype(int)
    comments   = (likes  * np.random.uniform(0.02, 0.12, size=n)).astype(int)
    volunteers_joined = (reach * np.random.uniform(0.001, 0.02, size=n)).astype(int)
    donations_received = (reach * np.random.uniform(0.005, 0.04, size=n) * 500).astype(int)

    # Image used?
    has_image  = np.random.choice([1, 0], size=n, p=[0.75, 0.25])
    has_video  = np.random.choice([1, 0], size=n, p=[0.30, 0.70])

    # Engagement rate
    eng_rate = (likes + shares*3 + comments*2) / np.maximum(reach, 1)

    # Impact label: High / Medium / Low
    thresholds = np.percentile(eng_rate, [33, 66])
    impact = np.where(eng_rate > thresholds[1], 'High',
              np.where(eng_rate > thresholds[0], 'Medium', 'Low'))

    df = pd.DataFrame({
        'campaign_id': [f'C{str(i).zfill(3)}' for i in range(1, n+1)],
        'campaign_type': camp_type,
        'platform': platform,
        'month': month,
        'reach': reach,
        'likes': likes,
        'shares': shares,
        'comments': comments,
        'has_image': has_image,
        'has_video': has_video,
        'volunteers_joined': volunteers_joined,
        'donations_received_inr': donations_received,
        'engagement_rate': np.round(eng_rate, 4),
        'impact_level': impact
    })
    df.to_csv(DATA_DIR / 'campaigns.csv', index=False)
    print(f"✅ campaigns.csv — {n} rows")

# ─────────────────────────────────────────────
# 3. DISTRICT DATASET  (UP districts, real names)
# ─────────────────────────────────────────────
def generate_district_data():
    districts = [
        'Agra','Aligarh','Allahabad','Ambedkar Nagar','Amethi','Amroha','Auraiya',
        'Azamgarh','Baghpat','Bahraich','Ballia','Balrampur','Banda','Barabanki',
        'Bareilly','Basti','Bijnor','Budaun','Bulandshahr','Chandauli','Chitrakoot',
        'Deoria','Etah','Etawah','Faizabad','Farrukhabad','Fatehpur','Firozabad',
        'Gautam Buddh Nagar','Ghaziabad','Ghazipur','Gonda','Gorakhpur','Hamirpur',
        'Hapur','Hardoi','Hathras','Jalaun','Jaunpur','Jhansi','Kannauj','Kanpur Dehat',
        'Kanpur Nagar','Kasganj','Kaushambi','Kheri','Kushinagar','Lalitpur','Lucknow',
        'Maharajganj','Mahoba','Mainpuri','Mathura','Mau','Meerut','Mirzapur',
        'Moradabad','Muzaffarnagar','Pilibhit','Pratapgarh','Raebareli','Rampur',
        'Saharanpur','Sambhal','Sant Kabir Nagar','Sant Ravidas Nagar','Shahjahanpur',
        'Shamli','Shravasti','Siddharthnagar','Sitapur','Sonbhadra','Sultanpur',
        'Unnao','Varanasi'
    ]
    n = len(districts)

    # Based loosely on NITI Aayog SDG data for UP
    poverty_rate     = np.random.beta(2.5, 3, size=n) * 60 + 15   # 15–75%
    literacy_rate    = np.random.beta(4, 2, size=n) * 40 + 45      # 45–85%
    sanitation_access= np.random.beta(3, 2, size=n) * 50 + 30      # 30–80%
    food_insecurity  = np.random.beta(2, 3, size=n) * 50 + 10      # 10–60%
    child_malnutrition=np.random.beta(2.5, 3, size=n) * 40 + 10   # 10–50%
    female_literacy  = literacy_rate - np.random.uniform(5, 15, size=n)
    population_lakhs = np.random.lognormal(mean=5.2, sigma=0.6, size=n)

    # Approx coordinates for UP districts (jittered around UP center)
    lat_base, lon_base = 26.8, 80.9
    lat = lat_base + np.random.uniform(-3.5, 3.5, size=n)
    lon = lon_base + np.random.uniform(-4.0, 4.0, size=n)

    df = pd.DataFrame({
        'district': districts,
        'poverty_rate_pct': np.round(poverty_rate, 1),
        'literacy_rate_pct': np.round(literacy_rate, 1),
        'female_literacy_pct': np.round(female_literacy, 1),
        'sanitation_access_pct': np.round(sanitation_access, 1),
        'food_insecurity_pct': np.round(food_insecurity, 1),
        'child_malnutrition_pct': np.round(child_malnutrition, 1),
        'population_lakhs': np.round(population_lakhs, 1),
        'lat': np.round(lat, 4),
        'lon': np.round(lon, 4),
        'nayepankh_active': np.random.choice([1, 0], size=n, p=[0.08, 0.92])
    })
    df.to_csv(DATA_DIR / 'districts.csv', index=False)
    print(f"✅ districts.csv — {n} rows")

if __name__ == '__main__':
    generate_donor_data()
    generate_campaign_data()
    generate_district_data()
    print("\n🎉 All datasets ready in /data/")
