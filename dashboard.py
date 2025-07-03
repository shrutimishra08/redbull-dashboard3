import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

st.title("RedBull AI Talent Matching Dashboard")

# Load dataset
url = "https://raw.githubusercontent.com/shrutimishra08/redbull-dashboard2/refs/heads/main/employee_data.csv"  # Replace with your URL
data = pd.read_csv(url)

# Define departments and relevant metrics
departments = ['Sales', 'Retail', 'Product', 'HR', 'Legal']
metrics = {
    'Sales': ['Diplomatic', 'Balanced', 'Sociable', 'Innovative', 'SalesRevenue', 'MarketShare'],
    'Retail': ['Diplomatic', 'Balanced', 'Sociable', 'Innovative', 'SalesRevenue', 'MarketShare'],
    'Product': ['Diplomatic', 'Balanced', 'Sociable', 'Innovative', 'EBIT', 'ProjectDeliveryRate'],
    'HR': ['Diplomatic', 'Balanced', 'Sociable', 'Innovative', 'RetentionRate'],
    'Legal': ['Diplomatic', 'Balanced', 'Sociable', 'Innovative', 'ComplianceRate']
}

# Create personas
personas = {}
for dept in departments:
    dept_data = data[data['Department'] == dept][metrics[dept]]
    if len(dept_data) > 0:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dept_data)
        kmeans = KMeans(n_clusters=1, random_state=42)
        kmeans.fit(scaled_data)
        persona = scaler.inverse_transform(kmeans.cluster_centers_)[0]
        personas[dept] = dict(zip(metrics[dept], persona))

# Display personas
st.header("Department Personas")
persona_df = pd.DataFrame(personas).T
st.dataframe(persona_df.round(2))  # Round to 2 decimal places

# Candidate input
st.header("Candidate WingFinder Input")
diplomatic = st.slider("Diplomatic", 0, 100, 85)
balanced = st.slider("Balanced", 0, 100, 90)
sociable = st.slider("Sociable", 0, 100, 75)
innovative = st.slider("Innovative", 0, 100, 80)

# Match candidate
def match_candidate(diplomatic, balanced, sociable, innovative):
    candidate = np.array([diplomatic, balanced, sociable, innovative])
    distances = {}
    for dept in departments:
        if dept in personas:
            persona_scores = np.array([personas[dept][m] for m in ['Diplomatic', 'Balanced', 'Sociable', 'Innovative']])
            distance = np.sqrt(np.sum((candidate - persona_scores) ** 2))
            distances[dept] = distance
        else:
            distances[dept] = float('inf')
    best_match = min(distances, key=distances.get)
    return best_match, distances

if st.button("Match Candidate"):
    best_dept, distances = match_candidate(diplomatic, balanced, sociable, innovative)
    st.success(f"Recommended Department: {best_dept}")
    st.subheader("Match Scores (Lower is Better)")
    dist_df = pd.DataFrame(list(distances.items()), columns=['Department', 'Distance'])
    st.dataframe(dist_df.round(2))
    fig = px.bar(dist_df, x='Department', y='Distance', title='Candidate Match Scores')
    st.plotly_chart(fig)
