# Save as advanced_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Advanced Tweet Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with custom styling
st.markdown("""
    <h1 style='text-align: center; color: #2e4053;'>
        Advanced Tweet Analysis Dashboard 🐦
    </h1>
    """, unsafe_allow_html=True)

# Create sample data
@st.cache_data
def create_sample_data(n_samples=1000):
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_samples)
    
    texts = [
        "I love this product! Amazing!",
        "This is terrible, worst experience",
        "Great service, highly recommend",
        "Not satisfied at all",
        "Outstanding performance",
        "Disappointing results",
        "Excellent work",
        "Could be better",
        "Best purchase ever",
        "Waste of money"
    ]
    
    data = {
        'date': np.random.choice(dates, n_samples),
        'text': np.random.choice(texts, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'engagement': np.random.randint(0, 1000, n_samples),
        'location': np.random.choice(['USA', 'UK', 'India', 'Canada', 'Australia'], n_samples)
    }
    
    return pd.DataFrame(data)

# Sidebar
st.sidebar.title("Dashboard Controls")
data_option = st.sidebar.radio(
    "Choose Data Source",
    ["Sample Data", "Upload Your Data"]
)

if data_option == "Upload Your Data":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = create_sample_data()
else:
    df = create_sample_data()

# Add date filter
st.sidebar.markdown("---")
st.sidebar.title("Filters")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    # Filter data based on date range
    mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
    df = df.loc[mask]

# Location filter
if 'location' in df.columns:
    locations = ['All'] + sorted(df['location'].unique().tolist())
    selected_location = st.sidebar.selectbox("Select Location", locations)
    if selected_location != 'All':
        df = df[df['location'] == selected_location]

# Main dashboard
# Key Metrics Row
st.markdown("### Key Metrics 📊")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tweets", len(df))
with col2:
    positive_pct = (df['target'] == 1).mean() * 100
    st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
with col3:
    avg_engagement = df['engagement'].mean()
    st.metric("Avg. Engagement", f"{avg_engagement:.0f}")
with col4:
    unique_locations = len(df['location'].unique())
    st.metric("Locations", unique_locations)

# Charts
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['target'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].map({1: 'Positive', 0: 'Negative'})
    
    fig_sentiment = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title="Tweet Sentiments",
        color_discrete_sequence=['#ff9999', '#66b3ff']
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col2:
    st.subheader("Engagement by Sentiment")
    fig_engagement = px.box(
        df,
        x='target',
        y='engagement',
        color='target',
        labels={'target': 'Sentiment', 'engagement': 'Engagement'},
        title="Engagement Distribution by Sentiment",
        color_discrete_sequence=['#ff9999', '#66b3ff']
    )
    fig_engagement.update_layout(
        xaxis_title="Sentiment (0=Negative, 1=Positive)",
        yaxis_title="Engagement Count"
    )
    st.plotly_chart(fig_engagement, use_container_width=True)

# Time Series Analysis
st.markdown("---")
st.subheader("Sentiment Trends Over Time")
daily_sentiment = df.groupby([df['date'].dt.date, 'target']).size().reset_index(name='count')
fig_timeline = px.line(
    daily_sentiment,
    x='date',
    y='count',
    color='target',
    title="Daily Sentiment Trends",
    labels={'count': 'Number of Tweets', 'date': 'Date', 'target': 'Sentiment'},
    color_discrete_sequence=['#ff9999', '#66b3ff']
)
st.plotly_chart(fig_timeline, use_container_width=True)

# Geographic Distribution
st.markdown("---")
st.subheader("Geographic Distribution")
location_counts = df.groupby('location').size().reset_index(name='count')
fig_geo = px.bar(
    location_counts,
    x='location',
    y='count',
    title="Tweet Distribution by Location",
    color='count',
    color_continuous_scale='Viridis'
)
fig_geo.update_layout(
    xaxis_title="Location",
    yaxis_title="Number of Tweets"
)
st.plotly_chart(fig_geo, use_container_width=True)

# Engagement Patterns
st.markdown("---")
st.subheader("Engagement Patterns")
fig_scatter = px.scatter(
    df,
    x='date',
    y='engagement',
    color='target',
    size='engagement',
    title="Engagement Patterns Over Time",
    labels={'date': 'Date', 'engagement': 'Engagement', 'target': 'Sentiment'},
    color_discrete_sequence=['#ff9999', '#66b3ff']
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Recent Tweets Table
st.markdown("---")
st.subheader("Recent Tweets")
tweet_display = df[['text', 'target', 'engagement', 'location']].copy()
tweet_display['sentiment'] = tweet_display['target'].map({1: '😊 Positive', 0: '😞 Negative'})
tweet_display['date'] = df['date'].dt.strftime('%Y-%m-%d')

# Add engagement categories
tweet_display['engagement_level'] = pd.qcut(
    tweet_display['engagement'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# Display the enhanced table
st.dataframe(
    tweet_display[['text', 'sentiment', 'engagement', 'engagement_level', 'location', 'date']]
    .sort_values('engagement', ascending=False)
    .head(10)
    .style.background_gradient(subset=['engagement'], cmap='Blues')
)

# Summary Statistics
st.markdown("---")
st.subheader("Summary Statistics")
col1, col2 = st.columns(2)

with col1:
    st.write("Engagement Statistics")
    st.dataframe(df['engagement'].describe())

with col2:
    st.write("Sentiment by Location")
    location_sentiment = df.groupby('location')['target'].mean().round(2)
    st.dataframe(location_sentiment)

# Download section
st.sidebar.markdown("---")
st.sidebar.subheader("Download Data")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "Download CSV",
    csv,
    "tweet_data.csv",
    "text/csv",
    key='download-csv'
)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        Created with Streamlit • Last updated: {}
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)