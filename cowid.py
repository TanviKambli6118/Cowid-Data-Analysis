import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="COVID-19 Data Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme with custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    h1, h2, h3, h4 {
        color: #00FFAA;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Tanvi\Downloads\owid-covid-data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("⚙️ Dashboard Controls")
countries = ["India", "United States", "Brazil", "United Kingdom"]
selected_country = st.sidebar.selectbox("Select Country", countries)
date_range = st.sidebar.date_input("Select Date Range", 
                                  [df['date'].min(), df['date'].max()])

country_data = df[df['location'] == selected_country]
if len(date_range) == 2:
    country_data = country_data[(country_data['date'] >= pd.to_datetime(date_range[0])) & 
                                (country_data['date'] <= pd.to_datetime(date_range[1]))]

# ---------------------------
# Tabs for sections
# ---------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🌍 Global Map", "💉 Vaccinations", "🔮 Predictions", "📈 Correlations"])

# ---------------------------
# Tab 1 - Overview
# ---------------------------
with tab1:
    st.title(f"🦠 COVID-19 Overview - {selected_country}")

    total_cases = int(country_data['total_cases'].max())
    total_deaths = int(country_data['total_deaths'].max())
    total_vaccinations = int(country_data['total_vaccinations'].max()) if not country_data['total_vaccinations'].isna().all() else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", f"{total_cases:,}")
    col2.metric("Total Deaths", f"{total_deaths:,}")
    col3.metric("Total Vaccinations", f"{total_vaccinations:,}")

    # Daily Cases and Deaths (Line Chart)
    st.subheader(f"📈 Daily Cases & Deaths in {selected_country}")
    fig = px.line(country_data, x="date", y=["new_cases", "new_deaths"],
                  labels={"value": "Count", "variable": "Metric"},
                  title=f"Daily Cases & Deaths - {selected_country}")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Area Chart
    st.subheader("📊 Cases vs Deaths Over Time (Area Chart)")
    fig_area = px.area(country_data, x="date", y=["new_cases", "new_deaths"],
                       labels={"value": "Count", "variable": "Metric"},
                       title="Daily Cases & Deaths (Area Chart)")
    fig_area.update_layout(template="plotly_dark")
    st.plotly_chart(fig_area, use_container_width=True)

# ---------------------------
# Tab 2 - Global Map
# ---------------------------
with tab2:
    st.title("🌍 Global COVID-19 Cases Map")
    world_cases = df.groupby("location")["total_cases"].max().reset_index()
    fig_map = px.choropleth(world_cases,
                            locations="location",
                            locationmode="country names",
                            color="total_cases",
                            hover_name="location",
                            title="Global COVID-19 Cases",
                            color_continuous_scale="Reds")
    fig_map.update_layout(template="plotly_dark")
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("🎥 Animated Global Cases Timeline")
    timeline = df.groupby(["date", "location"])["total_cases"].max().reset_index()
    fig_anim = px.choropleth(timeline,
                             locations="location",
                             locationmode="country names",
                             color="total_cases",
                             animation_frame="date",
                             title="Global Spread of COVID-19 Over Time",
                             color_continuous_scale="Viridis")
    fig_anim.update_layout(template="plotly_dark")
    st.plotly_chart(fig_anim, use_container_width=True)

# ---------------------------
# Tab 3 - Vaccinations
# ---------------------------
with tab3:
    st.title("💉 Vaccination Analysis")
    vacc_data = df[df['location'].isin(countries)].groupby("location")["total_vaccinations"].max().reset_index()
    fig2 = px.bar(vacc_data, x="location", y="total_vaccinations",
                  color="location", title="Total Vaccinations per Country")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"💉 Vaccination Progress - {selected_country}")
    fig_vacc = px.bar(country_data, x="date", y="new_vaccinations",
                      title=f"Daily Vaccinations in {selected_country}")
    fig_vacc.update_layout(template="plotly_dark")
    st.plotly_chart(fig_vacc, use_container_width=True)

# ---------------------------
# Tab 4 - Predictions
# ---------------------------
with tab4:
    st.title("🔮 Simple Predictions (Next 7 Days)")
    country_data['new_cases_MA7'] = country_data['new_cases'].rolling(7).mean()
    future = country_data[['date', 'new_cases_MA7']].dropna().tail(14)
    st.line_chart(future.set_index('date'))

# ---------------------------
# Tab 5 - Correlations
# ---------------------------
with tab5:
    st.title(f"📈 Feature Correlations - {selected_country}")
    corr_features = ["total_cases", "total_deaths", "total_vaccinations"]
    corr = country_data[corr_features].corr()

    fig_corr = px.imshow(corr,
                         text_auto=True,
                         title="Correlation Heatmap",
                         color_continuous_scale="RdBu_r")
    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("🔵 Cases vs Deaths Bubble Chart")
    bubble_data = df[df['location'].isin(countries)].groupby("location").agg(
        {"total_cases":"max", "total_deaths":"max", "population":"max"}).reset_index()

    fig_bubble = px.scatter(bubble_data,
                            x="total_cases", y="total_deaths",
                            size="population", color="location",
                            hover_name="location",
                            size_max=60,
                            title="Cases vs Deaths (Bubble Size = Population)")
    fig_bubble.update_layout(template="plotly_dark")
    st.plotly_chart(fig_bubble, use_container_width=True)