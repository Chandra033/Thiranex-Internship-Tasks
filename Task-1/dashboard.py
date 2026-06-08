import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Netflix Data Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Netflix Data Cleaning & Visualization Dashboard")

# Load Dataset
df = pd.read_csv("netflix_titles.csv")

# --------------------------
# Data Cleaning
# --------------------------

df.drop_duplicates(inplace=True)

df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Not Available")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna(df["rating"].mode()[0])

df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)

df["year_added"] = df["date_added"].dt.year

# --------------------------
# Dashboard Metrics
# --------------------------

total_titles = len(df)
movies = len(df[df["type"] == "Movie"])
tvshows = len(df[df["type"] == "TV Show"])

col1, col2, col3 = st.columns(3)

col1.metric("Total Titles", total_titles)
col2.metric("Movies", movies)
col3.metric("TV Shows", tvshows)

st.divider()

# --------------------------
# Dataset Preview
# --------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------
# Movies vs TV Shows
# --------------------------

st.subheader("Movies vs TV Shows")

fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="type", ax=ax1)
st.pyplot(fig1)

# --------------------------
# Top 10 Countries
# --------------------------

st.subheader("Top 10 Content Producing Countries")

top_countries = df["country"].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    ax=ax2
)

ax2.set_xlabel("Number of Titles")
ax2.set_ylabel("Country")

st.pyplot(fig2)

# --------------------------
# Ratings Distribution
# --------------------------

st.subheader("Content Rating Distribution")

fig3, ax3 = plt.subplots(figsize=(8, 5))

sns.countplot(
    y="rating",
    data=df,
    order=df["rating"].value_counts().index,
    ax=ax3
)

st.pyplot(fig3)

# --------------------------
# Content Growth Trend
# --------------------------

st.subheader("Netflix Content Growth")

yearly = (
    df["year_added"]
    .value_counts()
    .sort_index()
)

fig4, ax4 = plt.subplots(figsize=(8, 5))

ax4.plot(
    yearly.index,
    yearly.values,
    marker="o"
)

ax4.set_xlabel("Year")
ax4.set_ylabel("Titles Added")
ax4.grid(True)

st.pyplot(fig4)

# --------------------------
# Outlier Detection
# --------------------------

st.subheader("Release Year Outliers")

fig5, ax5 = plt.subplots(figsize=(8, 3))

sns.boxplot(
    x=df["release_year"],
    ax=ax5
)

st.pyplot(fig5)

# --------------------------
# Key Findings
# --------------------------

st.subheader("Key Findings")

st.success("""
• Movies dominate the Netflix catalog.

• United States contributes the highest content.

• TV-MA and TV-14 are the most common ratings.

• Netflix content additions increased significantly after 2015.

• Missing values and duplicates were successfully handled.
""")
