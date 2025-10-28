# 📦 Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🧠 Load dataset
df = pd.read_csv("merged_billboard_spotify.csv")

# 🎵 Filter to unique songs using track_id (avoid duplicates)
df_unique = df.drop_duplicates(subset=['track_id']).copy()

# 🧹 Drop rows missing key Spotify features
df_unique = df_unique.dropna(subset=['danceability', 'energy', 'valence', 'popularity', 'genre'])

# 🎨 Set visual style
sns.set(style="whitegrid", palette="pastel")
plt.rcParams["figure.figsize"] = (8, 5)

# ==============================================================
# 3.1 Visualization Creation
# ==============================================================

# 1HISTOGRAMS for 3 numerical variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df_unique['danceability'], bins=30, kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Distribution of Danceability")
axes[0].set_xlabel("Danceability")
axes[0].set_ylabel("Count")

sns.histplot(df_unique['energy'], bins=30, kde=True, ax=axes[1], color="salmon")
axes[1].set_title("Distribution of Energy")
axes[1].set_xlabel("Energy")
axes[1].set_ylabel("Count")

sns.histplot(df_unique['valence'], bins=30, kde=True, ax=axes[2], color="lightgreen")
axes[2].set_title("Distribution of Valence (Positivity)")
axes[2].set_xlabel("Valence")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.show()


# BOX PLOTS for 2 numerical variables
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Top 10 genres for clarity
top_genres = df_unique['genre'].value_counts().head(10).index

sns.boxplot(data=df_unique[df_unique['genre'].isin(top_genres)],
            x='genre', y='popularity', ax=axes[0])
axes[0].set_title("Popularity Distribution by Top 10 Genres")
axes[0].set_xlabel("Genre")
axes[0].set_ylabel("Popularity")
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(data=df_unique, x='mode', y='loudness', ax=axes[1])
axes[1].set_title("Loudness by Mode (0 = Minor, 1 = Major)")
axes[1].set_xlabel("Mode")
axes[1].set_ylabel("Loudness (dB)")

plt.tight_layout()
plt.show()


# BAR CHARTS for 2 categorical variables
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart of Top 10 genres
df_unique['genre'].value_counts().head(10).plot(
    kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title("Top 10 Genres by Song Count")
axes[0].set_xlabel("Genre")
axes[0].set_ylabel("Number of Songs")

# Bar chart of Top 10 artists
top_artists = df_unique['artist_name'].value_counts().head(10)
sns.barplot(x=top_artists.values, y=top_artists.index, ax=axes[1], palette="coolwarm")
axes[1].set_title("Top 10 Artists by Number of Songs")
axes[1].set_xlabel("Number of Songs")
axes[1].set_ylabel("Artist")

plt.tight_layout()
plt.show()

# ==============================================================
# 3.2 Advanced Visualizations
# ==============================================================

import plotly.express as px

# 1️⃣ CORRELATION HEATMAP (Spotify audio features)
plt.figure(figsize=(12, 8))
features = [
    'danceability', 'energy', 'speechiness', 'acousticness', 
    'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity'
]

corr = df_unique[features].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Spotify Audio Features", fontsize=14)
plt.show()


# 2️⃣ PAIR PLOT — Relationships among key musical attributes
pairplot_data = df_unique[['danceability', 'energy', 'valence', 'popularity']]
sns.pairplot(pairplot_data, diag_kind="kde", corner=True)
plt.suptitle("Pairwise Relationships: Danceability, Energy, Valence, and Popularity", y=1.02)
plt.show()


# 3️⃣ DASHBOARD-STYLE VISUALIZATION (interactive bubble chart)
# Shows popularity vs. energy, bubble size = danceability, color = valence

fig = px.scatter(
    df_unique.sample(1000, random_state=42),  # sample for speed
    x="energy",
    y="popularity",
    size="danceability",
    color="valence",
    hover_name="track_name",
    hover_data=["artist_name", "genre"],
    title="Interactive Dashboard: Popularity vs Energy (Size = Danceability, Color = Valence)",
    color_continuous_scale="RdYlGn",
    size_max=30
)
fig.update_layout(
    xaxis_title="Energy",
    yaxis_title="Popularity",
    template="plotly_white"
)
fig.show()
