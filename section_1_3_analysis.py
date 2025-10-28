#!/usr/bin/env python3
# Section 1.3 — Bivariate Analysis
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
try:
    from scipy.stats import pearsonr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# --- Load & prep ---
path = Path("merged_billboard_spotify_matched_only.csv")  # <-- put your local path here if different
df = pd.read_csv(path, low_memory=False)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception:
        pass

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

num_df = df[numeric_cols].dropna(how="any")
corr = num_df.corr(method="pearson")

# p-values
pval = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
n = len(num_df)
for i, a in enumerate(corr.columns):
    for j, b in enumerate(corr.columns):
        if i <= j:
            r = corr.loc[a, b]
            if SCIPY_OK:
                r_sc, p_sc = pearsonr(num_df[a], num_df[b])
                pval.loc[a, b] = p_sc
                pval.loc[b, a] = p_sc
            else:
                if abs(r) < 1:
                    t = r * np.sqrt((n - 2) / (1 - r**2))
                    from math import erf, sqrt
                    def normal_cdf(x): return 0.5*(1+erf(x/np.sqrt(2)))
                    p = 2*(1 - normal_cdf(abs(t)))
                else:
                    p = 0.0
                pval.loc[a, b] = p
                pval.loc[b, a] = p

# --- Heatmap (matplotlib) ---
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr.values, aspect='auto')
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=60, ha='right', fontsize=8)
ax.set_yticklabels(corr.index, fontsize=8)
ax.set_title("Correlation Matrix (Pearson)", pad=12)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_ylabel('r', rotation=270, labelpad=12)
plt.tight_layout(); plt.show()

# --- Significant correlations ---
sig_mask = (pval < 0.001) & (corr.abs() >= 0.20)
sig_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        a, b = corr.columns[i], corr.columns[j]
        if sig_mask.loc[a, b]:
            sig_pairs.append((a, b, corr.loc[a, b], pval.loc[a, b]))
sig_df = pd.DataFrame(sig_pairs, columns=["var1", "var2", "r", "p_value"]).sort_values("r", ascending=False)
print("\\nTop significant correlations (|r|≥0.20, p<0.001):")
print(sig_df.head(20).to_string(index=False))

# --- Scatter plots (sampled) ---
np.random.seed(7)
sample_n = min(15000, len(df))
df_sample = df.sample(sample_n) if sample_n < len(df) else df.copy()

def scatter_with_trend(xcol, ycol, data, title):
    data = data[[xcol, ycol]].dropna()
    x, y = data[xcol].values, data[ycol].values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(x, y, s=6, alpha=0.4)
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, m*xx + b, linewidth=2)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(title)
    plt.tight_layout(); plt.show()

pairs = [
    ("popularity", "rank", "Popularity vs Chart Rank (lower rank is better)"),
    ("energy", "loudness", "Energy vs Loudness"),
    ("valence", "danceability", "Valence vs Danceability"),
    ("tempo", "duration_ms", "Tempo vs Duration (ms)")
]
for xcol, ycol, title in pairs:
    if xcol in df_sample.columns and ycol in df_sample.columns:
        scatter_with_trend(xcol, ycol, df_sample, title)

# --- Categorical vs Numeric: Genre vs Popularity (boxplot) ---
if "genre" in df.columns and "popularity" in df.columns:
    top_genres = df["genre"].value_counts().head(8).index.tolist()
    box_df = df.loc[df["genre"].isin(top_genres), ["genre", "popularity"]].dropna()
    order = box_df["genre"].value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(10,5))
    data_per_group = [box_df.loc[box_df["genre"] == g, "popularity"].values for g in order]
    ax.boxplot(data_per_group, labels=order, showfliers=False)
    ax.set_title("Popularity by Top Genres (Boxplot)")
    ax.set_xlabel("Genre"); ax.set_ylabel("Popularity")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout(); plt.show()

