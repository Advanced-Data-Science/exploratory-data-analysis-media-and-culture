#!/usr/bin/env python3
"""
SUPER-FAST Segmentation Visuals (rule-based only by default)
============================================================

What this script does (quickly):
- Builds *rule-based* segments from your CSV:
    • popularity_seg: terciles (Low/Mid/High) from 'popularity'
    • tempo_seg     : Slow (<90), Medium (90–120), Fast (>=120) from 'tempo'
    • era_seg       : decade from 'year' (e.g., "2010s")
- Saves a small set of *fast* plots and a summary CSV into segmentation_outputs/

FAST Figures saved (if columns exist):
  bar_mean_weeks_on_board_by_popularity_seg.png
  box_popularity_by_tempo_seg.png
  bar_mean_popularity_by_tempo_seg.png
  stacked_era_vs_popularity_seg_norm.png

Also writes:
  segmentation_summary.csv

OPTIONAL (off by default): Light K-Means with sampling (set RUN_KMEANS=True)
  - Samples up to SAMPLE_N rows
  - Fixed K (KMEANS_K=4) for speed (no silhouette loop)
  - Saves:
      kmeans_pca_scatter.png
      bar_mean_popularity_by_kmeans.png

USAGE:
  - Adjust DATA_PATH if needed (current default points to your absolute path).
  - Run: python segmentation_visuals_simple.py
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------- Config ---------------------------------

DATA_PATH = Path("/Users/jonahballard/Desktop/1 Personal/Fall 2025/Data Science Pinacle /merged_billboard_spotify_matched_only.csv")
OUTDIR = Path("segmentation_outputs")
MAX_POINTS_FOR_BOXPLOT = 150_000  # downsample for boxplot whiskers if needed

# Optional quick KMeans (OFF by default to guarantee fast finish)
RUN_KMEANS = False
SAMPLE_N = 25_000       # rows sampled if RUN_KMEANS=True
KMEANS_K = 4            # fixed number of clusters for speed
RANDOM_STATE = 42

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
]

# ---------------------------- Helpers -----------------------------------

def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

def _mk_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "popularity" in out.columns:
        s = pd.to_numeric(out["popularity"], errors="coerce")
        # handle ties gracefully using qcut; fallback to cut on unique quantiles
        try:
            out["popularity_seg"] = pd.qcut(s, q=[0, 1/3, 2/3, 1], labels=["Low", "Mid", "High"])
        except Exception:
            qs = s.quantile([0, 1/3, 2/3, 1]).drop_duplicates().to_list()
            if len(qs) >= 3:
                out["popularity_seg"] = pd.cut(s, bins=qs, include_lowest=True, labels=["Low", "Mid", "High"])
            else:
                out["popularity_seg"] = pd.Series(pd.NA, index=out.index, dtype="category")

    if "tempo" in out.columns:
        t = pd.to_numeric(out["tempo"], errors="coerce")
        out["tempo_seg"] = pd.cut(t, bins=[-np.inf, 90, 120, np.inf], labels=["Slow", "Medium", "Fast"])

    if "year" in out.columns:
        y = pd.to_numeric(out["year"], errors="coerce")
        decade = (np.floor(y / 10) * 10).astype("Int64")
        out["era_seg"] = (decade.astype("string") + "s").where(decade.notna())

    return out

def _summary(df: pd.DataFrame) -> pd.DataFrame:
    seg_cols = [c for c in ["popularity_seg", "tempo_seg", "era_seg"] if c in df.columns]
    rows = []
    for seg in seg_cols:
        vc = df[seg].value_counts(dropna=False).sort_index()
        for lvl, n in vc.items():
            sub = df[df[seg] == lvl]
            row = {
                "segment_type": seg,
                "segment": str(lvl),
                "count": int(n),
                "share": float(n) / len(df) if len(df) else np.nan
            }
            for metric in ["popularity", "weeks-on-board", "rank", "tempo"]:
                if metric in sub.columns:
                    m = pd.to_numeric(sub[metric], errors="coerce")
                    row[f"{metric}_mean"] = m.mean()
                    row[f"{metric}_median"] = m.median()
            rows.append(row)
    return pd.DataFrame(rows)

def _bar_mean(df: pd.DataFrame, ycol: str, group: str, title: str, fname: str):
    if (ycol not in df.columns) or (group not in df.columns):
        return
    g = df[[ycol, group]].copy()
    g[ycol] = pd.to_numeric(g[ycol], errors="coerce")
    g = g.dropna()
    if g.empty:
        return
    means = g.groupby(group)[ycol].mean().sort_index()
    fig = plt.figure()
    plt.bar(means.index.astype(str), means.values)
    plt.ylabel(f"Mean {ycol}")
    plt.title(title)
    _savefig(fig, OUTDIR / fname)

def _box_by_group(df: pd.DataFrame, ycol: str, group: str, title: str, fname: str):
    if (ycol not in df.columns) or (group not in df.columns):
        return
    g = df[[ycol, group]].copy()
    g[ycol] = pd.to_numeric(g[ycol], errors="coerce")
    g = g.dropna()
    if g.empty:
        return
    # limit size for speed
    if len(g) > MAX_POINTS_FOR_BOXPLOT:
        g = g.sample(MAX_POINTS_FOR_BOXPLOT, random_state=RANDOM_STATE)
    if pd.api.types.is_categorical_dtype(g[group]):
        order = list(g[group].cat.categories)
    else:
        order = sorted(g[group].unique(), key=lambda x: (str(x)))
    data = [g.loc[g[group] == lvl, ycol].values for lvl in order]
    fig = plt.figure()
    plt.boxplot(data, labels=[str(x) for x in order], showfliers=False)
    plt.ylabel(ycol)
    plt.title(title)
    _savefig(fig, OUTDIR / fname)

def _stacked_norm(df: pd.DataFrame, row: str, col: str, title: str, fname: str):
    if (row not in df.columns) or (col not in df.columns):
        return
    sub = df[[row, col]].dropna()
    if sub.empty:
        return
    ctab = pd.crosstab(sub[row], sub[col])
    ctab = ctab.div(ctab.sum(axis=1), axis=0)  # normalize within row
    fig = plt.figure(figsize=(8, 4 + 0.1 * len(ctab)))
    bottom = np.zeros(len(ctab))
    x = np.arange(len(ctab.index))
    for k in ctab.columns:
        vals = ctab[k].values
        plt.bar(x, vals, bottom=bottom, label=str(k))
        bottom += vals
    plt.xticks(x, [str(v) for v in ctab.index], rotation=45, ha="right")
    plt.ylabel("Proportion within " + row)
    plt.title(title)
    plt.legend(frameon=False, ncol=3)
    _savefig(fig, OUTDIR / fname)

# ----------------------------- Main -------------------------------------

def main():
    # Load
    if not DATA_PATH.exists():
        hits = list(Path(".").glob("**/merged_billboard_spotify_matched_only.csv"))
        if hits:
            df = pd.read_csv(hits[0], low_memory=False)
        else:
            raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH, low_memory=False)

    # Keep only columns we need for speed (if present)
    keep_cols = [c for c in ["popularity", "tempo", "year", "weeks-on-board", "rank"] if c in df.columns]
    if keep_cols:
        df = df[keep_cols].copy()

    # Make segments
    df = _mk_segments(df)

    # === Fast visuals ===
    _bar_mean(df, "weeks-on-board", "popularity_seg",
              "Mean weeks-on-board by popularity segment",
              "bar_mean_weeks_on_board_by_popularity_seg.png")

    _box_by_group(df, "popularity", "tempo_seg",
                  "Popularity distribution by tempo segment",
                  "box_popularity_by_tempo_seg.png")

    _bar_mean(df, "popularity", "tempo_seg",
              "Mean popularity by tempo segment",
              "bar_mean_popularity_by_tempo_seg.png")

    _stacked_norm(df, "era_seg", "popularity_seg",
                  "Era vs Popularity segment (normalized within era)",
                  "stacked_era_vs_popularity_seg_norm.png")

    # Summary CSV
    summary = _summary(df)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTDIR / "segmentation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    # === Optional very light KMeans ===
    if RUN_KMEANS:
        feats = [c for c in AUDIO_FEATURES if c in df.columns]
        if len(feats) >= 2:
            # reload full CSV for features (fast path kept only a subset)
            # we search locally first to avoid re-reading a huge file if already present
            if DATA_PATH.exists():
                full = pd.read_csv(DATA_PATH, low_memory=False)
            else:
                hit2 = list(Path(".").glob("**/merged_billboard_spotify_matched_only.csv"))
                full = pd.read_csv(hit2[0], low_memory=False) if hit2 else df
            X = full[feats].apply(pd.to_numeric, errors="coerce").dropna()
            if len(X) > SAMPLE_N:
                X = X.sample(SAMPLE_N, random_state=RANDOM_STATE)

            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.values)
            km = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(Xs)

            # PCA scatter
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            XY = pca.fit_transform(Xs)
            fig = plt.figure()
            for k in range(KMEANS_K):
                mask = labels == k
                plt.scatter(XY[mask, 0], XY[mask, 1], s=6, alpha=0.5, label=f"C{k+1}")
            plt.xlabel("PC1"); plt.ylabel("PC2")
            plt.title(f"KMeans (k={KMEANS_K}) clusters – PCA")
            plt.legend(frameon=False, markerscale=2)
            _savefig(fig, OUTDIR / "kmeans_pca_scatter.png")

            # Mean popularity by cluster (if we can align indices)
            if "popularity" in full.columns:
                pop = pd.to_numeric(full.loc[X.index, "popularity"], errors="coerce")
                means = pd.Series(labels, index=X.index).groupby(labels).apply(lambda ix: pop.iloc[ix].mean())
                fig = plt.figure()
                plt.bar([f"C{i+1}" for i in range(KMEANS_K)], means.values)
                plt.ylabel("Mean popularity")
                plt.title("Mean popularity by KMeans cluster")
                _savefig(fig, OUTDIR / "bar_mean_popularity_by_kmeans.png")
        else:
            print("[info] RUN_KMEANS=True but not enough audio feature columns present; skipping.")

    print("\nDone. Outputs in:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
