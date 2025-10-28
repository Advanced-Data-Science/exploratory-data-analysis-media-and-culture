#!/usr/bin/env python3
"""
Alt Segmentation Visualizations — FAST
======================================

Major speedups vs your previous version:
- Reads only necessary columns (2-pass header sniff + usecols)
- Uses float32 arrays to cut memory
- Limits K search to {3,4,5} by default; toggleable
- Uses Calinski–Harabasz index (cheap) instead of full Silhouette (O(n^2))
  * Optional: sampled silhouette for sanity check
- Uses KMeans(algorithm="elkan", n_init=5, max_iter=200), or MiniBatchKMeans if desired
- Samples rows for model selection (MAX_FOR_MODEL_SELECTION) and for PCA plot (PCA_SCATTER_MAX)
- Chunked predict (optional) to assign clusters to *all* rows without blowing RAM
- Boxplots are downsampled per-group

Outputs (if applicable) to segmentation_outputs/:
  kmeans_elbow.png
  kmeans_calinski.png      (replaces heavy silhouette curve)
  kmeans_pca_scatter.png
  cluster_profile_heatmap_z.png
  cluster_profile_heatmap_orig.png
  bar_mean_weeks_on_board_by_popularity_seg.png
  box_popularity_by_tempo_seg.png
  stacked_era_vs_kmeans_norm.png
  bar_mean_popularity_by_kmeans.png
  box_weeks_on_board_by_kmeans.png
  segmentation_summary.csv
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Use non-interactive backend to avoid GUI lockups
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
# Optional (off): silhouette_score is expensive; sampling enabled below
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# --------------------------- Configuration --------------------------- #

DEFAULT_CSV = Path('/Users/jonahballard/Desktop/1 Personal/Fall 2025/Data Science Pinacle /T4/merged_billboard_spotify_matched_only.csv')  # you can pass a path via CLI
DEFAULT_OUTDIR = Path("segmentation_outputs")
RANDOM_STATE = 42

# Faster K search (can widen if needed)
K_CANDIDATES = [3, 4, 5]

# Limits / sampling
MAX_FOR_MODEL_SELECTION = 120_000     # rows for choosing K
PCA_SCATTER_MAX = 6_000               # points in PCA scatter
BOXPLOT_PER_GROUP_MAX = 6_000         # sample cap per group for boxplots
PREDICT_CHUNK_SIZE = 200_000          # rows per chunk for km.predict on full dataset

# If True, run a light sampled silhouette just for the **chosen** K
RUN_SAMPLED_SILHOUETTE_FOR_BEST_K = False
SILHOUETTE_SAMPLE = 10_000

# Choose clustering engine
USE_MINIBATCH = False                 # set True for very large datasets
MINIBATCH_BATCH_SIZE = 1024
MINIBATCH_MAX_NO_IMPROVEMENT = 10

# Candidate audio features (use what exists)
AUDIO_FEATURE_CANDIDATES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]

# Columns we may summarize if present
SUMMARY_NUM_COLS = ["popularity", "weeks-on-board", "rank", "tempo"]

# Columns needed for rule segments
SEGMENT_BASE_COLS = ["popularity", "tempo", "year"]


# ----------------------------- Utilities ----------------------------- #

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

def find_csv(start_path: Path) -> Optional[Path]:
    candidates = list(start_path.glob("**/merged_billboard_spotify_matched_only.csv"))
    return candidates[0] if candidates else None

def read_csv_fast(path: Path, desired_cols: List[str]) -> pd.DataFrame:
    """Two-pass read: sniff header -> intersect usecols -> read only needed columns."""
    header = pd.read_csv(path, nrows=0, low_memory=False)
    present = [c for c in desired_cols if c in header.columns]
    if not present:
        # fallback: read full (shouldn't happen in normal use)
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, usecols=present, low_memory=False)

def to_float32(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out

def available_columns(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    return [c for c in wanted if c in df.columns]

def terciles(series: pd.Series) -> pd.Categorical:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 3:
        return pd.Categorical([np.nan] * len(s))
    try:
        return pd.qcut(s, q=[0, 1/3, 2/3, 1], labels=["Low", "Mid", "High"])
    except Exception:
        qs = np.nanpercentile(s, [0, 33.3333, 66.6667, 100])
        for i in range(1, len(qs)):
            if qs[i] <= qs[i - 1]:
                qs[i] = qs[i - 1] + 1e-9
        return pd.cut(s, bins=qs, include_lowest=True, labels=["Low", "Mid", "High"])

def decade_labels(years: pd.Series) -> pd.Series:
    y = pd.to_numeric(years, errors="coerce")
    dec = (np.floor(y / 10) * 10).astype("Int64")
    return (dec.astype("string") + "s")

def safe_group_means(df: pd.DataFrame, y: str, grp: str) -> Optional[pd.Series]:
    if y not in df.columns or grp not in df.columns:
        return None
    g = df[[y, grp]].dropna()
    if g.empty:
        return None
    g[y] = pd.to_numeric(g[y], errors="coerce")
    return g.groupby(grp, sort=True)[y].mean()

def safe_group_boxdata(df: pd.DataFrame, y: str, grp: str, cap_per_group: int) -> Optional[Tuple[List[np.ndarray], List[str]]]:
    if y not in df.columns or grp not in df.columns:
        return None
    g = df[[y, grp]].dropna()
    if g.empty:
        return None
    g[y] = pd.to_numeric(g[y], errors="coerce")
    # order groups
    if pd.api.types.is_categorical_dtype(g[grp]):
        order = list(g[grp].cat.categories)
    else:
        order = sorted(map(str, g[grp].unique()))
    arrays, labels = [], []
    rng = np.random.default_rng(RANDOM_STATE)
    for lvl in order:
        mask = g[grp].astype(str) == str(lvl)
        vals = g.loc[mask, y].to_numpy(dtype="float32", copy=False)
        if len(vals) == 0:
            continue
        if len(vals) > cap_per_group:
            sel = rng.choice(len(vals), cap_per_group, replace=False)
            vals = vals[sel]
        arrays.append(vals)
        labels.append(str(lvl))
    if not arrays:
        return None
    return arrays, labels


# ------------------------- Core Functionality ------------------------ #

@dataclass
class ClusteringResult:
    used_features: List[str]
    scaler: Optional[StandardScaler]
    model: Optional[KMeans]
    labels_full: Optional[pd.Series]  # aligned to full df (Int64 with NaNs)
    k_pairs: List[Tuple[int, float]]  # (k, calinski_score)
    centers_z: Optional[np.ndarray]
    centers_orig: Optional[np.ndarray]
    pca_xy: Optional[np.ndarray]
    pca_labels: Optional[np.ndarray]
    sampled_silhouette: Optional[float]

def build_rule_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "popularity" in out.columns:
        out["popularity_seg"] = terciles(out["popularity"])
    if "tempo" in out.columns:
        t = pd.to_numeric(out["tempo"], errors="coerce")
        out["tempo_seg"] = pd.cut(t, bins=[-np.inf, 90, 120, np.inf], labels=["Slow", "Medium", "Fast"])
    if "year" in out.columns:
        out["era_seg"] = decade_labels(out["year"])
    return out

def run_kmeans_fast(df: pd.DataFrame, feature_pool: List[str]) -> ClusteringResult:
    feats = available_columns(df, feature_pool)
    result = ClusteringResult(
        used_features=feats, scaler=None, model=None, labels_full=None,
        k_pairs=[], centers_z=None, centers_orig=None, pca_xy=None, pca_labels=None,
        sampled_silhouette=None,
    )
    if len(feats) < 2:
        print("[warn] <2 usable audio features; skipping clustering.")
        return result

    # Numeric matrix, drop NaNs, float32
    X_raw = df[feats].apply(pd.to_numeric, errors="coerce").dropna().astype("float32")
    n = X_raw.shape[0]
    if n < 10:
        print("[warn] Not enough rows after dropping NaNs; skipping clustering.")
        return result

    # Sample for model selection
    rng = np.random.default_rng(RANDOM_STATE)
    if n > MAX_FOR_MODEL_SELECTION:
        idx_sel = rng.choice(n, MAX_FOR_MODEL_SELECTION, replace=False)
        X_sel = X_raw.to_numpy()[idx_sel]
    else:
        X_sel = X_raw.to_numpy()

    scaler = StandardScaler(copy=False)
    Xs_sel = scaler.fit_transform(X_sel)

    # Choose K using Calinski–Harabasz (fast) on the sampled set
    best_k, best_score = None, -np.inf
    k_pairs: List[Tuple[int, float]] = []
    for k in K_CANDIDATES:
        if Xs_sel.shape[0] <= k:
            continue
        if USE_MINIBATCH:
            km = MiniBatchKMeans(
                n_clusters=k, batch_size=MINIBATCH_BATCH_SIZE, random_state=RANDOM_STATE,
                n_init=3, max_no_improvement=MINIBATCH_MAX_NO_IMPROVEMENT
            )
        else:
            km = KMeans(n_clusters=k, algorithm="elkan", n_init=5, max_iter=200, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs_sel)
        score = calinski_harabasz_score(Xs_sel, labels)
        k_pairs.append((k, score))
        if score > best_score:
            best_score, best_k = score, k

    if best_k is None:
        print("[warn] No valid k found; skipping clustering.")
        return result

    # Fit final model on the *sampled* set for speed
    if USE_MINIBATCH:
        km_final = MiniBatchKMeans(
            n_clusters=best_k, batch_size=MINIBATCH_BATCH_SIZE, random_state=RANDOM_STATE,
            n_init=3, max_no_improvement=MINIBATCH_MAX_NO_IMPROVEMENT
        )
    else:
        km_final = KMeans(n_clusters=best_k, algorithm="elkan", n_init=5, max_iter=200, random_state=RANDOM_STATE)
    labels_sel = km_final.fit_predict(Xs_sel)

    # Optional: compute a sampled silhouette on sampled set just for the chosen k
    samp_sil = None
    if RUN_SAMPLED_SILHOUETTE_FOR_BEST_K and Xs_sel.shape[0] > 1000:
        samp_size = min(SILHOUETTE_SAMPLE, Xs_sel.shape[0])
        samp_sil = silhouette_score(Xs_sel, labels_sel, sample_size=samp_size, random_state=RANDOM_STATE)

    # Centers
    centers_z = km_final.cluster_centers_
    centers_orig = centers_z * scaler.scale_ + scaler.mean_

    # PCA scatter from sampled set
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    XY = pca.fit_transform(Xs_sel).astype("float32")
    if XY.shape[0] > PCA_SCATTER_MAX:
        idx = rng.choice(XY.shape[0], PCA_SCATTER_MAX, replace=False)
        XY, labels_plot = XY[idx], labels_sel[idx]
    else:
        labels_plot = labels_sel

    # Optionally predict labels for ALL rows in chunks (fast and RAM-safe)
    labels_full = pd.Series(index=df.index, data=np.nan, dtype="Float64")
    try:
        # To predict on full data we need full X scaled: reuse scaler mean/scale
        # Apply to chunks to avoid big memory spikes
        arr_full = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
        mask_valid = ~np.isnan(arr_full).any(axis=1)
        valid_idx = np.where(mask_valid)[0]
        for start in range(0, len(valid_idx), PREDICT_CHUNK_SIZE):
            sl = valid_idx[start:start+PREDICT_CHUNK_SIZE]
            block = arr_full[sl]
            block = (block - scaler.mean_) / scaler.scale_
            pred = km_final.predict(block)
            labels_full.iloc[sl] = pred
        labels_full = labels_full.astype("Int64")
    except Exception as e:
        print(f"[info] Skipping full prediction due to: {e}. Using sampled labels only.")

    result.used_features = feats
    result.scaler = scaler
    result.model = km_final
    result.labels_full = labels_full
    result.k_pairs = k_pairs
    result.centers_z = centers_z
    result.centers_orig = centers_orig
    result.pca_xy = XY
    result.pca_labels = labels_plot
    result.sampled_silhouette = samp_sil
    return result


# ----------------------------- Plotting ------------------------------ #

def plot_k_curves(outdir: Path, k_pairs: List[Tuple[int, float]], sampled_sil: Optional[float]) -> None:
    if k_pairs:
        ks, cs = zip(*k_pairs)
        fig = plt.figure()
        plt.plot(ks, cs, marker="o")
        plt.xlabel("k")
        plt.ylabel("Calinski–Harabasz score (higher is better)")
        plt.title("Cluster quality vs k (fast)")
        savefig(fig, outdir / "kmeans_calinski.png")
    if sampled_sil is not None:
        print(f"[info] Sampled silhouette (best k): {sampled_sil:.4f}")

def plot_pca_scatter(outdir: Path, xy: Optional[np.ndarray], labels: Optional[np.ndarray]) -> None:
    if xy is None or labels is None:
        return
    fig = plt.figure()
    uniq = np.unique(labels)
    for k in uniq:
        m = labels == k
        plt.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, label=f"C{k+1}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans clusters (PCA projection, sampled)")
    plt.legend(frameon=False, markerscale=2)
    savefig(fig, outdir / "kmeans_pca_scatter.png")

def plot_center_heatmaps(outdir: Path, centers_z: Optional[np.ndarray], centers_orig: Optional[np.ndarray], feat_names: List[str]) -> None:
    if centers_z is None or centers_orig is None:
        return
    def _heat(data: np.ndarray, title: str, fname: str) -> None:
        fig = plt.figure(figsize=(max(6, len(feat_names) * 0.9), max(3, data.shape[0] * 0.5 + 2)))
        im = plt.imshow(data, aspect="auto", interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.yticks(range(data.shape[0]), [f"C{i+1}" for i in range(data.shape[0])])
        plt.xticks(range(len(feat_names)), feat_names, rotation=45, ha="right")
        plt.title(title)
        savefig(fig, outdir / fname)
    _heat(centers_z, "Cluster centers (z-score space)", "cluster_profile_heatmap_z.png")
    _heat(centers_orig, "Cluster centers (original units)", "cluster_profile_heatmap_orig.png")

def comparative_plots(df: pd.DataFrame, outdir: Path) -> None:
    # Bar: mean weeks-on-board by popularity segment
    m = safe_group_means(df, "weeks-on-board", "popularity_seg")
    if m is not None and len(m):
        fig = plt.figure()
        plt.bar(m.index.astype(str), m.values)
        plt.ylabel("Mean weeks-on-board")
        plt.title("Mean weeks-on-board by popularity segment")
        savefig(fig, outdir / "bar_mean_weeks_on_board_by_popularity_seg.png")

    # Box: popularity by tempo segment (downsample per group)
    bx = safe_group_boxdata(df, "popularity", "tempo_seg", cap_per_group=BOXPLOT_PER_GROUP_MAX)
    if bx is not None:
        arrays, labels = bx
        fig = plt.figure()
        plt.boxplot(arrays, labels=labels, showfliers=False)
        plt.ylabel("popularity")
        plt.title("Popularity distribution by tempo segment (downsampled)")
        savefig(fig, outdir / "box_popularity_by_tempo_seg.png")

    # Bar: mean popularity by kmeans
    if "kmeans_lbl" in df.columns:
        m2 = safe_group_means(df, "popularity", "kmeans_lbl")
        if m2 is not None and len(m2):
            fig = plt.figure()
            plt.bar(m2.index.astype(str), m2.values)
            plt.ylabel("Mean popularity")
            plt.title("Mean popularity by KMeans cluster")
            savefig(fig, outdir / "bar_mean_popularity_by_kmeans.png")

    # Box: weeks-on-board by kmeans (downsample per group)
    if "kmeans_lbl" in df.columns:
        bx2 = safe_group_boxdata(df, "weeks-on-board", "kmeans_lbl", cap_per_group=BOXPLOT_PER_GROUP_MAX)
        if bx2 is not None:
            arrays, labels = bx2
            fig = plt.figure()
            plt.boxplot(arrays, labels=labels, showfliers=False)
            plt.ylabel("weeks-on-board")
            plt.title("Weeks-on-board distribution by KMeans cluster (downsampled)")
            savefig(fig, outdir / "box_weeks_on_board_by_kmeans.png")

    # Stacked normalized proportions: era vs kmeans
    if "era_seg" in df.columns and "kmeans_lbl" in df.columns and df["kmeans_lbl"].notna().any():
        ctab = pd.crosstab(df["era_seg"], df["kmeans_lbl"])
        if ctab.shape[0] > 0 and ctab.shape[1] > 0:
            ctab_norm = ctab.div(ctab.sum(axis=1), axis=0)
            fig = plt.figure(figsize=(8, 4 + 0.1 * len(ctab_norm)))
            x = np.arange(len(ctab_norm.index))
            bottom = np.zeros(len(ctab_norm))
            for col in ctab_norm.columns:
                vals = ctab_norm[col].values
                plt.bar(x, vals, bottom=bottom, label=str(col))
                bottom += vals
            plt.xticks(x, ctab_norm.index.astype(str), rotation=45, ha="right")
            plt.ylabel("Proportion within era")
            plt.title("Era vs KMeans cluster (normalized within era)")
            plt.legend(frameon=False, ncol=3)
            savefig(fig, outdir / "stacked_era_vs_kmeans_norm.png")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    seg_cols = [c for c in ["popularity_seg", "tempo_seg", "era_seg", "kmeans_lbl"] if c in df.columns]
    rows: List[Dict[str, object]] = []
    n_total = len(df)
    for seg in seg_cols:
        counts = df[seg].value_counts(dropna=False).sort_index()
        for level, n in counts.items():
            sub = df[df[seg] == level]
            row = {
                "segment_type": seg,
                "segment": str(level),
                "count": int(n),
                "share": float(n) / n_total if n_total else np.nan,
            }
            for col in SUMMARY_NUM_COLS:
                if col in sub.columns:
                    vals = pd.to_numeric(sub[col], errors="coerce")
                    row[f"{col}_mean"] = float(vals.mean())
                    row[f"{col}_median"] = float(vals.median())
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------- Main -------------------------------- #

def main():
    # CLI: optional args: csv_path, outdir
    csv_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_CSV
    outdir = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_OUTDIR

    if not csv_path.exists():
        fb = find_csv(Path("."))
        if fb is None:
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"[info] Using discovered CSV: {fb}")
        csv_path = fb

    # Read only what we need
    desired = list(set(SEGMENT_BASE_COLS + SUMMARY_NUM_COLS + AUDIO_FEATURE_CANDIDATES))
    df = read_csv_fast(csv_path, desired_cols=desired)
    df.columns = [c.strip() for c in df.columns]

    # Convert frequent numeric columns to float32 for speed/memory
    cast_cols = available_columns(df, SEGMENT_BASE_COLS + SUMMARY_NUM_COLS + AUDIO_FEATURE_CANDIDATES)
    df = to_float32(df, cast_cols)

    # Rule-based segments
    df = build_rule_segments(df)

    # Clustering (fast path)
    cluster = run_kmeans_fast(df, AUDIO_FEATURE_CANDIDATES)

    # Append labels for plotting (C1..Ck)
    if cluster.labels_full is not None:
        df = df.copy()
        df["kmeans"] = cluster.labels_full
        df["kmeans_lbl"] = df["kmeans"].apply(lambda x: f"C{int(x)+1}" if pd.notna(x) else np.nan).astype("string")

    # Plots for K curves, PCA, and heatmaps
    plot_k_curves(outdir, cluster.k_pairs, cluster.sampled_silhouette)
    plot_pca_scatter(outdir, cluster.pca_xy, cluster.pca_labels)
    if cluster.centers_z is not None and cluster.used_features:
        plot_center_heatmaps(outdir, cluster.centers_z, cluster.centers_orig, cluster.used_features)

    # Comparative plots (rule-based + kmeans if present)
    comparative_plots(df, outdir)

    # Summary CSV
    summary = build_summary(df)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "segmentation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")
    print("\nDone. Outputs in:", outdir.resolve())


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
