# ============================================================
#  NBA PLAYER CLUSTERING — GOOGLE COLAB NOTEBOOK
#  2019-2020 Season | Play-Type Data Analysis
#  Final Year Project | Unsupervised Machine Learning
# ============================================================
#
#  HOW TO USE IN GOOGLE COLAB:
#  Split this file at each "# ── CELL N" marker into separate
#  notebook cells. Each cell is clearly delimited below.
#
# ============================================================


# ═══════════════════════════════════════════════════════════
# CELL 1 ── Install Required Libraries
# ═══════════════════════════════════════════════════════════

# Run this cell first. Restart runtime after installation if needed.

# !pip install kmodes scikit-learn pandas numpy matplotlib seaborn plotly openpyxl


# ═══════════════════════════════════════════════════════════
# CELL 2 ── Import Libraries
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

from kmodes.kmodes import KModes

import warnings
warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plot styling
sns.set_theme(style="darkgrid", palette="husl")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

print("✅ All libraries imported successfully.")
print(f"   pandas   {pd.__version__}")
print(f"   numpy    {np.__version__}")
print(f"   sklearn  {__import__('sklearn').__version__}")


# ═══════════════════════════════════════════════════════════
# CELL 3 ── Load Dataset
# ═══════════════════════════════════════════════════════════

# ── Option A: Upload directly in Colab ──
# from google.colab import files
# uploaded = files.upload()
# FILE_PATH = list(uploaded.keys())[0]

# ── Option B: Mount Google Drive ──
# from google.colab import drive
# drive.mount('/content/drive')
# FILE_PATH = "/content/drive/MyDrive/playtype2019-2020.csv"

# ── Option C: Local path (adjust as needed) ──
FILE_PATH = "playtype2019-2020__2_.csv"   # ← change if needed

df_raw = pd.read_csv('D:\FYP Final one\playtype2019-2020 (2).csv')

print("=" * 55)
print("  DATASET OVERVIEW")
print("=" * 55)
print(f"  Rows    : {df_raw.shape[0]}")
print(f"  Columns : {df_raw.shape[1]}")
print("=" * 55)
print()
print(df_raw.head())


# ═══════════════════════════════════════════════════════════
# CELL 4 ── Data Preprocessing
#           Clean column names | Handle missing/invalid values
# ═══════════════════════════════════════════════════════════

# ── Step 4.1: Rename columns to clean, consistent names ──
# Note: The raw CSV contains a duplicate 'p&rr-pts' column.
# The first occurrence is actually the rollman FREQUENCY and
# the second is the rollman PPP (points-per-possession).
# Pandas auto-renames the duplicate to 'p&rr-pts.1'.
COLUMN_MAP = {
    "Name"         : "player",
    "Team"         : "team",
    "Iso-freq"     : "iso_freq",
    "iso-pts"      : "iso_pts",
    "tra-freq"     : "tra_freq",
    "tra-pts"      : "tra_pts",
    "p&rh-freq"    : "prh_freq",
    "p&rh-pts"     : "prh_pts",
    "p&rr-pts"     : "prr_freq",   # ← mislabeled in source; actually frequency
    "p&rr-pts.1"   : "prr_pts",
    "postup-freq"  : "postup_freq",
    "postup-pts"   : "postup_pts",
    "spotup-freq"  : "spotup_freq",
    "spotup-pts"   : "spotup_pts",
    "handoff-freq" : "handoff_freq",
    "handoff-pts"  : "handoff_pts",
    "cut-freq"     : "cut_freq",
    "cut-pts"      : "cut_pts",
}

df = df_raw.rename(columns=COLUMN_MAP).copy()

# ── Step 4.2: Identify numerical feature columns ──
META_COLS  = ["player", "team"]          # not used in modelling
FREQ_COLS  = ["iso_freq", "tra_freq", "prh_freq", "prr_freq",
               "postup_freq", "spotup_freq", "handoff_freq", "cut_freq"]
PPP_COLS   = ["iso_pts", "tra_pts", "prh_pts", "prr_pts",
               "postup_pts", "spotup_pts", "handoff_pts", "cut_pts"]
NUM_COLS   = FREQ_COLS + PPP_COLS

# ── Step 4.3: Remove rows where player name is missing ──
before = len(df)
df = df.dropna(subset=["player"])
print(f"Rows removed (missing player name): {before - len(df)}")

# ── Step 4.4: Fill missing Team with 'Unknown' ──
df["team"] = df["team"].fillna("Unknown")

# ── Step 4.5: Replace invalid PPP values ──
# PPP (points per possession) must be in [0, 3].
# Frequency must be in [0, 1].
# Values outside these ranges are coerced to NaN and imputed.
for col in PPP_COLS:
    invalid_mask = (df[col] < 0) | (df[col] > 3)
    if invalid_mask.any():
        print(f"  Invalid PPP values in '{col}': {invalid_mask.sum()}")
        df.loc[invalid_mask, col] = np.nan

for col in FREQ_COLS:
    invalid_mask = (df[col] < 0) | (df[col] > 1)
    if invalid_mask.any():
        print(f"  Invalid FREQ values in '{col}': {invalid_mask.sum()}")
        df.loc[invalid_mask, col] = np.nan

# ── Step 4.6: Impute remaining NaN values with column median ──
imputer = SimpleImputer(strategy="median")
df[NUM_COLS] = imputer.fit_transform(df[NUM_COLS])

# ── Step 4.7: Reset index ──
df.reset_index(drop=True, inplace=True)

print(f"\nDataset after preprocessing: {df.shape}")
print(f"Missing values remaining   : {df[NUM_COLS].isnull().sum().sum()}")
print()
print(df.head(3))


# ═══════════════════════════════════════════════════════════
# CELL 5 ── Exploratory Data Analysis (EDA)
#           Summary statistics | Histograms | Correlation heatmap
# ═══════════════════════════════════════════════════════════

print("=" * 55)
print("  SUMMARY STATISTICS — FREQUENCY COLUMNS")
print("=" * 55)
print(df[FREQ_COLS].describe().round(3).to_string())
print()
print("=" * 55)
print("  SUMMARY STATISTICS — PPP COLUMNS")
print("=" * 55)
print(df[PPP_COLS].describe().round(3).to_string())


# ── Histogram: Frequency distributions ──
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Play-Type FREQUENCY Distributions (2019-20 NBA)", fontsize=16, fontweight="bold")

play_labels = ["Isolation", "Transition", "P&R Handler", "P&R Rollman",
               "Post-Up", "Spot-Up", "Hand-Off", "Cut"]

for ax, col, label in zip(axes.flatten(), FREQ_COLS, play_labels):
    ax.hist(df[col], bins=20, color="#3B82F6", edgecolor="white", alpha=0.85)
    ax.set_title(label)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Players")

plt.tight_layout()
plt.savefig("eda_freq_histograms.png", bbox_inches="tight")
plt.show()
print("✅ Frequency histograms saved.")


# ── Histogram: PPP distributions ──
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Play-Type PPP (Efficiency) Distributions (2019-20 NBA)", fontsize=16, fontweight="bold")

for ax, col, label in zip(axes.flatten(), PPP_COLS, play_labels):
    ax.hist(df[col], bins=20, color="#10B981", edgecolor="white", alpha=0.85)
    ax.set_title(label)
    ax.set_xlabel("Points Per Possession")
    ax.set_ylabel("Players")

plt.tight_layout()
plt.savefig("eda_ppp_histograms.png", bbox_inches="tight")
plt.show()
print("✅ PPP histograms saved.")


# ── Correlation heatmap ──
corr_matrix = df[NUM_COLS].corr()

fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Feature Correlation Matrix — NBA Play-Type Data", fontsize=16, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("✅ Correlation heatmap saved.")


# ═══════════════════════════════════════════════════════════
# CELL 6 ── Outlier Detection & Removal
#           Boxplots BEFORE → IQR removal → Boxplots AFTER
# ═══════════════════════════════════════════════════════════

def plot_boxplots(data, cols, labels, title, filename):
    """Plot boxplots for a set of columns."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for ax, col, label in zip(axes.flatten(), cols, labels):
        ax.boxplot(
            data[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="#93C5FD", color="#1D4ED8"),
            medianprops=dict(color="#DC2626", linewidth=2),
            whiskerprops=dict(color="#1D4ED8"),
            capprops=dict(color="#1D4ED8"),
            flierprops=dict(marker="o", color="#F59E0B", alpha=0.5),
        )
        ax.set_title(label)
        ax.set_ylabel("Value")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


# ── Boxplots BEFORE outlier removal ──
print("📊 Boxplots BEFORE Outlier Removal")
plot_boxplots(df, FREQ_COLS, play_labels,
              "Frequency Boxplots — BEFORE Outlier Removal",
              "boxplot_freq_before.png")
plot_boxplots(df, PPP_COLS, play_labels,
              "PPP Boxplots — BEFORE Outlier Removal",
              "boxplot_ppp_before.png")

# ── IQR-based outlier removal ──
df_clean = df.copy()

def remove_outliers_iqr(data, cols, multiplier=1.5):
    """Remove rows where any column value is beyond IQR fences."""
    mask = pd.Series([True] * len(data), index=data.index)
    outlier_report = {}
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        col_mask = (data[col] >= lower) & (data[col] <= upper)
        n_outliers = (~col_mask).sum()
        if n_outliers > 0:
            outlier_report[col] = n_outliers
        mask = mask & col_mask
    return data[mask].copy(), outlier_report

before_count = len(df_clean)
df_clean, outlier_info = remove_outliers_iqr(df_clean, NUM_COLS, multiplier=1.5)
removed = before_count - len(df_clean)

print(f"\n📋 Outlier Removal Report (IQR × 1.5):")
for col, count in outlier_info.items():
    print(f"   {col:<20} → {count:>3} potential outlier rows")
print(f"\n   Rows before : {before_count}")
print(f"   Rows after  : {len(df_clean)}")
print(f"   Rows removed: {removed}")

df_clean.reset_index(drop=True, inplace=True)

# ── Boxplots AFTER outlier removal ──
print("\n📊 Boxplots AFTER Outlier Removal")
plot_boxplots(df_clean, FREQ_COLS, play_labels,
              "Frequency Boxplots — AFTER Outlier Removal",
              "boxplot_freq_after.png")
plot_boxplots(df_clean, PPP_COLS, play_labels,
              "PPP Boxplots — AFTER Outlier Removal",
              "boxplot_ppp_after.png")


# ═══════════════════════════════════════════════════════════
# CELL 7 ── Data Augmentation
#           Method 1: Gaussian Noise
#           Method 2: k-NN Interpolation (synthetic players)
# ═══════════════════════════════════════════════════════════

print(f"Dataset size before augmentation: {len(df_clean)}")

# ── Method 1: Gaussian Noise ──
# Add small Gaussian noise (5% of each column's std) to create
# synthetic copies of existing players, increasing dataset size.

def augment_gaussian(data, cols, noise_factor=0.05, n_copies=1, seed=42):
    """
    Create synthetic samples by adding Gaussian noise.
    noise_factor: fraction of column std to use as noise std.
    """
    rng = np.random.default_rng(seed)
    augmented_frames = [data.copy()]

    stds = data[cols].std()

    for _ in range(n_copies):
        noisy = data.copy()
        noise = rng.normal(
            loc=0,
            scale=(stds * noise_factor).values,
            size=(len(data), len(cols)),
        )
        noisy[cols] = (data[cols].values + noise).clip(0)
        # Clip frequencies at 1.0
        for col in [c for c in cols if "freq" in c]:
            noisy[col] = noisy[col].clip(0, 1.0)
        # Clip PPP at 3.0
        for col in [c for c in cols if "pts" in c]:
            noisy[col] = noisy[col].clip(0, 3.0)
        # Mark as synthetic
        noisy["player"] = noisy["player"].apply(lambda x: f"{x}_aug_gauss")
        augmented_frames.append(noisy)

    return pd.concat(augmented_frames, ignore_index=True)

df_aug = augment_gaussian(df_clean, NUM_COLS, noise_factor=0.05, n_copies=1)
print(f"After Gaussian noise augmentation : {len(df_aug)} rows (+{len(df_aug)-len(df_clean)})")

# ── Method 2: k-NN Interpolation ──
# For each original player, find their 3 nearest neighbours (in feature
# space) and create a synthetic player by interpolating at a random λ.

def augment_knn_interpolation(data, cols, n_neighbors=3, n_synthetic=50, seed=42):
    """
    Create synthetic samples by linearly interpolating between
    a real player and one of their k nearest neighbours.
    """
    rng = np.random.default_rng(seed)
    X = data[cols].values

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(X)
    _, indices = nbrs.kneighbors(X)   # shape (n_players, n_neighbors+1)

    synthetic_rows = []
    for _ in range(n_synthetic):
        i   = rng.integers(0, len(data))
        j   = rng.choice(indices[i][1:])   # skip self (index 0)
        lam = rng.uniform(0.2, 0.8)        # blend weight

        synth = data.iloc[i].copy()
        synth[cols] = lam * data.iloc[i][cols].values + (1 - lam) * data.iloc[j][cols].values
        synth["player"] = f"Synth_{i}_{j}"
        synth["team"]   = "SYN"
        synthetic_rows.append(synth)

    synth_df = pd.DataFrame(synthetic_rows)
    return pd.concat([data, synth_df], ignore_index=True)

df_aug = augment_knn_interpolation(df_aug, NUM_COLS, n_neighbors=3, n_synthetic=60)
print(f"After k-NN interpolation          : {len(df_aug)} rows (+60)")
print(f"\nFinal augmented dataset size: {len(df_aug)} rows")


# ═══════════════════════════════════════════════════════════
# CELL 8 ── Feature Engineering
#           Create domain-meaningful composite features
# ═══════════════════════════════════════════════════════════

df_feat = df_aug.copy()

# ── On-Ball Creation Score ──
# How often a player is the primary ball-handler in complex actions.
df_feat["on_ball_creation"] = (
    df_feat["iso_freq"] +
    df_feat["prh_freq"] +
    df_feat["postup_freq"] +
    df_feat["handoff_freq"]
)

# ── Off-Ball Finishing Score ──
# How often a player scores without initiating the play.
df_feat["off_ball_finishing"] = (
    df_feat["cut_freq"] +
    df_feat["spotup_freq"] +
    df_feat["prr_freq"]
)

# ── Transition Threat ──
# Frequency-weighted PPP in transition — rewards both activity & efficiency.
df_feat["transition_impact"] = df_feat["tra_freq"] * df_feat["tra_pts"]

# ── Isolation Threat ──
df_feat["iso_impact"] = df_feat["iso_freq"] * df_feat["iso_pts"]

# ── Pick & Roll Handler Impact ──
df_feat["prh_impact"] = df_feat["prh_freq"] * df_feat["prh_pts"]

# ── Spot-Up Impact ──
df_feat["spotup_impact"] = df_feat["spotup_freq"] * df_feat["spotup_pts"]

# ── Overall Scoring Efficiency ──
# Weighted average PPP across all play types, weighted by frequency.
EPS = 1e-9
total_freq = (
    df_feat["iso_freq"] + df_feat["tra_freq"] + df_feat["prh_freq"] +
    df_feat["prr_freq"] + df_feat["postup_freq"] + df_feat["spotup_freq"] +
    df_feat["handoff_freq"] + df_feat["cut_freq"] + EPS
)
weighted_ppp = (
    df_feat["iso_freq"]     * df_feat["iso_pts"]     +
    df_feat["tra_freq"]     * df_feat["tra_pts"]     +
    df_feat["prh_freq"]     * df_feat["prh_pts"]     +
    df_feat["prr_freq"]     * df_feat["prr_pts"]     +
    df_feat["postup_freq"]  * df_feat["postup_pts"]  +
    df_feat["spotup_freq"]  * df_feat["spotup_pts"]  +
    df_feat["handoff_freq"] * df_feat["handoff_pts"] +
    df_feat["cut_freq"]     * df_feat["cut_pts"]
)
df_feat["overall_efficiency"] = weighted_ppp / total_freq

# ── Play-Type Diversity ──
# Count of play types where the player's frequency > 5%
THRESH = 0.05
df_feat["play_diversity"] = (df_feat[FREQ_COLS] > THRESH).sum(axis=1).astype(float)

# Collect all feature columns to be used in modelling
ENGINEERED_COLS = [
    "on_ball_creation", "off_ball_finishing",
    "transition_impact", "iso_impact", "prh_impact", "spotup_impact",
    "overall_efficiency", "play_diversity",
]

MODEL_FEATURES = NUM_COLS + ENGINEERED_COLS

print("Engineered features added:")
for f in ENGINEERED_COLS:
    print(f"  ✔  {f}")

print(f"\nTotal modelling features: {len(MODEL_FEATURES)}")
print(f"Dataset shape           : {df_feat.shape}")
print()
print(df_feat[ENGINEERED_COLS].describe().round(3))


# ═══════════════════════════════════════════════════════════
# CELL 9 ── Scaling (StandardScaler)
# ═══════════════════════════════════════════════════════════

X_raw = df_feat[MODEL_FEATURES].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=MODEL_FEATURES)

print(f"Scaled feature matrix shape: {X_scaled.shape}")
print()
print("Mean (should be ~0):")
print(np.round(X_scaled_df.mean(), 4).to_string())


# ═══════════════════════════════════════════════════════════
# CELL 10 ── Dimensionality Reduction (PCA)
#            Used for 2-D visualization and as input to DBSCAN
# ═══════════════════════════════════════════════════════════

# ── Full PCA to check explained variance ──
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_scaled)

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100,
        marker="o", linewidth=2, color="#6366F1")
ax.axhline(90, linestyle="--", color="#EF4444", label="90% threshold")
ax.axvline(n_components_90, linestyle="--", color="#10B981",
           label=f"{n_components_90} components @ 90%")
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_title("PCA — Cumulative Explained Variance", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_variance.png", bbox_inches="tight")
plt.show()
print(f"Components needed for 90% variance: {n_components_90}")

# ── 2-D PCA for visualization ──
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_2d = pca_2d.fit_transform(X_scaled)

df_feat["pca1"] = X_pca_2d[:, 0]
df_feat["pca2"] = X_pca_2d[:, 1]

var1 = pca_2d.explained_variance_ratio_[0] * 100
var2 = pca_2d.explained_variance_ratio_[1] * 100

print(f"\nPC1 explains {var1:.1f}% | PC2 explains {var2:.1f}% | Total: {var1+var2:.1f}%")

# ── PCA plot (coloured by transition frequency as a reference) ──
fig, ax = plt.subplots(figsize=(11, 7))
scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                     c=df_feat["on_ball_creation"], cmap="coolwarm",
                     alpha=0.7, edgecolors="white", linewidth=0.3, s=60)
plt.colorbar(scatter, ax=ax, label="On-Ball Creation Score")
ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=12)
ax.set_title("PCA 2-D Projection — NBA Players (coloured by On-Ball Creation)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("pca_2d_scatter.png", bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════
# CELL 11 ── KMeans Clustering
#            Elbow Method → Optimal K → Cluster assignment
# ═══════════════════════════════════════════════════════════

print("=" * 55)
print("  KMEANS CLUSTERING")
print("=" * 55)

# ── Elbow Method ──
K_RANGE = range(2, 12)
inertias   = []
sil_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    if k >= 2:
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

# ── Elbow plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_RANGE), inertias, marker="o", linewidth=2, color="#3B82F6")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-Cluster SSE)")
axes[0].set_title("Elbow Method — KMeans Inertia", fontweight="bold")
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_RANGE), sil_scores, marker="s", linewidth=2, color="#10B981")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs K", fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kmeans_elbow.png", bbox_inches="tight")
plt.show()

# ── Choose optimal K (best silhouette score) ──
OPTIMAL_K = list(K_RANGE)[np.argmax(sil_scores)]
print(f"\n✅ Optimal K selected: {OPTIMAL_K}  (highest silhouette score: {max(sil_scores):.4f})")

# ── Fit final KMeans ──
kmeans_model = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE, n_init=10)
df_feat["cluster_kmeans"] = kmeans_model.fit_predict(X_scaled)

print(f"\nKMeans Cluster Distribution:")
print(df_feat["cluster_kmeans"].value_counts().sort_index().to_string())

# ── PCA scatter coloured by KMeans ──
fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.tab10(np.linspace(0, 1, OPTIMAL_K))
for k in range(OPTIMAL_K):
    mask = df_feat["cluster_kmeans"] == k
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               label=f"Cluster {k}", alpha=0.75, edgecolors="white",
               linewidth=0.3, s=60, color=colors[k])
ax.set_xlabel(f"PC1 ({var1:.1f}%)")
ax.set_ylabel(f"PC2 ({var2:.1f}%)")
ax.set_title(f"KMeans Clusters (K={OPTIMAL_K}) on PCA 2-D Space",
             fontsize=14, fontweight="bold")
ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("kmeans_pca.png", bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════
# CELL 12 ── DBSCAN Clustering
#            k-NN distance plot to find epsilon
# ═══════════════════════════════════════════════════════════

print("=" * 55)
print("  DBSCAN CLUSTERING")
print("=" * 55)

# ── Find optimal epsilon via k-NN distance plot ──
# Use 2-D PCA as input (reduces dimensionality noise for DBSCAN)
X_dbscan = X_pca_2d.copy()

K_NN = 5
nbrs = NearestNeighbors(n_neighbors=K_NN).fit(X_dbscan)
distances, _ = nbrs.kneighbors(X_dbscan)
kth_distances = np.sort(distances[:, K_NN - 1])[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(kth_distances, linewidth=2, color="#7C3AED")
ax.axhline(y=np.percentile(kth_distances, 30), color="#EF4444",
           linestyle="--", label="Suggested ε (30th pct)")
ax.set_xlabel("Points sorted by distance")
ax.set_ylabel(f"{K_NN}-NN Distance")
ax.set_title("k-NN Distance Plot — DBSCAN Epsilon Selection",
             fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dbscan_epsilon.png", bbox_inches="tight")
plt.show()

DBSCAN_EPS = round(float(np.percentile(kth_distances, 30)), 2)
DBSCAN_MIN_SAMPLES = 5
print(f"Selected ε (epsilon) : {DBSCAN_EPS}")
print(f"Selected min_samples : {DBSCAN_MIN_SAMPLES}")

# ── Fit DBSCAN ──
dbscan_model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
raw_labels = dbscan_model.fit_predict(X_dbscan)

# Remap labels so -1 (noise) becomes its own group label
df_feat["cluster_dbscan"] = raw_labels

n_clusters_dbscan = len(set(raw_labels)) - (1 if -1 in raw_labels else 0)
n_noise           = (raw_labels == -1).sum()
print(f"\nDBSCAN found {n_clusters_dbscan} cluster(s) and {n_noise} noise point(s)")
print("\nDBSCAN Label Distribution:")
print(pd.Series(raw_labels).value_counts().sort_index().to_string())

# ── DBSCAN PCA scatter ──
fig, ax = plt.subplots(figsize=(11, 7))
unique_labels = sorted(set(raw_labels))
palette = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, palette):
    mask = df_feat["cluster_dbscan"] == label
    marker = "x" if label == -1 else "o"
    cluster_name = "Noise" if label == -1 else f"Cluster {label}"
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               label=cluster_name, alpha=0.75, edgecolors="white",
               linewidth=0.3, s=60, color=color, marker=marker)

ax.set_xlabel(f"PC1 ({var1:.1f}%)")
ax.set_ylabel(f"PC2 ({var2:.1f}%)")
ax.set_title("DBSCAN Clusters on PCA 2-D Space", fontsize=14, fontweight="bold")
ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("dbscan_pca.png", bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════
# CELL 13 ── KModes Clustering
#            Discretize features → Categorical clustering
# ═══════════════════════════════════════════════════════════

print("=" * 55)
print("  KMODES CLUSTERING")
print("=" * 55)

# KModes requires categorical data.
# We discretize every numerical feature into 3 bins: Low / Medium / High.

def discretize_features(data, cols, n_bins=3):
    """Convert numerical columns to ordinal category labels."""
    bins = ["Low", "Medium", "High"]
    disc = pd.DataFrame(index=data.index)
    for col in cols:
        disc[col] = pd.cut(
            data[col],
            bins=n_bins,
            labels=bins,
            include_lowest=True
        ).astype(str)
    return disc

X_kmodes = discretize_features(df_feat, MODEL_FEATURES, n_bins=3)

print(f"KModes input shape: {X_kmodes.shape}")
print(f"Sample of discretized data:")
print(X_kmodes.head(3))

# ── Find optimal K for KModes using cost (inertia equivalent) ──
KMODES_K_RANGE = range(2, 9)
kmodes_costs = []

for k in KMODES_K_RANGE:
    km_mode = KModes(n_clusters=k, init="Huang", n_init=5, verbose=0, random_state=RANDOM_STATE)
    km_mode.fit(X_kmodes)
    kmodes_costs.append(km_mode.cost_)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(list(KMODES_K_RANGE), kmodes_costs, marker="D", linewidth=2, color="#F59E0B")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("KModes Cost")
ax.set_title("KModes Elbow — Cost vs K", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kmodes_elbow.png", bbox_inches="tight")
plt.show()

OPTIMAL_K_MODES = OPTIMAL_K   # align with KMeans for fairer comparison
print(f"\nUsing K = {OPTIMAL_K_MODES} for KModes (same as KMeans for comparability)")

km_final = KModes(n_clusters=OPTIMAL_K_MODES, init="Huang", n_init=10, verbose=0, random_state=RANDOM_STATE)
df_feat["cluster_kmodes"] = km_final.fit_predict(X_kmodes)

print(f"\nKModes Cluster Distribution:")
print(df_feat["cluster_kmodes"].value_counts().sort_index().to_string())

# ── KModes PCA scatter ──
fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.tab10(np.linspace(0, 1, OPTIMAL_K_MODES))
for k in range(OPTIMAL_K_MODES):
    mask = df_feat["cluster_kmodes"] == k
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               label=f"Cluster {k}", alpha=0.75, edgecolors="white",
               linewidth=0.3, s=60, color=colors[k])
ax.set_xlabel(f"PC1 ({var1:.1f}%)")
ax.set_ylabel(f"PC2 ({var2:.1f}%)")
ax.set_title(f"KModes Clusters (K={OPTIMAL_K_MODES}) on PCA 2-D Space",
             fontsize=14, fontweight="bold")
ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("kmodes_pca.png", bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════
# CELL 14 ── Model Evaluation
#            Silhouette Score | Davies-Bouldin Index
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  MODEL EVALUATION METRICS")
print("=" * 60)

def evaluate_model(X, labels, model_name):
    """Compute Silhouette and Davies-Bouldin metrics."""
    # Exclude noise points (label == -1) for metric computation
    valid_mask = labels != -1
    X_valid    = X[valid_mask]
    y_valid    = labels[valid_mask]

    n_clusters = len(set(y_valid))
    if n_clusters < 2:
        return {"model": model_name, "n_clusters": n_clusters,
                "silhouette": np.nan, "davies_bouldin": np.nan}

    sil = silhouette_score(X_valid, y_valid)
    db  = davies_bouldin_score(X_valid, y_valid)
    return {
        "model": model_name,
        "n_clusters": n_clusters,
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
    }

results = []
results.append(evaluate_model(X_scaled, df_feat["cluster_kmeans"].values, "KMeans"))
results.append(evaluate_model(X_scaled, df_feat["cluster_dbscan"].values, "DBSCAN"))
results.append(evaluate_model(X_scaled, df_feat["cluster_kmodes"].values, "KModes"))

eval_df = pd.DataFrame(results)
print(eval_df.to_string(index=False))
print()
print("  Interpretation:")
print("  • Silhouette Score : Higher is better  (range −1 to +1)")
print("  • Davies-Bouldin   : Lower  is better  (range 0 to ∞ )")


# ═══════════════════════════════════════════════════════════
# CELL 15 ── Model Comparison Bar Charts
# ═══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Comparison — Clustering Quality Metrics",
             fontsize=16, fontweight="bold")

colors_bar = ["#3B82F6", "#10B981", "#F59E0B"]

# Silhouette
models = eval_df["model"].tolist()
sil_vals = eval_df["silhouette"].fillna(0).tolist()
bars = axes[0].bar(models, sil_vals, color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars, sil_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005, f"{val:.4f}",
                 ha="center", va="bottom", fontweight="bold")
axes[0].set_ylim(0, max(sil_vals) * 1.25)
axes[0].set_title("Silhouette Score (Higher = Better)", fontweight="bold")
axes[0].set_ylabel("Score")
axes[0].grid(axis="y", alpha=0.3)

# Davies-Bouldin
db_vals = eval_df["davies_bouldin"].fillna(0).tolist()
bars = axes[1].bar(models, db_vals, color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars, db_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01, f"{val:.4f}",
                 ha="center", va="bottom", fontweight="bold")
axes[1].set_ylim(0, max(db_vals) * 1.25)
axes[1].set_title("Davies-Bouldin Index (Lower = Better)", fontweight="bold")
axes[1].set_ylabel("Index Value")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches="tight")
plt.show()

best_model = eval_df.loc[eval_df["silhouette"].idxmax(), "model"]
print(f"\n🏆 Best performing model (highest silhouette): {best_model}")


# ═══════════════════════════════════════════════════════════
# CELL 16 ── Cluster Analysis
#            Feature averages per cluster | Archetype labelling
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  KMEANS CLUSTER CHARACTERISTICS (mean feature values)")
print("=" * 60)

ANALYSIS_FEATURES = FREQ_COLS + ["on_ball_creation", "off_ball_finishing",
                                   "overall_efficiency", "play_diversity"]

for cluster_col, model_name in [
    ("cluster_kmeans", "KMeans"),
    ("cluster_dbscan", "DBSCAN"),
    ("cluster_kmodes", "KModes"),
]:
    print(f"\n{'─'*60}")
    print(f"  {model_name} — Mean Feature Values per Cluster")
    print(f"{'─'*60}")
    cluster_means = (
        df_feat[df_feat[cluster_col] != -1]
        .groupby(cluster_col)[ANALYSIS_FEATURES]
        .mean()
        .round(3)
    )
    print(cluster_means.to_string())

# ── KMeans cluster archetype naming ──
print("\n" + "=" * 60)
print("  KMEANS CLUSTER ARCHETYPES (heuristic labelling)")
print("=" * 60)

km_means = df_feat.groupby("cluster_kmeans")[ANALYSIS_FEATURES].mean()

ARCHETYPE_LABELS = {}
for cluster_id, row in km_means.iterrows():
    if row["on_ball_creation"] > km_means["on_ball_creation"].median() and \
       row["iso_freq"] > km_means["iso_freq"].median():
        label = "⚡ Iso Creator / Star Guard"
    elif row["prh_freq"] == km_means["prh_freq"].max():
        label = "🎯 P&R Initiator / Point Guard"
    elif row["off_ball_finishing"] > km_means["off_ball_finishing"].median() and \
         row["cut_freq"] > km_means["cut_freq"].median():
        label = "✂️ Off-Ball Cutter / Finisher"
    elif row["spotup_freq"] == km_means["spotup_freq"].max():
        label = "🎯 Spot-Up 3PT Shooter"
    elif row["postup_freq"] > km_means["postup_freq"].median():
        label = "💪 Post-Up Big Man"
    else:
        label = "🔄 Versatile Role Player"

    ARCHETYPE_LABELS[cluster_id] = label
    n = (df_feat["cluster_kmeans"] == cluster_id).sum()
    print(f"  Cluster {cluster_id}: {label}  ({n} players)")

df_feat["archetype"] = df_feat["cluster_kmeans"].map(ARCHETYPE_LABELS)

print()
print("Example players per archetype:")
for cluster_id, label in ARCHETYPE_LABELS.items():
    sample = df_feat[df_feat["cluster_kmeans"] == cluster_id]["player"].head(5).tolist()
    sample = [p for p in sample if "_aug" not in p and "Synth" not in p][:5]
    print(f"  Cluster {cluster_id} ({label}):")
    for p in sample:
        print(f"    • {p}")


# ═══════════════════════════════════════════════════════════
# CELL 17 ── Save Final Dataset
# ═══════════════════════════════════════════════════════════

OUTPUT_COLS = (
    ["player", "team"] +
    NUM_COLS +
    ENGINEERED_COLS +
    ["pca1", "pca2"] +
    ["cluster_kmeans", "cluster_dbscan", "cluster_kmodes", "archetype"]
)

final_df = df_feat[OUTPUT_COLS].copy()

OUTPUT_FILE = "final_clustered_output.csv"
final_df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print(f"  ✅ Final dataset saved to: {OUTPUT_FILE}")
print(f"  Rows    : {final_df.shape[0]}")
print(f"  Columns : {final_df.shape[1]}")
print("=" * 60)
print()
print("Columns in output:")
for col in OUTPUT_COLS:
    print(f"  • {col}")

print()
print(final_df.head(5).to_string())

print("\n🎉 Pipeline complete! Run the Streamlit dashboard with:")
print("   streamlit run streamlit_app.py")
