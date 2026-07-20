# ===========================================================
# Step 8: Analysis by Project Category + Word Cloud
# Input:  data/processed/kickstarter_adjectives.csv
# Output: figures/fig8_category_adj.png
#         figures/fig9_wordcloud_before.png
#         figures/fig9_wordcloud_after.png
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, subprocess, sys

# Install wordcloud if needed
try:
    from wordcloud import WordCloud
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "wordcloud"], check=True)
    from wordcloud import WordCloud

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

ADJ_COLS = ["rate_superlative","rate_urgency_scarcity","rate_emotional","rate_quality_technical"]
LABELS   = ["Superlative","Urgency/Scarcity","Emotional","Quality/Technical"]
COLORS   = ["#2196F3","#F44336","#4CAF50","#FF9800"]

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/kickstarter_adjectives.csv", low_memory=False)
df = df[df["launch_year"].between(2014, 2025)]

# ── Figure 8: Adjective rates by top project categories ──────────────────────
top_cats = df["parent_category"].value_counts().nlargest(8).index
df_cats  = df[df["parent_category"].isin(top_cats)]

cat_means = df_cats.groupby("parent_category")[ADJ_COLS].mean() * 100
cat_means = cat_means.loc[cat_means.mean(axis=1).sort_values(ascending=False).index]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (col, label, color) in enumerate(zip(ADJ_COLS, LABELS, COLORS)):
    ax = axes[i]
    vals = cat_means[col].sort_values(ascending=True)
    bars = ax.barh(vals.index, vals.values, color=color, alpha=0.85, height=0.6)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=8)
    ax.set_title(label, fontsize=11, color=color)
    ax.set_xlabel("Mean Adjective Rate (%)")
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Adjective Category Usage by Project Category\n(Top 8 Parent Categories)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig8_category_adj.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig8_category_adj.png")

# Print summary
print("\nAdjective rates by category (%):")
print(cat_means.round(3).to_string())

# ── Word Clouds: before/after changepoints ────────────────────────────────────
# Main changepoint: Superlative CP=2021, Emotional CP=2016
# Use 2016 as the main divide (first CP)
MAIN_CP = 2019  # Use 2019 as natural midpoint for word cloud

df_ks = pd.read_csv("data/processed/kickstarter_unified.csv", low_memory=False)
df_ks["launched_date"] = pd.to_datetime(df_ks["launched_date"], errors="coerce")
df_ks["launch_year"]   = df_ks["launched_date"].dt.year

mask = df_ks["launch_year"].isna()
df_ks.loc[mask, "launch_year"] = pd.to_numeric(df_ks.loc[mask, "snapshot_year"], errors="coerce")
df_ks = df_ks[df_ks["launch_year"].between(2014, 2025) & df_ks["blurb"].notna()]

import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Categorized adjective word sets
SUPERLATIVES     = {'best','greatest','largest','fastest','highest','biggest','first','ultimate','top','finest','premier','leading','unmatched','unrivaled','supreme','perfect','ideal','only','sole'}
URGENCY_SCARCITY = {'limited','exclusive','rare','scarce','urgent','immediate','critical','essential','vital','crucial','last','final','early','special','unique','temporary','few','remaining'}
EMOTIONAL        = {'amazing','incredible','awesome','fantastic','wonderful','beautiful','stunning','breathtaking','inspiring','passionate','exciting','thrilling','magical','extraordinary','remarkable','unforgettable','powerful','heartfelt','moving','joyful','fun','hilarious','adorable','lovely','charming','delightful','epic','bold','brave','daring','creative','innovative','revolutionary','groundbreaking'}
QUALITY_TECHNICAL= {'professional','premium','quality','durable','reliable','robust','advanced','smart','efficient','effective','precise','accurate','secure','safe','certified','tested','proven','engineered','designed','crafted','handmade','custom','sustainable','organic','natural','authentic','original','genuine','lightweight','compact','portable','wireless','digital','modular','waterproof'}
ALL_ADJ = SUPERLATIVES | URGENCY_SCARCITY | EMOTIONAL | QUALITY_TECHNICAL

COLORS_WC = {
    **{w: '#2196F3' for w in SUPERLATIVES},
    **{w: '#F44336' for w in URGENCY_SCARCITY},
    **{w: '#4CAF50' for w in EMOTIONAL},
    **{w: '#FF9800' for w in QUALITY_TECHNICAL},
}

def color_func(word, **kwargs):
    return COLORS_WC.get(word.lower(), '#888888')

def get_adj_freq(texts, max_texts=5000):
    freq = {}
    sample = texts.sample(min(max_texts, len(texts)), random_state=42)
    for doc in nlp.pipe(sample.tolist(), batch_size=256):
        for t in doc:
            if t.pos_ == "ADJ" and t.is_alpha and len(t.text) > 2:
                w = t.lemma_.lower()
                if w in ALL_ADJ:
                    freq[w] = freq.get(w, 0) + 1
    return freq

print("\nBuilding word clouds...")
before_texts = df_ks[df_ks["launch_year"] < MAIN_CP]["blurb"]
after_texts  = df_ks[df_ks["launch_year"] >= MAIN_CP]["blurb"]

freq_before = get_adj_freq(before_texts)
freq_after  = get_adj_freq(after_texts)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, freq, title in zip(axes,
    [freq_before, freq_after],
    [f"Before {MAIN_CP} (2014–{MAIN_CP-1})", f"After {MAIN_CP} ({MAIN_CP}–2025)"]):

    if freq:
        wc = WordCloud(
            width=700, height=400,
            background_color="white",
            max_words=60,
            color_func=color_func,
            prefer_horizontal=0.8,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

# Legend
from matplotlib.patches import Patch
legend = [
    Patch(color='#2196F3', label='Superlative'),
    Patch(color='#F44336', label='Urgency/Scarcity'),
    Patch(color='#4CAF50', label='Emotional'),
    Patch(color='#FF9800', label='Quality/Technical'),
]
fig.legend(handles=legend, loc="lower center", ncol=4,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Most Frequent Marketing Adjectives in Crowdfunding Pitches\n(colored by category)", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig9_wordclouds.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig9_wordclouds.png")

print("\nTop adjectives before/after changepoint:")
print(f"\nBefore {MAIN_CP}: {sorted(freq_before.items(), key=lambda x: -x[1])[:10]}")
print(f"After {MAIN_CP}:  {sorted(freq_after.items(), key=lambda x: -x[1])[:10]}")
print("\nAll figures saved to figures/")
