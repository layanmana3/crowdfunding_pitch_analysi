# ===========================================================
# Step 9: Sentiment Analysis with TextBlob
# Input:  data/processed/kickstarter_adjectives.csv
#         data/processed/kickstarter_unified.csv
# Output: data/processed/sentiment_results.csv
#         figures/fig10_sentiment_trends.png
#         figures/fig11_sentiment_success.png
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, subprocess, sys

# Install TextBlob if needed
try:
    from textblob import TextBlob
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "textblob"], check=True)
    from textblob import TextBlob

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/processed/kickstarter_unified.csv", low_memory=False)
df["launched_date"] = pd.to_datetime(df["launched_date"], errors="coerce")
df["launch_year"]   = df["launched_date"].dt.year

mask = df["launch_year"].isna()
df.loc[mask, "launch_year"] = pd.to_numeric(df.loc[mask, "snapshot_year"], errors="coerce")

df = df[df["launch_year"].between(2014, 2025)].copy()
df = df[df["blurb"].notna() & (df["blurb"].str.len() >= 10)].copy()
df = df.drop_duplicates(subset="id").reset_index(drop=True)
print(f"Processing {len(df):,} projects...")

# ── Compute sentiment scores ──────────────────────────────────────────────────
print("Computing sentiment scores (this takes ~5-10 min)...")

def get_sentiment(text):
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0

# Process in chunks for speed
polarities    = []
subjectivities = []

CHUNK = 5000
for i in range(0, len(df), CHUNK):
    batch = df["blurb"].iloc[i:i+CHUNK]
    for text in batch:
        pol, sub = get_sentiment(text)
        polarities.append(pol)
        subjectivities.append(sub)
    if (i // CHUNK) % 5 == 0:
        print(f"  {min(i+CHUNK, len(df)):,} / {len(df):,}")

df["polarity"]      = polarities       # -1 (negative) to +1 (positive)
df["subjectivity"]  = subjectivities   # 0 (objective) to 1 (subjective)

# Save
df[["id","source","launch_year","success","blurb_len",
    "polarity","subjectivity"]].to_csv(
    "data/processed/sentiment_results.csv", index=False)
print("Saved -> data/processed/sentiment_results.csv")

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\nOverall sentiment stats:")
print(f"  Mean polarity:     {df['polarity'].mean():.4f}  (positive)")
print(f"  Mean subjectivity: {df['subjectivity'].mean():.4f}")
print(f"\nBy source:")
print(df.groupby("source")[["polarity","subjectivity"]].mean().round(4).to_string())
print(f"\nBy success:")
print(df.groupby("success")[["polarity","subjectivity"]].mean().round(4).to_string())

# ── Figure 10: Sentiment trends over time ────────────────────────────────────
yearly = df.groupby("launch_year")[["polarity","subjectivity","success"]].agg(
    polarity      =("polarity",      "mean"),
    subjectivity  =("subjectivity",  "mean"),
    success_rate  =("success",       "mean"),
    n_projects    =("success",       "count"),
).reset_index().sort_values("launch_year")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Polarity over time
ax = axes[0]
ax.plot(yearly["launch_year"], yearly["polarity"],
        color="#E91E63", marker="o", linewidth=2.5, markersize=6)
ax.fill_between(yearly["launch_year"], yearly["polarity"], alpha=0.1, color="#E91E63")
ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Launch Year")
ax.set_ylabel("Mean Polarity Score")
ax.set_title("Pitch Sentiment Polarity Over Time\n(−1=Negative, 0=Neutral, +1=Positive)")
ax.grid(axis="y", alpha=0.3)
ax.set_xticks(yearly["launch_year"])

# Subjectivity over time
ax = axes[1]
ax.plot(yearly["launch_year"], yearly["subjectivity"],
        color="#9C27B0", marker="s", linewidth=2.5, markersize=6)
ax.fill_between(yearly["launch_year"], yearly["subjectivity"], alpha=0.1, color="#9C27B0")
ax.set_xlabel("Launch Year")
ax.set_ylabel("Mean Subjectivity Score")
ax.set_title("Pitch Subjectivity Over Time\n(0=Objective, 1=Subjective)")
ax.grid(axis="y", alpha=0.3)
ax.set_xticks(yearly["launch_year"])

plt.suptitle("Sentiment Trends in Crowdfunding Pitches (2014–2025)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig10_sentiment_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig10_sentiment_trends.png")

# ── Figure 11: Sentiment vs Success ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Polarity bins vs success
df["pol_bin"] = pd.cut(df["polarity"],
    bins=[-1, -0.1, 0, 0.1, 0.3, 1.0],
    labels=["Very Neg.", "Slightly Neg.", "Neutral", "Slightly Pos.", "Very Pos."])

pol_success = df.groupby("pol_bin", observed=True)["success"].mean() * 100
pol_success = pol_success.dropna()

colors_pol = ["#F44336","#FF9800","#9E9E9E","#8BC34A","#4CAF50"][:len(pol_success)]
bars = axes[0].bar(pol_success.index, pol_success.values,
                   color=colors_pol, width=0.6)
for bar, val in zip(bars, pol_success.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_ylabel("Success Rate (%)")
axes[0].set_title("Success Rate by Pitch Polarity")
axes[0].set_ylim(0, 90)
axes[0].grid(axis="y", alpha=0.3)
axes[0].tick_params(axis="x", rotation=15)

# Subjectivity bins vs success
df["sub_bin"] = pd.cut(df["subjectivity"],
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=["Low\n(Objective)", "Medium-Low", "Medium-High", "High\n(Subjective)"])

sub_success = df.groupby("sub_bin", observed=True)["success"].mean() * 100
sub_success = sub_success.dropna()

colors_sub = ["#90CAF9","#42A5F5","#1E88E5","#1565C0"][:len(sub_success)]
bars = axes[1].bar(sub_success.index, sub_success.values,
                   color=colors_sub, width=0.6)
for bar, val in zip(bars, sub_success.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_ylabel("Success Rate (%)")
axes[1].set_title("Success Rate by Pitch Subjectivity")
axes[1].set_ylim(0, 90)
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Campaign Success Rate by Pitch Sentiment", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig11_sentiment_success.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig11_sentiment_success.png")

# ── Platform comparison ───────────────────────────────────────────────────────
print("\nSentiment by platform and success:")
print(df.groupby(["source","success"])[["polarity","subjectivity"]].mean().round(4).to_string())

print("\nAll figures saved to figures/")
