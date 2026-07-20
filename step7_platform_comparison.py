# ===========================================================
# Step 7: Cross-Platform Comparison — Kickstarter vs Indiegogo
# Input:  data/processed/kickstarter_adjectives.csv
# Output: figures/fig6_platform_comparison.png
#         figures/fig7_category_comparison.png
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ADJ_COLS = ["rate_superlative","rate_urgency_scarcity","rate_emotional","rate_quality_technical"]
LABELS   = ["Superlative","Urgency/Scarcity","Emotional","Quality/Technical"]
COLORS   = ["#2196F3","#F44336","#4CAF50","#FF9800"]
CAT_KEYS = ["superlative","urgency_scarcity","emotional","quality_technical"]

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/kickstarter_adjectives.csv", low_memory=False)
df = df[df["launch_year"].between(2014, 2025)]

ks  = df[df["source"] == "kickstarter"]
igg = df[df["source"] == "indiegogo"]

# ── Figure 6: Adjective trends by platform ────────────────────────────────────
ks_yearly  = ks.groupby("launch_year")[ADJ_COLS].mean()
igg_yearly = igg.groupby("launch_year")[ADJ_COLS].mean()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (col, label, color) in enumerate(zip(ADJ_COLS, LABELS, COLORS)):
    ax = axes[i]
    if col in ks_yearly.columns and len(ks_yearly) > 0:
        ax.plot(ks_yearly.index, ks_yearly[col]*100,
                marker="o", linewidth=2.5, markersize=5,
                label="Kickstarter", color=color, linestyle="-")
    if col in igg_yearly.columns and len(igg_yearly) > 0:
        ax.plot(igg_yearly.index, igg_yearly[col]*100,
                marker="s", linewidth=2.5, markersize=5,
                label="Indiegogo", color=color, linestyle="--", alpha=0.7)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Adjective Category Usage: Kickstarter vs. Indiegogo (2014–2025)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig6_platform_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6_platform_comparison.png")

# ── Figure 7: Overall adj rates by platform ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: mean adj rates per category by platform
ks_means  = ks[ADJ_COLS].mean() * 100
igg_means = igg[ADJ_COLS].mean() * 100

x = np.arange(len(LABELS))
width = 0.35

bars1 = axes[0].bar(x - width/2, ks_means.values,  width, label="Kickstarter", color="#1565C0", alpha=0.85)
bars2 = axes[0].bar(x + width/2, igg_means.values, width, label="Indiegogo",   color="#E53935", alpha=0.85)

for bar in bars1:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8)

axes[0].set_xticks(x)
axes[0].set_xticklabels(LABELS, fontsize=9)
axes[0].set_ylabel("Mean Adjective Rate (%)")
axes[0].set_title("Mean Adjective Rates by Platform")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Right: success rate by adj usage per platform
results = []
for source, source_df in [("Kickstarter", ks), ("Indiegogo", igg)]:
    for col, label in zip(ADJ_COLS, LABELS):
        med = source_df[col].median()
        hi  = source_df[source_df[col] >  med]["success"].mean() * 100
        lo  = source_df[source_df[col] <= med]["success"].mean() * 100
        results.append({"Platform": source, "Category": label, "High": hi, "Low": lo, "Diff": hi-lo})

res_df = pd.DataFrame(results)

ks_diffs  = res_df[res_df["Platform"] == "Kickstarter"]["Diff"].values
igg_diffs = res_df[res_df["Platform"] == "Indiegogo"]["Diff"].values

bars1 = axes[1].bar(x - width/2, ks_diffs,  width, label="Kickstarter", color="#1565C0", alpha=0.85)
bars2 = axes[1].bar(x + width/2, igg_diffs, width, label="Indiegogo",   color="#E53935", alpha=0.85)

for bar in bars1:
    axes[1].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.3 if bar.get_height() >= 0 else -0.8),
                 f"{bar.get_height():+.1f}pp", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    axes[1].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.3 if bar.get_height() >= 0 else -0.8),
                 f"{bar.get_height():+.1f}pp", ha="center", va="bottom", fontsize=8)

axes[1].axhline(y=0, color="black", linewidth=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(LABELS, fontsize=9)
axes[1].set_ylabel("Success Rate Difference (High − Low, pp)")
axes[1].set_title("Adjective Usage Advantage by Platform (High vs. Low)")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig7_category_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig7_category_comparison.png")

# ── Print comparison summary ──────────────────────────────────────────────────
print("\n--- Platform Comparison Summary ---")
print(f"\nMean adjective rates:")
print(f"{'Category':<22} {'Kickstarter':>12} {'Indiegogo':>12} {'Difference':>12}")
for col, label in zip(ADJ_COLS, LABELS):
    ks_m  = ks[col].mean() * 100
    igg_m = igg[col].mean() * 100
    print(f"{label:<22} {ks_m:>11.3f}% {igg_m:>11.3f}% {igg_m-ks_m:>+11.3f}%")

print(f"\nSuccess rate advantage (High vs Low adjective usage):")
print(f"{'Category':<22} {'Kickstarter':>12} {'Indiegogo':>12}")
for _, row in res_df.iterrows():
    if row["Platform"] == "Kickstarter":
        igg_row = res_df[(res_df["Platform"]=="Indiegogo") & (res_df["Category"]==row["Category"])].iloc[0]
        print(f"{row['Category']:<22} {row['Diff']:>+11.1f}pp {igg_row['Diff']:>+11.1f}pp")

print("\nAll figures saved to figures/")
