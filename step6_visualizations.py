# ===========================================================
# Step 6: Generate all figures for the paper
# Output: figures/fig1_trends.png
#         figures/fig2_changepoints.png
#         figures/fig3_success_by_adj.png
#         figures/fig4_coefficients.png
#         figures/fig5_auc.png
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os

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

COLORS = {
    "superlative":       "#2196F3",
    "urgency_scarcity":  "#F44336",
    "emotional":         "#4CAF50",
    "quality_technical": "#FF9800",
}
LABELS = {
    "rate_superlative":       "Superlative",
    "rate_urgency_scarcity":  "Urgency/Scarcity",
    "rate_emotional":         "Emotional",
    "rate_quality_technical": "Quality/Technical",
}
ADJ_COLS = list(LABELS.keys())
CAT_KEYS = ["superlative", "urgency_scarcity", "emotional", "quality_technical"]

# ── Load ─────────────────────────────────────────────────────────────────────
yearly  = pd.read_csv("data/processed/yearly_adj_rates.csv")
df      = pd.read_csv("data/processed/kickstarter_adjectives.csv", low_memory=False)
df      = df[df["launch_year"].between(2014, 2025)]

with open("data/processed/changepoint_results.json") as f:
    cp_results = json.load(f)
with open("data/processed/regression_results.json") as f:
    reg_results = json.load(f)
coef_df = pd.read_csv("data/processed/regression_coefficients.csv")

# ── Figure 1: Trends ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for col, label in LABELS.items():
    cat = col.replace("rate_", "")
    ax.plot(yearly["launch_year"], yearly[col]*100,
            marker="o", linewidth=2.5, markersize=6,
            label=label, color=COLORS[cat])
ax.set_xlabel("Launch Year")
ax.set_ylabel("Adjective Usage Rate (% of words)")
ax.set_title("Marketing Adjective Usage in Kickstarter & Indiegogo Pitches (2014–2025)", fontsize=12)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
ax.set_xticks(yearly["launch_year"])
plt.tight_layout()
plt.savefig("figures/fig1_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_trends.png")

# ── Figure 2: Changepoints ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, (cat, col) in enumerate(zip(CAT_KEYS, ADJ_COLS)):
    ax   = axes[i]
    yrs  = yearly["launch_year"].values
    vals = yearly[col].values * 100
    ax.plot(yrs, vals, color=COLORS[cat], linewidth=2.5, marker="o", markersize=6)
    ax.fill_between(yrs, vals, alpha=0.08, color=COLORS[cat])
    cp_years = cp_results.get(cat, {}).get("changepoint_years", [])
    sig = ""
    if cp_years:
        p = cp_results[cat].get("p_value", 1)
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        ax.axvline(x=cp_years[0], color="red", linestyle="--", linewidth=2, alpha=0.85)
        ax.text(cp_years[0]+0.15, max(vals)*0.97, f"CP: {cp_years[0]}", color="red", fontsize=9, va="top")
    ax.set_title(f"{LABELS[col]}  {sig}", fontsize=11)
    ax.set_xlabel("Year"); ax.set_ylabel("Rate (%)")
    ax.set_xticks(yrs[::2]); ax.grid(axis="y", alpha=0.3)
plt.suptitle("Changepoint Detection in Adjective Category Usage\n(red dashed = detected changepoint, *** p<0.001)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig2_changepoints.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_changepoints.png")

# ── Figure 3: Success by adj usage ───────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i, (cat, col) in enumerate(zip(CAT_KEYS, ADJ_COLS)):
    ax  = axes[i]
    med = df[col].median()
    df["_grp"] = df[col].apply(lambda x: "High" if x > med else "Low")
    sr  = df.groupby("_grp")["success"].mean() * 100
    sr  = sr.reindex(["Low", "High"])
    bars = ax.bar(["Low", "High"], sr.values,
                  color=["#CCCCCC", COLORS[cat]], width=0.5)
    for bar, val in zip(bars, sr.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(LABELS[col], fontsize=10)
    ax.set_ylabel("Success Rate (%)" if i==0 else "")
    ax.set_ylim(0, 90); ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Adjective Usage")
df.drop(columns="_grp", inplace=True)
plt.suptitle("Campaign Success Rate by Adjective Category Usage (High vs. Low median split)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig3_success_by_adj.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_success_by_adj.png")

# ── Figure 4: Regression coefficients ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
adj_coefs = coef_df[coef_df["feature"].isin(ADJ_COLS)].copy()
adj_coefs["label"] = adj_coefs["feature"].map(LABELS)
adj_coefs["cat"]   = adj_coefs["feature"].str.replace("rate_", "")
adj_coefs = adj_coefs.sort_values("coefficient")
colors = [COLORS[c] for c in adj_coefs["cat"]]
ax.barh(adj_coefs["label"], adj_coefs["coefficient"], color=colors, height=0.5)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Logistic Regression Coefficient (standardized)")
ax.set_title("Effect of Adjective Categories on Campaign Success")
ax.grid(axis="x", alpha=0.3)
for i, (_, row) in enumerate(adj_coefs.iterrows()):
    offset = 0.001 if row["coefficient"] >= 0 else -0.001
    ha = "left" if row["coefficient"] >= 0 else "right"
    ax.text(row["coefficient"]+offset, i, f"{row['coefficient']:+.3f}", va="center", ha=ha, fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig4_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_coefficients.png")

# ── Figure 5: AUC comparison ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
models  = ["Dummy\nBaseline", "No Adjectives", "With\nAdjectives"]
aucs    = [reg_results["auc_dummy"], reg_results["auc_baseline"], reg_results["auc_full"]]
colors  = ["#DDDDDD", "#90CAF9", "#1565C0"]
bars    = ax.bar(models, aucs, color=colors, width=0.45)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("AUC-ROC (5-fold CV)")
ax.set_title("Predictive Performance: Baseline vs. Full Model")
ax.set_ylim(0.45, min(1.0, max(aucs)+0.06))
ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig5_auc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_auc.png")

print("\nAll 5 figures saved to figures/")