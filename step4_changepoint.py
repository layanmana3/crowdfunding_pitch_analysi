# ===========================================================
# Step 4: Changepoint Detection on adjective time series
# Input:  data/processed/kickstarter_adjectives.csv
# Output: data/processed/yearly_adj_rates.csv
#         data/processed/changepoint_results.json
# ===========================================================

import pandas as pd
import numpy as np
import json
import subprocess, sys

try:
    import ruptures as rpt
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "ruptures"], check=True)
    import ruptures as rpt

from scipy import stats

# ── Load & aggregate by year ─────────────────────────────────────────────────
df = pd.read_csv("data/processed/kickstarter_adjectives.csv", low_memory=False)
df = df[df["launch_year"].between(2014, 2025)]

ADJ_COLS = ["rate_superlative","rate_urgency_scarcity","rate_emotional","rate_quality_technical"]

yearly = df.groupby("launch_year").agg(
    rate_superlative      =("rate_superlative",      "mean"),
    rate_urgency_scarcity =("rate_urgency_scarcity", "mean"),
    rate_emotional        =("rate_emotional",         "mean"),
    rate_quality_technical=("rate_quality_technical", "mean"),
    success_rate          =("success",                "mean"),
    n_projects            =("success",                "count"),
).reset_index().sort_values("launch_year")

yearly.to_csv("data/processed/yearly_adj_rates.csv", index=False)

print("Yearly summary:")
print(yearly[["launch_year","rate_superlative","rate_urgency_scarcity",
              "rate_emotional","rate_quality_technical","success_rate","n_projects"]
             ].to_string(index=False))

# ── Changepoint detection ────────────────────────────────────────────────────
CATEGORY_MAP = {
    "superlative":       "rate_superlative",
    "urgency_scarcity":  "rate_urgency_scarcity",
    "emotional":         "rate_emotional",
    "quality_technical": "rate_quality_technical",
}

cp_results = {}
print("\n--- Changepoint Detection Results ---")

for cat, col in CATEGORY_MAP.items():
    signal = yearly[col].values.reshape(-1, 1)
    sig_norm = (signal - signal.mean()) / (signal.std() + 1e-10)

    model = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(sig_norm)
    breakpoints = model.predict(pen=2.0)
    cp_years = [int(yearly["launch_year"].iloc[bp-1]) for bp in breakpoints[:-1]]

    cp_results[cat] = {"changepoint_years": cp_years, "col": col}
    print(f"  {cat:20s}: changepoint(s) at {cp_years}")

    if cp_years:
        main_cp = cp_years[0]
        before = df[df["launch_year"] <  main_cp][col]
        after  = df[df["launch_year"] >= main_cp][col]
        t_stat, p_val = stats.ttest_ind(before, after)
        cp_results[cat]["before_mean"] = float(before.mean())
        cp_results[cat]["after_mean"]  = float(after.mean())
        cp_results[cat]["t_stat"]      = float(t_stat)
        cp_results[cat]["p_value"]     = float(p_val)
        cp_results[cat]["significant"] = bool(p_val < 0.05)

        direction = "↑" if after.mean() > before.mean() else "↓"
        sig_str   = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else "ns"))
        print(f"    before={before.mean():.5f}  after={after.mean():.5f}  {direction}  p={p_val:.4f} {sig_str}")

with open("data/processed/changepoint_results.json", "w") as f:
    json.dump(cp_results, f, indent=2)

print("\nSaved -> data/processed/yearly_adj_rates.csv")
print("Saved -> data/processed/changepoint_results.json")