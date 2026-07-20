# ===========================================================
# Step 5: Logistic Regression — adjective categories vs success
# Input:  data/processed/kickstarter_adjectives.csv
#         data/processed/changepoint_results.json
# Output: data/processed/regression_coefficients.csv
#         data/processed/regression_results.json
# ===========================================================

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from scipy import stats

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/kickstarter_adjectives.csv", low_memory=False)
df = df[df["launch_year"].between(2014, 2025)].copy()
df = df.dropna(subset=["success", "goal_usd", "blurb_len"]).copy()
df = df[df["goal_usd"] >= 0].copy()

with open("data/processed/changepoint_results.json") as f:
    cp_results = json.load(f)

ADJ_FEATURES = [
    "rate_superlative", "rate_urgency_scarcity",
    "rate_emotional",   "rate_quality_technical"
]

# ── Feature engineering ──────────────────────────────────────────────────────
df["log_goal"] = np.log1p(df["goal_usd"])

top_cats = df["parent_category"].value_counts().nlargest(10).index
df["parent_cat"] = df["parent_category"].where(df["parent_category"].isin(top_cats), "Other")
cat_dummies = pd.get_dummies(df["parent_cat"], prefix="cat", drop_first=True)

CONTROLS = ["log_goal", "blurb_len", "adj_rate"]
y = df["success"]

X_base = pd.concat([df[CONTROLS], cat_dummies], axis=1).fillna(0)
X_full = pd.concat([df[ADJ_FEATURES + CONTROLS], cat_dummies], axis=1).fillna(0)

scaler_b = StandardScaler(); X_base_s = scaler_b.fit_transform(X_base)
scaler_f = StandardScaler(); X_full_s = scaler_f.fit_transform(X_full)

# ── Cross-validated AUC ──────────────────────────────────────────────────────
auc_dummy = cross_val_score(DummyClassifier(), X_base_s, y, cv=5, scoring="roc_auc").mean()
auc_base  = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), X_base_s, y, cv=5, scoring="roc_auc").mean()
auc_full  = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), X_full_s, y, cv=5, scoring="roc_auc").mean()

print(f"Dummy baseline AUC : {auc_dummy:.4f}")
print(f"Baseline AUC       : {auc_base:.4f}")
print(f"Full model AUC     : {auc_full:.4f}")
print(f"Improvement        : +{auc_full - auc_base:.4f}")

# ── Fit full model for coefficients ─────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_full_s, y)
feat_names = ADJ_FEATURES + CONTROLS + list(cat_dummies.columns)
coef_df = pd.DataFrame({
    "feature": feat_names,
    "coefficient": lr.coef_[0]
}).sort_values("coefficient", ascending=False)

print("\nAdjective feature coefficients:")
print(coef_df[coef_df["feature"].isin(ADJ_FEATURES)].to_string(index=False))

# ── Before/after changepoint: success rate ───────────────────────────────────
print("\n--- Before/After Changepoint: Success Rate ---")
before_after = {}
for cat, col in zip(
    ["superlative", "urgency_scarcity", "emotional", "quality_technical"],
    ADJ_FEATURES
):
    cp_years = cp_results.get(cat, {}).get("changepoint_years", [])
    if not cp_years:
        continue
    main_cp = cp_years[0]

    before_df = df[df["launch_year"] <  main_cp]
    after_df  = df[df["launch_year"] >= main_cp]
    med_b = before_df[col].median()
    med_a = after_df[col].median()

    hi_b = before_df[before_df[col] > med_b]["success"].mean()
    lo_b = before_df[before_df[col] <= med_b]["success"].mean()
    hi_a = after_df[after_df[col] > med_a]["success"].mean()
    lo_a = after_df[after_df[col] <= med_a]["success"].mean()

    before_after[cat] = {
        "changepoint_year":        main_cp,
        "high_adj_success_before": float(hi_b),
        "low_adj_success_before":  float(lo_b),
        "high_adj_success_after":  float(hi_a),
        "low_adj_success_after":   float(lo_a),
    }
    print(f"\n  {cat} (CP={main_cp}):")
    print(f"    Before: high={hi_b:.1%}  low={lo_b:.1%}  diff={hi_b-lo_b:+.1%}")
    print(f"    After:  high={hi_a:.1%}  low={lo_a:.1%}  diff={hi_a-lo_a:+.1%}")

# ── Save ─────────────────────────────────────────────────────────────────────
coef_df.to_csv("data/processed/regression_coefficients.csv", index=False)

reg_results = {
    "auc_dummy":       float(auc_dummy),
    "auc_baseline":    float(auc_base),
    "auc_full":        float(auc_full),
    "auc_improvement": float(auc_full - auc_base),
    "before_after":    before_after,
}
with open("data/processed/regression_results.json", "w") as f:
    json.dump(reg_results, f, indent=2)

print("\nSaved -> data/processed/regression_coefficients.csv")
print("Saved -> data/processed/regression_results.json")