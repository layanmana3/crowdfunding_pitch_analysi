# ===========================================================
# Step 3: Extract & categorize adjectives from blurbs
# Input:  data/processed/kickstarter_unified.csv
# Output: data/processed/kickstarter_adjectives.csv
# ===========================================================

import pandas as pd, spacy, os

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ── 4 adjective category dictionaries ───────────────────────────────────────
SUPERLATIVES = {
    "best","greatest","largest","fastest","highest","biggest","first","ultimate",
    "top","finest","premier","leading","unmatched","unrivaled","unparalleled",
    "supreme","perfect","ideal","only","sole"
}
URGENCY_SCARCITY = {
    "limited","exclusive","rare","scarce","urgent","immediate","critical",
    "essential","vital","crucial","last","final","early","special","unique",
    "temporary","short","few","remaining","expiring","closing"
}
EMOTIONAL = {
    "amazing","incredible","awesome","fantastic","wonderful","beautiful","stunning",
    "breathtaking","inspiring","passionate","exciting","thrilling","magical",
    "extraordinary","remarkable","unforgettable","powerful","heartfelt","moving",
    "touching","joyful","fun","hilarious","adorable","lovely","charming",
    "delightful","epic","bold","brave","daring","creative","innovative",
    "revolutionary","groundbreaking"
}
QUALITY_TECHNICAL = {
    "professional","premium","quality","durable","reliable","robust","advanced",
    "smart","efficient","effective","precise","accurate","secure","safe",
    "certified","tested","proven","engineered","designed","crafted","handmade",
    "custom","sustainable","organic","natural","authentic","original","genuine",
    "lightweight","compact","portable","wireless","digital","modular","waterproof",
    "eco-friendly","high-quality"
}

def categorize(word):
    w = word.lower()
    cats = []
    if w in SUPERLATIVES:      cats.append("superlative")
    if w in URGENCY_SCARCITY:  cats.append("urgency_scarcity")
    if w in EMOTIONAL:         cats.append("emotional")
    if w in QUALITY_TECHNICAL: cats.append("quality_technical")
    return cats or ["other"]

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/kickstarter_unified.csv", low_memory=False)

# Fix: use launched_date for Kickstarter, snapshot_year for Indiegogo
df["launched_date"] = pd.to_datetime(df["launched_date"], errors="coerce")
df["launch_year"]   = df["launched_date"].dt.year

# For Indiegogo (no launched_date): use snapshot_year
mask = df["launch_year"].isna()
df.loc[mask, "launch_year"] = pd.to_numeric(df.loc[mask, "snapshot_year"], errors="coerce")

df = df[df["launch_year"].between(2014, 2025)].copy()
df = df[df["blurb"].notna() & (df["blurb"].str.len() >= 10)].copy()
df = df.drop_duplicates(subset="id").reset_index(drop=True)

print(f"Processing {len(df):,} projects...")
print(f"By source:")
print(df.groupby("source")["launch_year"].count().to_string())

# ── Process in batches ───────────────────────────────────────────────────────
BATCH = 2000
results = []

for i in range(0, len(df), BATCH):
    batch = df.iloc[i:i+BATCH]
    docs  = list(nlp.pipe(batch["blurb"].tolist(), batch_size=256))

    for j, doc in enumerate(docs):
        row  = batch.iloc[j]
        adjs = [t.lemma_.lower() for t in doc
                if t.pos_ == "ADJ" and t.is_alpha and len(t.text) > 2]

        cnt = {"superlative":0,"urgency_scarcity":0,
               "emotional":0,"quality_technical":0,"other_adj":0}
        for a in adjs:
            for c in categorize(a):
                cnt["other_adj" if c == "other" else c] += 1

        wc = max(len([t for t in doc if t.is_alpha]), 1)
        results.append({
            "id":                     row["id"],
            "source":                 row["source"],
            "launch_year":            row["launch_year"],
            "success":                row["success"],
            "category":               row["category"],
            "parent_category":        row["parent_category"],
            "country":                row["country"],
            "goal_usd":               row["goal_usd"],
            "blurb_len":              row["blurb_len"],
            "word_count":             wc,
            "total_adj":              len(adjs),
            "adj_rate":               len(adjs)/wc,
            "n_superlative":          cnt["superlative"],
            "n_urgency_scarcity":     cnt["urgency_scarcity"],
            "n_emotional":            cnt["emotional"],
            "n_quality_technical":    cnt["quality_technical"],
            "rate_superlative":       cnt["superlative"]/wc,
            "rate_urgency_scarcity":  cnt["urgency_scarcity"]/wc,
            "rate_emotional":         cnt["emotional"]/wc,
            "rate_quality_technical": cnt["quality_technical"]/wc,
        })

    if (i // BATCH) % 20 == 0:
        print(f"  {min(i+BATCH, len(df)):,} / {len(df):,}")

out = pd.DataFrame(results)
out.to_csv("data/processed/kickstarter_adjectives.csv", index=False)

print("\nAverage rates per year (all sources):")
print(out.groupby("launch_year")[[
    "rate_superlative","rate_urgency_scarcity",
    "rate_emotional","rate_quality_technical"
]].mean().round(5).to_string())

print("\nProjects per source:")
print(out.groupby("source")["success"].agg(
    count="count", success_rate="mean"
).round(3).to_string())

print(f"\nSaved -> data/processed/kickstarter_adjectives.csv")