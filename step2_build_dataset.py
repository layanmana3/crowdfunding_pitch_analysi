# ===========================================================
# Step 2: Build unified dataset from Kickstarter + Indiegogo
# Output: data/processed/kickstarter_unified.csv
# Sources: Web Robots (webrobots.io)
# ===========================================================

import urllib.request, gzip, json, os
import pandas as pd

# ── Kickstarter snapshots ────────────────────────────────────────────────────
KS_SNAPSHOTS = [
    ("2016", "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2016-01-28T09_15_08_781Z.json.gz"),
    ("2018", "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2018-01-12T10_20_09_196Z.json.gz"),
    ("2020", "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2020-01-16T03_20_15_556Z.json.gz"),
    ("2022", "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2022-01-20T03_20_11_451Z.json.gz"),
    ("2024", "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2024-01-15T14_13_05_649Z.json.gz"),
    ("2026", "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2026-01-12T09_37_51_016Z.json.gz"),
]

# ── Indiegogo snapshots (2020+ are accessible) ───────────────────────────────
IGG_SNAPSHOTS = [
    ("2020", "https://s3.amazonaws.com/weruns/forfun/Indiegogo/Indiegogo_2020-01-17T10_40_12_316Z.json.gz"),
    ("2021", "https://s3.amazonaws.com/weruns/forfun/Indiegogo/Indiegogo_2021-01-15T10_41_39_329Z.json.gz"),
    ("2022", "https://s3.amazonaws.com/weruns/forfun/Indiegogo/Indiegogo_2022-01-21T20_40_42_239Z.json.gz"),
    ("2023", "https://s3.amazonaws.com/weruns/forfun/Indiegogo/Indiegogo_2023-01-13T20_40_43_372Z.json.gz"),
    ("2024", "https://s3.amazonaws.com/weruns/forfun/Indiegogo/Indiegogo_2024-01-15T14_13_29_668Z.json.gz"),
    ("2025", "https://s3.amazonaws.com/weruns/forfun/Indiegogo/Indiegogo_2025-01-13T07_43_23_952Z.json.gz"),
]

os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Kickstarter parser ───────────────────────────────────────────────────────
def parse_kickstarter(line, snapshot_year):
    try:
        obj = json.loads(line)
        p   = obj.get("data", obj)
        if not isinstance(p, dict): return None
        state = p.get("state", "")
        if state not in ("successful", "failed"): return None
        blurb = (p.get("blurb") or "").strip()
        if len(blurb) < 10: return None
        cat = p.get("category") or {}
        return {
            "source":          "kickstarter",
            "snapshot_year":   snapshot_year,
            "id":              f"ks_{p.get('id')}",
            "name":            p.get("name", ""),
            "blurb":           blurb,
            "category":        cat.get("name", ""),
            "parent_category": cat.get("parent_name", ""),
            "country":         p.get("country", ""),
            "goal_usd":        float(p.get("goal", 0) or 0),
            "pledged_usd":     float(p.get("usd_pledged") or p.get("pledged") or 0),
            "backers":         p.get("backers_count", 0),
            "state":           state,
            "success":         1 if state == "successful" else 0,
            "launched_at":     p.get("launched_at"),
            "deadline":        p.get("deadline"),
        }
    except Exception:
        return None

# ── Indiegogo parser ─────────────────────────────────────────────────────────
def parse_indiegogo(line, snapshot_year):
    try:
        obj = json.loads(line)
        p   = obj.get("data", obj)
        if not isinstance(p, dict): return None

        # tagline = short pitch text (equivalent to Kickstarter blurb)
        blurb = (p.get("tagline") or p.get("bullet_point") or "").strip()
        if len(blurb) < 10: return None

        # funds_raised_percent is decimal (0.075 = 7.5%)
        pct_raw = float(p.get("funds_raised_percent") or 0)
        pct     = pct_raw * 100 if pct_raw <= 1 else pct_raw

        # Indiegogo success definition:
        # Most campaigns are Flexible Funding — creator gets money regardless
        # So we define success as: raised at least 25% of goal
        # This is a reasonable threshold that filters out campaigns with
        # almost no backers vs genuinely funded campaigns
        success = 1 if pct >= 25 else 0
        state   = "successful" if success == 1 else "failed"

        pledged = float(p.get("funds_raised_amount") or 0)

        return {
            "source":          "indiegogo",
            "snapshot_year":   snapshot_year,
            "id":              f"igg_{p.get('project_id') or p.get('id')}",
            "name":            p.get("title", ""),
            "blurb":           blurb,
            "category":        p.get("category", ""),
            "parent_category": p.get("category", ""),
            "country":         "",
            "goal_usd":        0.0,
            "pledged_usd":     pledged,
            "backers":         0,
            "state":           state,
            "success":         success,
            "launched_at":     None,
            "deadline":        None,
        }
    except Exception:
        return None

# ── Download + parse helper ──────────────────────────────────────────────────
def download_file(url, path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response, \
         open(path, "wb") as out_f:
        out_f.write(response.read())

def process_snapshots(snapshots, parser_fn, source_name):
    all_frames = []
    for year, url in snapshots:
        gz = f"data/raw/{source_name}_{year}.json.gz"
        if not os.path.exists(gz):
            print(f"[{source_name} {year}] Downloading...")
            try:
                download_file(url, gz)
                print(f"[{source_name} {year}] Done.")
            except Exception as e:
                print(f"[{source_name} {year}] SKIP — {e}")
                continue
        else:
            print(f"[{source_name} {year}] Cached.")

        records = []
        try:
            with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    r = parser_fn(line.strip(), year)
                    if r: records.append(r)
        except Exception as e:
            print(f"[{source_name} {year}] Error reading: {e}")
            continue

        if not records:
            print(f"[{source_name} {year}] WARNING: no valid records!")
            continue

        df_y = pd.DataFrame(records).drop_duplicates(subset="id")
        print(f"[{source_name} {year}] {len(df_y):,} projects | "
              f"success={df_y['success'].mean():.1%} | "
              f"avg blurb={df_y['blurb'].str.len().mean():.0f} chars")
        all_frames.append(df_y)
    return all_frames

# ── Run both sources ─────────────────────────────────────────────────────────
print("=" * 55)
print("KICKSTARTER")
print("=" * 55)
ks_frames = process_snapshots(KS_SNAPSHOTS, parse_kickstarter, "kickstarter")

print("\n" + "=" * 55)
print("INDIEGOGO")
print("=" * 55)
igg_frames = process_snapshots(IGG_SNAPSHOTS, parse_indiegogo, "indiegogo")

# ── Merge all ────────────────────────────────────────────────────────────────
all_frames = ks_frames + igg_frames
df = pd.concat(all_frames, ignore_index=True)

df["launched_date"] = pd.to_datetime(df["launched_at"], unit="s", errors="coerce")
df["deadline_date"] = pd.to_datetime(df["deadline"],    unit="s", errors="coerce")
df["duration_days"] = (df["deadline_date"] - df["launched_date"]).dt.days
df["blurb_len"]     = df["blurb"].str.len()

out = "data/processed/kickstarter_unified.csv"
df.to_csv(out, index=False)

print("\n" + "=" * 55)
print("COMBINED DATASET SUMMARY")
print("=" * 55)
print(f"Total projects: {len(df):,}")
print(f"\nBy source:")
print(df.groupby("source")["success"].agg(
    count="count", success_rate="mean"
).round(3).to_string())
print(f"\nSaved -> {out}")