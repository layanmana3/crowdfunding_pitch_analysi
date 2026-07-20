# ===========================================================
# run_all.py — Run the entire pipeline from scratch
# Usage: python run_all.py
# ===========================================================

import subprocess, sys, os, time

STEPS = [
    ("step2_build_dataset.py",     "Build unified dataset — Kickstarter + Indiegogo (~45 min)"),
    ("step3_extract_adjectives.py","Extract adjectives with spaCy (~20 min)"),
    ("step4_changepoint.py",       "Changepoint detection"),
    ("step5_regression.py",        "Logistic regression analysis"),
    ("step6_visualizations.py",    "Generate all figures"),
]

# Skip step2 if data already exists
if os.path.exists("data/processed/kickstarter_unified.csv"):
    print("kickstarter_unified.csv already exists — skipping step2.")
    STEPS = STEPS[1:]

for script, description in STEPS:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Running: {script}")
    print("="*60)
    t0 = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nERROR in {script}. Fix the error and re-run.")
        sys.exit(1)
    print(f"  Completed in {elapsed:.1f}s")

print("\n" + "="*60)
print("  ALL STEPS COMPLETE!")
print("  data/processed/ — CSV files")
print("  figures/        — 5 figures for the paper")
print("="*60)