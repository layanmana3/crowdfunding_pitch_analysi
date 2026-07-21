# Crowdfunding Pitch Language Analysis

A longitudinal cross-platform study of marketing adjective usage and sentiment in Kickstarter and Indiegogo crowdfunding campaigns (2014–2025) and their correlation with fundraising success.

## Research Questions
1. Is there a detectable changepoint in adjective usage in crowdfunding pitches across 2014–2025?
2. Do such shifts correlate with campaign fundraising success?
3. Do rhetorical patterns differ between Kickstarter and Indiegogo?
4. Do adjective usage patterns vary across project categories?
5. How has overall pitch sentiment evolved, and does it predict success?

## Key Findings
- **Superlative** adjectives declined significantly from **2021** (p < 0.001, −30%)
- **Emotional** adjectives declined from **2016** (p < 0.001, −14%)
- **Urgency/Scarcity** is the strongest predictor of success (β = +0.068, Δ = +10.7pp)
- **Film & Video** campaigns use the most urgency/scarcity language (1.35%)
- Pitch **polarity and subjectivity** have both declined since 2016 — pitches are becoming more factual
- Highly **subjective** pitches achieve 3.5pp higher success rates
- Combined corpus: **340,055 campaigns** (278,628 Kickstarter + 61,427 Indiegogo)

## Data Sources
- **Kickstarter**: 6 snapshots (Jan 2016, 2018, 2020, 2022, 2024, 2026) via [Web Robots](https://webrobots.io/kickstarter-datasets/)
- **Indiegogo**: 6 snapshots (2020–2025) via [Web Robots](https://webrobots.io/indiegogo-datasets/)

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
python run_all.py
```

Or step by step:

```bash
python step2_build_dataset.py        # Download & build dataset — Kickstarter + Indiegogo (~45 min)
python step3_extract_adjectives.py   # Extract & categorize adjectives with spaCy (~20 min)
python step4_changepoint.py          # Detect changepoints (PELT algorithm)
python step5_regression.py           # Logistic regression analysis
python step6_visualizations.py       # Generate figures 1–5
python step7_platform_comparison.py  # Cross-platform comparison (figures 6–7)
python step8_category_analysis.py    # Category analysis + Word Cloud (figures 8–9)
python step9_sentiment_analysis.py   # Sentiment analysis with TextBlob (figures 10–11)
```

## Project Structure

```
crowdfunding_pitch_analysis/
├── run_all.py
├── step2_build_dataset.py           # Kickstarter + Indiegogo data collection
├── step3_extract_adjectives.py      # spaCy POS tagging + adjective classification
├── step4_changepoint.py             # PELT changepoint detection
├── step5_regression.py              # Logistic regression + AUC evaluation
├── step6_visualizations.py          # Figures 1–5: trends, changepoints, regression
├── step7_platform_comparison.py     # Figures 6–7: Kickstarter vs Indiegogo
├── step8_category_analysis.py       # Figures 8–9: by category + word cloud
├── step9_sentiment_analysis.py      # Figures 10–11: sentiment trends & success
├── requirements.txt
├── data/
│   ├── raw/                         # Raw JSON snapshots (not on GitHub)
│   └── processed/                   # Processed CSVs
├── figures/                         # 11 output figures for paper
└── notebooks/
    └── Crowdfunding_Pitch_Analysis.ipynb
```

## Output Figures

| Figure | Description |
|--------|-------------|
| fig1_trends.png | Adjective category rates over time |
| fig2_changepoints.png | PELT changepoint detection |
| fig3_success_by_adj.png | Success rate by adjective usage |
| fig4_coefficients.png | Logistic regression coefficients |
| fig5_auc.png | AUC comparison |
| fig6_platform_comparison.png | Kickstarter vs Indiegogo trends |
| fig7_category_comparison.png | Platform success advantage |
| fig8_category_adj.png | Adjective usage by project category |
| fig9_wordclouds.png | Word clouds before/after 2019 |
| fig10_sentiment_trends.png | Sentiment polarity & subjectivity over time |
| fig11_sentiment_success.png | Success rate by sentiment |

## Acknowledgments
This project used Claude (Anthropic) as a coding assistant.
