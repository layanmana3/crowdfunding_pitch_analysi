# Crowdfunding Pitch Language Analysis

A longitudinal study of marketing adjective usage in Kickstarter and Indiegogo crowdfunding campaigns (2014–2025) and its correlation with fundraising success.

## Research Questions
1. Is there a detectable changepoint in the use of specific adjective categories in crowdfunding pitches across 2014–2025?
2. Do such shifts correlate with campaign fundraising success?

## Key Findings
- **Superlative** adjectives declined significantly from 2021 (p < 0.001)
- **Emotional** adjectives declined from 2016 (p < 0.001)
- **Urgency/Scarcity** language shows the strongest positive association with success (+10.7pp)
- Combined corpus: **340,055 campaigns** (Kickstarter + Indiegogo)

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
python step2_build_dataset.py       # Download & build dataset (~45 min)
python step3_extract_adjectives.py  # Extract adjectives with spaCy (~20 min)
python step4_changepoint.py         # Detect changepoints (PELT algorithm)
python step5_regression.py          # Logistic regression analysis
python step6_visualizations.py      # Generate 5 figures
```

## Project Structure

```
crowdfunding_pitch_analysis/
├── run_all.py
├── step2_build_dataset.py
├── step3_extract_adjectives.py
├── step4_changepoint.py
├── step5_regression.py
├── step6_visualizations.py
├── requirements.txt
├── data/
│   ├── raw/               # Raw JSON snapshots (not on GitHub)
│   └── processed/         # Processed CSVs
├── figures/               # 5 output figures for paper
└── notebooks/             # Colab notebook
```

## Acknowledgments
This project used Claude (Anthropic) as a coding assistant.