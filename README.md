# Underdog Advisor (Starter Kit)

This kit gives you an end-to-end pipeline to find likely **underdog surprises** and allocate stakes for a batch of games on the same day.

## 1) Train the model & produce recommendations

```bash
python underdog_model.py --data_dir ./data --bankroll 1000 --top_k 12 --upset_bias 1.1
# or for a specific match date
python underdog_model.py --data_dir ./data --date 2024-12-26 --bankroll 1200 --top_k 15
```

- Outputs:
  - `model.pkl`, `features.json`
  - `recommendations.csv` with columns: `Date,Div,HomeTeam,AwayTeam,B365H,B365D,B365A,p_upset,EV_H,EV_D,EV_A,EV_max,EV_side,stake_conservative,stake_balanced,stake_risky`

## 2) Re-allocate bankroll after the fact

```bash
python bet_allocator.py --reco_csv recommendations.csv --bankroll 1500 --mode balanced
```

## 3) Visualize in the web dashboard

Open `advisor_dashboard.html` in your browser and load `recommendations.csv` via the file picker. The look-and-feel is compatible with your existing tracker page so you can iframe or merge sections.

## Modeling notes

- **Target (upset):** We mark an upset if the realized outcome (H/D/A) had the *lowest* normalized implied probability from Bet365 odds **or** that realized probability was `< 0.33`.
- **Features:** odds (B365H/D/A), Asian handicap line & odds if present, shots/SOT when available, and **recency form** for both teams (rolling last-5 goals for/against and average points).
- **Model:** `GradientBoostingClassifier` (sklearn). You can swap in XGBoost/LightGBM later.
- **Ranking:** For a given match day we compute model `p_upset` then expected value (EV) for each side via `EV = p*(odds-1) - (1-p)` and keep the best side per game.
- **Allocation:** Kelly fraction with caps → conservative (5%), balanced (15%), risky (30%).

## Integrating with your tracker

- This dashboard mirrors the styling of your **Soccer Betting Tracker** (colors, tables, Chart.js). You can embed it or copy its components into your existing `soccer_betting.html`.
