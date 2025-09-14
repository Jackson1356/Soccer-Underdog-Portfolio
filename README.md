# Underdog Advisor — Exact Columns, Unified Portfolio

Your data uses the exact column names (no renaming). Date is not required. FTHG/FTAG are integers and FTR is in {H,D,A}.

This toolkit trains on `./data` and recommends a **unified portfolio** (singles + 2-leg parlays) under two modes:
- **Low risk**: conservative Kelly caps and parlay exposure
- **Higher risk**: larger caps and parlay exposure

---

## Quick Start

```bash
# Train models (no date features/backtests)
python train_model.py --data_dir ./data --models_dir models

# Recommend unified portfolios (low/high) from a fixture file with the same columns
python recommend.py --fixture_file tonight.csv --models_dir models --bankroll 1000 --max_games 5 --max_picks 8
```

Outputs:
- `models/` → `moneyline.pkl`, `over25.pkl`, `ah_home_ev.pkl`, `ah_away_ev.pkl`, `meta.json`
- `portfolio_low.csv`, `portfolio_high.csv` — combined singles + parlays with stakes
- `portfolio_low_summary.csv`, `portfolio_high_summary.csv` — bankroll used, expected profit, approx risk std

---

## Columns expected (exact names)

`['Div','HomeTeam','AwayTeam','HS','AS','HST','AST','B365H','B365D','B365A','B365C>2.5','B365C<2.5','AHh','B365AHH','B365AHA','FTHG','FTAG','FTR']`

Some columns can be missing; models will just skip the affected market or use fewer features.

---

## Modeling & Portfolio

- **Moneyline (1X2):** Multinomial logistic regression → \(P(H), P(D), P(A)\) and EV per side.
- **Totals 2.5:** Gradient Boosting for \(P(\text{Over2.5})\); pick Over/Under by EV.
- **Asian Handicap:** Two regressors predict **expected profit per unit** (Home/Away) handling quarter-lines.

We construct candidates (singles + 2-leg parlays), compute per-unit \(\mu\) and \(\sigma\), score via \(S=\mu-\lambda\sigma\), then allocate stakes using **capped Kelly** while respecting:
- per-bet caps (different for singles vs parlays),
- total parlay pool cap,
- max distinct games and max total picks.

---

## Dashboard

Open `advisor_dashboard.html` and upload:
- `portfolio_low.csv` and `portfolio_high.csv` (and summaries if you want). It shows stake distribution and expected profit.
