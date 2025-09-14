# Underdog Advisor — Opening vs Closing Odds

This repo trains models **only** from opening & closing odds/lines and builds unified portfolios (singles + 2‑leg parlays).

## Required Columns (in every training file and fixture)
- 1X2 open/close: `B365H, B365D, B365A, B365CH, B365CD, B365CA`
- Totals open/close: `B365>2.5, B365<2.5, B365C>2.5, B365C<2.5`
- AH open/close: `AHh, B365AHH, B365AHA, AHCh, B365CAHH, B365CAHA`

Training also needs labels: `FTR` for 1X2; `FTHG, FTAG` for Totals/AH.

## Usage
```bash
python train_model.py --data_dir ./data --models_dir models
python recommend.py --fixture_file tonight.csv --models_dir models --bankroll 1000 --max_games 5 --max_picks 8
```

- Recommender uses **closing odds** for EV; falls back to opening where closing is NaN.
- Portfolios mix singles + 2‑leg parlays with risk controls and capped Kelly staking.
