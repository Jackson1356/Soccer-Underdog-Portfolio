# Underdog Advisor — Opening vs Closing Models & Unified Portfolios

This project learns from **opening and closing odds/lines** to identify **value on underdogs/outliers**, then allocates bankroll across **singles + 2‑leg parlays** under two risk styles (low / higher).

## Required Columns (every training file & any fixture)
- 1X2 open/close: `B365H, B365D, B365A, B365CH, B365CD, B365CA`
- Totals 2.5 open/close: `B365>2.5, B365<2.5, B365C>2.5, B365C<2.5`
- Asian handicap open/close: `AHh, B365AHH, B365AHA, AHCh, B365CAHH, B365CAHA`
- Labels to train: `FTR` for 1X2; `FTHG, FTAG` for Totals/AH. (Not required for recommending)

Files missing any required columns are **skipped** during training. The recommender will **error** with a list of missing cols.

## Modeling
We deliberately avoid team IDs or dates to isolate information in **prices & their movements**.

### Moneyline (1X2) — Multinomial Logistic
Softmax probabilities \(P(H),P(D),P(A)\) from opening/closing levels and deltas:
\[ P(y=k\mid x) = \frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}. \]
Loss: cross-entropy (log-loss).

### Totals (Over/Under 2.5) — Gradient Boosting Classifier
Label \(y=1[\text{FTHG}+\text{FTAG}>2.5]\). Features: O/U open & close, deltas, and 1X2 close. Output: \(P(\text{Over})\).

### Asian Handicap EV — Gradient Boosting Regressors
Target = per‑unit net profit. Quarter-lines are split across adjacent half-lines; profit per part is \(o-1\) (win), 0 (push), \(-1)\) (loss).

## EV, Risk & Stakes
- **EV (per unit)**: \( \mathrm{EV}(p,o) = p(o-1) - (1-p) \). Positive EV iff \( p > 1/o \).
- **Variance (per unit)**: payoff \(+o-1\) w.p. \(p\), else \(-1\).  
  \( \sigma = \sqrt{ p(o-1-\mu)^2 + (1-p)(-1-\mu)^2 } \), where \(\mu\) is EV per unit.
- **Score**: \( S = \mu - \lambda\,\sigma \) (λ higher for “low risk”).
- **Stake**: capped **Kelly** fraction \( f = \max(0, ((o-1)p - (1-p))/(o-1)) \) with per‑bet and parlay caps.
- **Parlays**: form 2‑legs from top singles, block same‑match combos; independence is an approximation ⇒ parlay cap.

## Usage
```bash
python train_model.py --data_dir ./data --models_dir models
python recommend.py --fixture_file tonight.csv --models_dir models --bankroll 1000 --max_games 5 --max_picks 8
```
Outputs: `portfolio_low.csv`, `portfolio_high.csv` + `_summary.csv` for each. Open `advisor_dashboard.html` to validate fixtures and visualize portfolios.

## Extensions
- Use overround-adjusted implied probabilities and their movements.
- Add probability calibration (isotonic/Platt) for 1X2 & Totals.
- Draw-aware class weights in 1X2 (`class_weight='balanced'`).

**Disclaimer:** Betting involves risk. Never bet more than you can afford to lose.
