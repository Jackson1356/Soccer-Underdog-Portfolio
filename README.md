# Underdog Advisor — Unified Singles + Parlays

Train on your historical data (no date features, no backtest), then recommend a **unified portfolio** that mixes **single bets** and **2-leg parlays** under two styles:

1) **Low risk** — smaller Kelly caps, tight parlay exposure, risk-averse allocation.  
2) **Higher risk** — larger caps, more parlay exposure, higher potential gain.

The optimizer allocates your bankroll **across both** singles *and* parlays simultaneously — it does **not** split them into separate lists.

---

## Quick Start

```bash
# 1) Train models from ./data
python train_model.py --data_dir ./data --models_dir models

# 2) Recommend unified portfolios (produces low & high)
python recommend.py --fixture_file tonight.csv --models_dir models --bankroll 1000 --max_games 5 --max_picks 8

# 3) Open the dashboard and upload the outputs
open advisor_dashboard.html
# Upload: portfolio_low.csv, portfolio_high.csv
# (Optional) also upload portfolio_low_summary.csv, portfolio_high_summary.csv
```

**Outputs**
- `models/` → `moneyline.pkl`, `over25.pkl`, `ah_home_ev.pkl`, `ah_away_ev.pkl`, `meta.json`
- `portfolio_low.csv` / `portfolio_high.csv` → unified allocation across singles + parlays
- `portfolio_*_summary.csv` → bankroll used, expected profit, approximate risk std

---

## Markets & Models (no dates needed)

- **Moneyline (1X2):** Multinomial logistic regression producing \\(P(H), P(D), P(A)\\).  
  Features: market odds (B365H/D/A), AH line & odds (if present), engineered odds spreads, and optional shots/SOT if available.
- **Totals 2.5:** Gradient Boosting classifier for \\(P(\\text{Over2.5})\\); EV computed for Over/Under.
- **Asian Handicap:** Two Gradient Boosting regressors predict **expected profit per unit** for Home and for Away, correctly handling quarter-lines (±0.25/±0.75) by splitting stakes into half-lines.

For any binary bet with probability \\(p\\) and decimal odds \\(o\\):  
\\[
\\text{EV} = p(o-1) - (1-p),\\quad
\\sigma = \\sqrt{p\\,(o-1-\\mu)^2 + (1-p)(-1-\\mu)^2},\\ \\mu=\\text{EV}.
\\]

---

## Portfolio Construction (what gets recommended)

1. **Candidate set**
   - Singles: best sides across Moneyline (H/D/A), Totals 2.5 (Over/Under), AH (Home/Away).
   - Parlays: top two-leg combos built from the best singles (skip same-match pairs).  
     Probability \\(p_{AB}\\approx p_A p_B\\), odds \\(o_{AB}=o_A o_B\\), EV from the same formula.

2. **Risk-adjusted score**
   For each candidate \(i\), compute per-unit mean \\(\\mu_i\\) and std \\(\\sigma_i\\).  
   Score: \\(S_i = \\mu_i - \\lambda\\,\\sigma_i\\). Keep \\(S_i>0\\).

   - **Low risk:** \\(\\lambda=0.6\\); per-bet caps 6% (singles) / 2% (parlays); parlay pool ≤ 20% of bankroll.
   - **Higher risk:** \\(\\lambda=0.25\\); caps 15% / 6%; parlay pool ≤ 50%.
   - Kelly fraction with small **mode-specific caps** (3%/1% low; 10%/4% high).
   - Respect **max distinct games** and **max picks**.

3. **Greedy allocation**
   Sort by score and allocate stake = \\(\\min(\\text{bankroll}\\cdot f, \\text{per-bet cap})\\) while obeying the total parlay cap and the game/pick limits. Keep a small cash buffer.

> Result: a single CSV per mode containing **both** singles and parlays with **Stake** and **Stake%**.

---

## Dashboard

`advisor_dashboard.html` lets you upload the two portfolios and shows stake distributions and expected profit. It also builds the exact shell command for your fixture file and bankroll.

---

## Practical Notes

- Missing columns are fine; that market is skipped.
- Parlay independence is an approximation — use conservative caps for strings.
- Always bet responsibly; never stake more than you can afford to lose.
