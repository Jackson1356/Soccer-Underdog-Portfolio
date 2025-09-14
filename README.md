# Underdog Advisor — Unified Betting Portfolios

This repository trains models on your historical soccer data and produces **unified portfolios** that combine **single bets** and **2‑leg parlays**. Two portfolio styles are generated per fixture file:

- **Low risk**: conservative caps and limited parlay exposure.  
- **Higher risk**: larger caps and more parlay exposure.

> The trainer **skips files** that lack the minimum columns for at least one market (moneyline / totals / Asian handicap). No date fields are required.

---

## Features
- **Markets**: Moneyline (1X2), Totals (Over/Under 2.5), Asian Handicap (quarter lines supported).
- **Models**: Multinomial logistic (1X2), Gradient Boosting (Over 2.5), Gradient Boosting regressors for AH **expected profit per unit**.
- **Unified allocation**: Portfolios mix singles and parlays; stakes are allocated to maximize a **risk‑adjusted expected value** subject to caps.
- **Robust IO**: Exact column names; some columns may be missing. The trainer **skips unusable files**; the recommender auto‑adds only the features required by trained models.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install numpy pandas scikit-learn openpyxl
```

---

## Data Format (exact column names)

Your files can be CSV/XLSX. Columns may be missing; see minimal requirements below.

Common columns (use exact names, all optional unless listed as "Required"):
```
Div, HomeTeam, AwayTeam, HS, AS, HST, AST,
B365H, B365D, B365A, B365C>2.5, B365C<2.5, AHh, B365AHH, B365AHA,
FTHG, FTAG, FTR
```

**Minimum to be usable by at least one model (per file):**
- Moneyline: `FTR, B365H, B365D, B365A, HomeTeam, AwayTeam`
- Totals: `FTHG, FTAG`
- Asian Handicap: `FTHG, FTAG, AHh, B365AHH, B365AHA`

If a file doesn’t satisfy **any** of the above sets, it’s skipped during training.

---

## Quick Start

```bash
# 1) Train models from ./data  (skips unusable files automatically)
python train_model.py --data_dir ./data --models_dir models

# 2) Recommend low/high unified portfolios from a fixture file with the same columns
python recommend.py --fixture_file tonight.csv --models_dir models --bankroll 1000 --max_games 5 --max_picks 8
```

**Outputs**
- `models/` → `moneyline.pkl`, `over25.pkl`, `ah_home_ev.pkl`, `ah_away_ev.pkl`, `meta.json`
- `portfolio_low.csv` / `portfolio_high.csv` — unified allocations (singles + parlays) with `Stake` and `StakePct`
- `portfolio_low_summary.csv` / `portfolio_high_summary.csv` — bankroll used, expected profit, approx risk std

---

## How It Works

### Modeling
- **Moneyline (1X2)** — Multinomial logistic regression returns \\(P(H), P(D), P(A)\\). EV per side:
  \\[ \mathrm{EV}=p(o-1)-(1-p). \\]
- **Totals (2.5)** — Gradient Boosting classifier for \\(P(\mathrm{Over2.5})\\); we compute EV for Over and Under and keep the better one.
- **Asian Handicap** — Two regressors (Home/Away) predict **expected profit per unit** and correctly handle quarter lines by splitting the stake across half‑lines.

### Portfolio construction
1. Build **candidates**: best sides across all three markets + top 2‑leg parlays from strong singles (no same‑match pairs).
2. Compute per‑unit mean \\( \mu \\) and stdev \\( \sigma \\) (from the binary payoff distribution).
3. Score each bet with \\( S=\mu-\lambda\sigma \\) (larger \\( \lambda \\) = more risk‑averse).
4. **Greedy allocation** with **capped Kelly** fractions and caps:
   - Low risk: smaller per‑bet caps (singles ≈ 6%, parlays ≈ 2%), parlay pool ≤ 20% bankroll, Kelly caps (≈3% / 1%).
   - Higher risk: larger caps (15% / 6%), parlay pool ≤ 50%, Kelly caps (10% / 4%).
   - Limits on **max distinct games** and **max total picks**.

> Parlays assume independence as a rough approximation; caps keep exposure in check.

---

## Troubleshooting

- **ValueError during training**: A file is likely missing required columns — it will now be **skipped** automatically. You’ll see `[train_model] Skipping <file>` in the logs.
- **“KeyError: column not found” during recommendation**: The recommender now auto‑adds only features needed by the trained models; ensure the fixture file uses the same exact column names.
- **Too few classes for moneyline**: If your labeled data contains only one of {H, D, A} in a given subset, the moneyline model is skipped; totals/AH will still train.

---

## Safety & Responsibility

This code is for educational/informational purposes. Betting involves risk. Use at your own discretion and never stake money you cannot afford to lose.
