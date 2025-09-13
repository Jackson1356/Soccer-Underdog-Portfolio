# Underdog Advisor: Mispriced Markets Finder

An end-to-end toolkit to (1) **train on your historical data** in `./data`, (2) **recommend tonight’s bets** from a fixture file, and (3) **visualize** singles and 2-leg parlays in a lightweight dashboard compatible with your tracker.

---

## Quick Start

```bash
# 1) Train all models from your past data
python train_model.py --data_dir ./data --models_dir models --do_backtest --backtest_k 3

# 2) Recommend for tonight (use your fixture CSV/XLSX with the same columns)
python recommend.py --fixture_file tonight.csv --models_dir models --bankroll 1000

# 3) Open the dashboard and upload outputs
# - recommendations_tonight.csv
# - parlays_tonight.csv
# - optionally backtests/backtest_moneyline.csv for ROI
open advisor_dashboard.html   # (or double-click)
```

Outputs:
- `models/` → `moneyline.pkl`, `over25.pkl`, `ah_home_ev.pkl`, `ah_away_ev.pkl`, `meta.json`
- `recommendations_tonight.csv` → all **single bets** across markets with EV and stakes
- `parlays_tonight.csv` → top **2-leg parlays** (EV under independence)
- `backtests/backtest_moneyline.csv` → simple historical moneyline backtest (optional)

---

## Data Columns (robust to missing fields)

We automatically harmonize common variants. Useful columns:
`['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HS','AS','HST','AST','B365H','B365D','B365A','B365C>2.5','B365C<2.5','AHh','B365AHH','B365AHA']`

All scripts tolerate absent odds/metrics; missing markets are skipped.

---

## Modeling Details

### Moneyline (1X2) Outcome Probabilities
We fit a **multinomial logistic regression** to estimate
\( P(Y=\text{H}\mid x),\;P(Y=\text{D}\mid x),\;P(Y=\text{A}\mid x) \)
with features combining:
- Market prices (Bet365 odds),
- Asian handicap line & odds (if present),
- Simple form features (rolling last-5 goals for/against and average points),
- League and team indicators.

Let \(x\) be features for a match. For class \(k \in \{H,D,A\}\),
\[
P(Y=k\mid x) = \frac{\exp(\beta_k^\top x)}{\sum_{j}\exp(\beta_j^\top x)}.
\]

**EV for side \(k\)** with decimal odds \(o_k\):
\[
\operatorname{EV}_k = P(Y=k\mid x)\,(o_k-1) - \bigl(1-P(Y=k\mid x)\bigr).
\]

We keep the best side per game and use EV for ranking.

### Totals (2.5) Probability
Define \(Z=\mathbb{1}\{ \text{FTHG}+\text{FTAG} > 2.5\}\). We train a **Gradient Boosting** classifier to estimate \(p_{\text{over}}=P(Z=1\mid x)\). Then:
\[
\operatorname{EV}_{\text{Over}} = p_{\text{over}}\,(o_{\text{over}}-1)-(1-p_{\text{over}}),\qquad
\operatorname{EV}_{\text{Under}} = (1-p_{\text{over}})\,(o_{\text{under}}-1)-p_{\text{over}}.
\]
Pick the larger EV (if positive).

### Asian Handicap (AH) Expected Profit (EV) Regressors
For AH we **directly learn EV** per stake for Home and Away, handling quarter-lines.
Let \(g=\text{FTHG}-\text{FTAG}\) (home goal difference), home line \(h\) (e.g., -0.25), and decimal odds \(o\).
We compute the realized profit \(\pi(g,h,o)\) per unit stake using half-stakes for quarter-lines:
- Split \(h\) into half-steps (e.g., \(-0.25 \to \{-0.5,0\}\), \(+0.75\to\{+0.5,+1.0\}\)).
- For each part \(\tilde h\), the adjusted difference \(g+\tilde h\) gives **win** \((o-1)\), **push** 0, **loss** -1.
- Sum halves for the total per-unit profit.

We train two regressors:
- \(f_H(x) \approx \mathbb{E}[\pi(g,h,o_H)\mid x]\) for Home AH @ \(o_H\)
- \(f_A(x) \approx \mathbb{E}[\pi(-g,-h,o_A)\mid x]\) for Away AH @ \(o_A\)

These predict EV directly (units per 1 stake). For staking we convert to an implied probability \(p\) via
\[ \text{EV} = p(o-1)-(1-p)\ \Rightarrow\ p=\frac{\text{EV}+1}{o}. \]

---

## Betting Terms (brief)

- **Moneyline (1X2):** Bet on Home (1), Draw (X), or Away (2).
- **Asian Handicap (AH):** Applies a goal handicap to even teams. Quarter-lines (±0.25, ±0.75) split your stake across two half-lines (e.g., +0.25 = half on +0.0, half on +0.5) enabling half-wins / half-losses and pushes.
- **Totals 2.5:** Over/Under the total goals threshold (2.5).
- **Decimal Odds \(o\):** Return on a 1-unit stake is \(o\) if win, else 0; net profit is \(o-1\) on win and \(-1\) on loss.
- **Expected Value (EV):** \(\mathbb{E}[\text{net profit}]\) per unit stake.
- **Kelly fraction:** \(f^*=\frac{b p - q}{b}\) with \(b=o-1,\ q=1-p\). We cap \(f^*\) for risk control.

---

## Strategy We Implement

1. **Surface mispriced sides** by ranking **EV** across three markets per game:
   - Moneyline: keep the best of H/D/A.
   - Totals 2.5: choose Over vs Under by EV.
   - Asian Handicap: predict EV directly for Home & Away; keep the better side.
2. **Pick 3–5 singles** with **EV > 0** (ideally > 0.03) and **sane odds** (avoid extremes like > 10x unless you accept variance).
3. **Stake via capped Kelly:**
   - Singles caps: 3% / 7% / 12% (conservative/balanced/risky).
   - Parlays: **tiny caps** (1%, 3%, 6%) due to correlation & compounding risk.
4. **Parlays (2-leg “strings”):** Combine top singles under an **independence approximation**: multiply probabilities and odds to compute parlay EV. Favor parlays only when **both legs** have clear edge.
5. **Skip/Filter:** Avoid picks created by thin data (missing key odds), sudden line chaos, or contradictory market signals.

**Caveats:** Market probabilities are correlated, and prices move with news. Treat parlay independence as a heuristic. Consider excluding intra-league correlated legs or same-team combos.

---

## Files You’ll Use

- **train_model.py** — trains all models; optional simple moneyline backtest.
- **recommend.py** — takes a *separate fixture file* (tonight) and emits singles + 2-leg parlays with stakes.
- **advisor_dashboard.html** — upload `recommendations_tonight.csv`, `parlays_tonight.csv`, and a daily ROI CSV to visualize.
- **models/meta.json** — records feature sets and training metrics for reproducibility.

---

## Extending

- Add drift features from **odds movement** if you store snapshots.
- Calibrate probabilities (Platt/Isotonic) for tighter EV.
- Enrich totals with team Poisson baselines.
- Add constraints (e.g., max total daily stake, exclude same-match legs in parlays).
