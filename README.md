# Underdog Advisor — Opening vs Closing Odds (Unified Betting Portfolios)

**Goal:** Learn from **opening** and **closing** odds/lines to spot value (esp. underdogs & mispricings), then build **unified portfolios** (mix of **singles** and **2‑leg parlays**) at two risk levels: **Low** and **High**.

> ⚠️ Research project. Betting involves risk. Use responsibly.

---

## Data requirements (exact column names)

Every **training** file must include **all** below (else it’s skipped). The **fixture** for recommending must also include them.

**1X2 (Moneyline)**  
Open: `B365H`, `B365D`, `B365A` • Close: `B365CH`, `B365CD`, `B365CA`

**Totals (2.5 goals)**  
Open: `B365>2.5`, `B365<2.5` • Close: `B365C>2.5`, `B365C<2.5`

**Asian Handicap**  
Open: `AHh`, `B365AHH`, `B365AHA` • Close: `AHCh`, `B365CAHH`, `B365CAHA`

**Labels (training only)**  
Moneyline: `FTR ∈ {H,D,A}` • Totals & AH: `FTHG`, `FTAG`

`HomeTeam` / `AwayTeam` are optional (display only).

---

## Notation (symbols used in math below)

We use short symbols; the **code still uses original column names**.

- Moneyline open: O_H(B365H), O_D(B365D), O_A(B365A)  
- Moneyline close: C_H(B365CH), C_D(B365CD), C_A(B365CA)  
- Totals open: O_>2.5(B365>2.5), O_<2.5(B365<2.5)  
- Totals close: C_>2.5(B365C>2.5), C_<2.5(B365C<2.5)  
- AH: h_open(AHh), h_close(AHCh); O_AH,H(B365AHH), O_AH,A(B365AHA); C_AH,H(B365CAHH), C_AH,A(B365CAHA)

---

## Engineered features (render‑safe code blocks)

### 1) 1X2 gaps & changes
```text
ml_open_gap   = O_A - O_H
ml_close_gap  = C_A - C_H
ml_gap_change = ml_close_gap - ml_open_gap

ml_open_min   = min(O_H, O_D, O_A)
ml_open_max   = max(O_H, O_D, O_A)
ml_close_min  = min(C_H, C_D, C_A)
ml_close_max  = max(C_H, C_D, C_A)

# per-side moves
ml_delta_H = C_H - O_H
ml_reld_H  = (C_H / O_H) - 1
ml_delta_D = C_D - O_D
ml_reld_D  = (C_D / O_D) - 1
ml_delta_A = C_A - O_A
ml_reld_A  = (C_A / O_A) - 1
```

### 2) Totals movement
```text
tot_delta_over  = C_>2.5 - O_>2.5
tot_delta_under = C_<2.5 - O_<2.5
tot_reld_over   = (C_>2.5 / O_>2.5) - 1
tot_reld_under  = (C_<2.5 / O_<2.5) - 1
```

### 3) Asian Handicap movement
```text
ah_line_change      = h_close - h_open
ah_delta_home_odds  = C_AH,H - O_AH,H
ah_reld_home_odds   = (C_AH,H / O_AH,H) - 1
ah_delta_away_odds  = C_AH,A - O_AH,A
ah_reld_away_odds   = (C_AH,A / O_AH,A) - 1
```

---

## Models

- **Moneyline (1X2)** — Multinomial Logistic Regression (softmax).  
- **Totals (Over 2.5)** — Gradient Boosting Classifier.  
- **AH EV (Home/Away)** — Gradient Boosting Regressors (predict per‑unit EV).

All numeric features are imputed and scaled.

---

## Math

### Implied probability (naïve)

Given decimal odds $o$,
$$
\hat{p}=\frac{1}{o}.
$$

### Moneyline (multinomial logistic)

For features $x$ and classes $k\in\{H,D,A\}$,
$$
P(y=k\mid x)=\frac{e^{w_k^\top x}}{\sum_{j\in\{H,D,A\}} e^{w_j^\top x}}.
$$

### Totals (binary)

$$
P(\text{Over}\mid x)=\sigma(f(x))=\frac{1}{1+e^{-f(x)}}.
$$

### Expected value (single bet)

With model probability $p$ and odds $o$,
$$
\mathrm{EV}=p\,(o-1)-(1-p).
$$

### Variance proxy (per \$1)

Let $\mu=\mathrm{EV}$, win payoff $o-1$, loss payoff $-1$:
$$
\sigma^2 = p\,(o-1-\mu)^2 + (1-p)\,(-1-\mu)^2.
$$

### Scoring & selection

Sharpe‑like score (keep $\mu>0$):
$$
\mathrm{score}=\frac{\mu}{\sigma+\varepsilon}.
$$

### Kelly stake (capped)

Let $b=o-1$, $q=1-p$,
$$
f^*=\frac{b\,p-q}{b}.
$$

### Parlays (2‑leg, independence)

$$
p_{\text{par}}=p_1p_2,\quad o_{\text{par}}=o_1o_2,\quad
\mathrm{EV}_{\text{par}}=p_{\text{par}}(o_{\text{par}}-1)-(1-p_{\text{par}}).
$$

### AH per‑unit profit (label)

Let $g=\mathrm{FTHG}-\mathrm{FTAG}$, line $h$, odds $o$.  
Quarter‑lines split the unit in half; profit:
$$
\mathrm{profit}(g,h,o)=
\begin{cases}
o-1, & \text{win},\\
0,   & \text{push},\\
-1,  & \text{loss},
\end{cases}
$$
with half‑wins / half‑losses for $h=\pm0.25,\pm0.75$.

---

## Usage

```bash
python train_model.py --data_dir ./data --models_dir models
python recommend.py --fixture_file tonight.csv --models_dir models --bankroll 1000 --max_games 5 --max_picks 8 --debug
```

Artifacts: `portfolio_low.csv`, `portfolio_high.csv`, `*_summary.csv`, (`--debug`) `recommend_debug.json`.

---

## Reading the portfolios

Columns: `Type`, `Market`, `Description`, `Odds`, `p`, `EV_per_unit`, `Stake`, `StakePct`.  
Low‑risk uses tighter caps; High‑risk allows larger stakes and more parlay exposure.

---

## Worked examples

Single (1X2): if $o=2.20$, $p=0.50$,
$$
\mathrm{EV}=0.5(2.20-1)-0.5=0.10.
$$

Parlay (2‑leg): $o_1=1.85$, $p_1=0.58$; $o_2=2.05$, $p_2=0.52$.
$$
p_{\text{par}}=0.3016,\quad o_{\text{par}}=3.7925,\quad
\mathrm{EV}_{\text{par}}\approx 0.143.
$$

---

## Troubleshooting

- Empty portfolios → all $\mathrm{EV}\le0$ or risk caps filtered them. Check `recommend_debug.json`.
- Missing columns → the recommender lists them explicitly.
- No models → run training; verify `models/` has pickles and `meta.json`.

---

## Ethics & Safety

Prefer **Low** risk mode and small stakes. Stop if it stops being fun.