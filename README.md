# Underdog Advisor for Soccer Betting — (Unified Betting Portfolios)

**Goal:** Learn from **opening** and **closing** odds/lines to spot value (esp. underdogs & mispricings), then build **unified portfolios** (mix of **singles** and **2‑leg parlays**) at two risk levels: **Low** and **High**.


---

## Data

Data Source: https://football-data.co.uk/

We used opening and closing odds of [Bet365](https://www.bet365.com/).

### Notation (symbols used in formulas)

- Moneyline open: $O_H$ (B365H), $O_D$ (B365D), $O_A$ (B365A)  
- Moneyline close: $C_H$ (B365CH), $C_D$ (B365CD), $C_A$ (B365CA)  
- Totals open: $O_{>2.5}$ (B365>2.5), $O_{<2.5}$ (B365<2.5)  
- Totals close: $C_{>2.5}$ (B365C>2.5), $C_{<2.5}$ (B365C<2.5)  
- AH: open line $h_{\text{open}}$ (AHh), close line $h_{\text{close}}$ (AHCh); home odds $O_{\text{AH,H}}$ (B365AHH), away odds $O_{\text{AH,A}}$ (B365AHA); closing $C_{\text{AH,H}}$ (B365CAHH), $C_{\text{AH,A}}$ (B365CAHA)

---

## Engineered features

To avoid ambiguous subscripts, we introduce simple aliases used only in the formulas below:

$$
\begin{aligned}
& \text{Let } O_H,O_D,O_A \text{ be OPEN 1X2 odds; } C_H,C_D,C_A \text{ be CLOSING 1X2 odds}.\\
& \text{Let } O_O,O_U \text{ be OPEN odds for Over/Under 2.5; } C_O,C_U \text{ be CLOSING odds for Over/Under 2.5}.\\
& \text{Let } h_o,h_c \text{ be AH open/close lines; } O_H^{AH},O_A^{AH},C_H^{AH},C_A^{AH} \text{ be AH odds}.
\end{aligned}
$$

### 1) 1X2 gaps & changes

$$
\mathrm{ml\_{open\_gap}} = O_A - O_H
$$

$$
\mathrm{ml\_{close\_gap}} = C_A - C_H
$$

$$
\mathrm{ml\_{gap\_change}} = \mathrm{ml\_{close\_gap}} - \mathrm{ml\_{open\_gap}}
$$

$$
\mathrm{ml\_{open\_min}} = \min\{O_H,O_D,O_A\}, \quad
\mathrm{ml\_{open\_max}} = \max\{O_H,O_D,O_A\}
$$

$$
\mathrm{ml\_{close\_min}} = \min\{C_H,C_D,C_A\}, \quad
\mathrm{ml\_{close\_max}} = \max\{C_H,C_D,C_A\}
$$

$$
\mathrm{ml\_{delta\_H}}=C_H-O_H, \quad \mathrm{ml\_{reld\_H}}= \frac{C_H}{O_H}-1
$$

$$
\mathrm{ml\_{delta\_D}}=C_D-O_D, \quad \mathrm{ml\_{reld\_D}}= \frac{C_D}{O_D}-1
$$

$$
\mathrm{ml\_{delta\_A}}=C_A-O_A, \quad \mathrm{ml\_{reld\_A}}= \frac{C_A}{O_A}-1
$$

### 2) Totals movement (Over/Under 2.5)

$$
\mathrm{tot\_{delta\_{over}}} = C_O - O_O, \qquad
\mathrm{tot\_{delta\_{under}}} = C_U - O_U
$$

$$
\mathrm{tot\_{reld\_{over}}} = \frac{C_O}{O_O} - 1, \qquad
\mathrm{tot\_{reld\_{under}}} = \frac{C_U}{O_U} - 1
$$

### 3) Asian Handicap movement

$$
\mathrm{ah\_{line\_change}} = h_c - h_o
$$

$$
\mathrm{ah\_{delta\_{home\_odds}}}=C_H^{AH}-O_H^{AH}, \qquad
\mathrm{ah\_{reld\_{home\_odds}}}= \frac{C_H^{AH}}{O_H^{AH}}-1
$$

$$
\mathrm{ah\_{delta\_{away\_odds}}}=C_A^{AH}-O_A^{AH}, \qquad
\mathrm{ah\_{reld\_{away\_odds}}}= \frac{C_A^{AH}}{O_A^{AH}}-1
$$

---

## Models

- **Moneyline (1X2)** — Multinomial Logistic Regression (softmax).  
- **Totals (Over 2.5)** — Gradient Boosting Classifier.  
- **AH EV (Home/Away)** — Gradient Boosting Regressors (predict per‑unit EV).

All numeric features are imputed and scaled.

---

## Some Math

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

Here is an example how we find an underdog game [EPL: Bournemouth vs Manchester City](underdog_example.md)

---

## Troubleshooting

- Empty portfolios → all $\mathrm{EV}\le0$ or risk caps filtered them. Check `recommend_debug.json`.
- Missing columns → the recommender lists them explicitly.
- No models → run training; verify `models/` has pickles and `meta.json`.

---
