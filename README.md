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

## Notation

We map your column names to compact symbols (math only; code uses raw names):

- **1X2 (open/close)**:  
  $O_H=\texttt{B365H}$, $O_D=\texttt{B365D}$, $O_A=\texttt{B365A}$;  
  $C_H=\texttt{B365CH}$, $C_D=\texttt{B365CD}$, $C_A=\texttt{B365CA}$.
- **Totals (open/close)**:  
  $O_{>2.5}=\texttt{B365>2.5}$, $O_{<2.5}=\texttt{B365<2.5}$;  
  $C_{>2.5}=\texttt{B365C>2.5}$, $C_{<2.5}=\texttt{B365C<2.5}$.
- **Asian Handicap (open/close)**:  
  $h_{\text{open}}=\texttt{AHh}$, $h_{\text{close}}=\texttt{AHCh}$;  
  $O_{\text{AH,H}}=\texttt{B365AHH}$, $O_{\text{AH,A}}=\texttt{B365AHA}$;  
  $C_{\text{AH,H}}=\texttt{B365CAHH}$, $C_{\text{AH,A}}=\texttt{B365CAHA}$.

---

## Engineered features

### 1) 1X2 gaps & changes

$$\mathrm{ml\_open\_gap} = O_A - O_H$$
$$\mathrm{ml\_close\_gap} = C_A - C_H$$
$$\mathrm{ml\_gap\_change} = \mathrm{ml\_close\_gap} - \mathrm{ml\_open\_gap}$$

Min/Max:
- $\mathrm{ml\_open\_min} = \min\{O_H,O_D,O_A\}$,  $\mathrm{ml\_open\_max} = \max\{O_H,O_D,O_A\}$  
- $\mathrm{ml\_close\_min} = \min\{C_H,C_D,C_A\}$, $\mathrm{ml\_close\_max} = \max\{C_H,C_D,C_A\}$

Per‑side moves:
- $\mathrm{ml\_delta\_H}=C_H-O_H$,  $\mathrm{ml\_reld\_H}= \dfrac{C_H}{O_H}-1$  
- $\mathrm{ml\_delta\_D}=C_D-O_D$,  $\mathrm{ml\_reld\_D}= \dfrac{C_D}{O_D}-1$  
- $\mathrm{ml\_delta\_A}=C_A-O_A$,  $\mathrm{ml\_reld\_A}= \dfrac{C_A}{O_A}-1$

### 2) Totals movement

$$\mathrm{tot\_delta\_over} = C_{>2.5} - O_{>2.5},\qquad
\mathrm{tot\_delta\_under} = C_{<2.5} - O_{<2.5}$$

$$\mathrm{tot\_reld\_over} = \frac{C_{>2.5}}{O_{>2.5}} - 1,\qquad
\mathrm{tot\_reld\_under} = \frac{C_{<2.5}}{O_{<2.5}} - 1$$

### 3) Asian Handicap movement

$$\mathrm{ah\_line\_change} = h_{\text{close}} - h_{\text{open}}$$

Home/Away odds moves:
- $\mathrm{ah\_delta\_home\_odds}=C_{\text{AH,H}}-O_{\text{AH,H}}$,  
  $\mathrm{ah\_reld\_home\_odds}=\dfrac{C_{\text{AH,H}}}{O_{\text{AH,H}}}-1$  
- $\mathrm{ah\_delta\_away\_odds}=C_{\text{AH,A}}-O_{\text{AH,A}}$,  
  $\mathrm{ah\_reld\_away\_odds}=\dfrac{C_{\text{AH,A}}}{O_{\text{AH,A}}}-1$

---

## Models

- **Moneyline (1X2)** — Multinomial Logistic Regression (softmax).  
- **Totals (Over 2.5)** — Gradient Boosting Classifier.  
- **Asian Handicap EV (Home/Away)** — Gradient Boosting Regressors (predict per‑unit EV).

All numeric features are imputed and scaled.

---

## Math

### Implied probability (naïve)
Given decimal odds $o$,
$$ \hat{p}=\frac{1}{o}. $$

### Moneyline (multinomial logistic)
For features $x$ and classes $k\in\{H,D,A\}$,
$$ P(y=k\mid x)=\frac{e^{w_k^\top x}}{\sum_{j\in\{H,D,A\}} e^{w_j^\top x}}. $$

### Totals (binary)
$$ P(\text{Over}\mid x)=\sigma(f(x))=\frac{1}{1+e^{-f(x)}}. $$

### Expected value (single bet)
With model probability $p$ and odds $o$,
$$ \mathrm{EV}=p\,(o-1)-(1-p). $$

### Variance proxy (per \$1)
Let $\mu=\mathrm{EV}$, win payoff $o-1$, loss payoff $-1$:
$$ \sigma^2 = p\,(o-1-\mu)^2 + (1-p)\,(-1-\mu)^2. $$

### Scoring & selection
Sharpe‑like score (keep $\mu>0$):
$$ \mathrm{score}=\frac{\mu}{\sigma+\varepsilon}. $$

### Kelly stake (capped)
Let $b=o-1$, $q=1-p$,
$$ f^*=\frac{b\,p-q}{b}. $$

### Parlays (2‑leg, independence)
$$ p_{\text{par}}=p_1p_2,\quad o_{\text{par}}=o_1o_2,\quad
\mathrm{EV}_{\text{par}}=p_{\text{par}}(o_{\text{par}}-1)-(1-p_{\text{par}}). $$

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
$$ \mathrm{EV}=0.5(2.20-1)-0.5=0.10. $$

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