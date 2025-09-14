# Underdog Advisor — Opening vs Closing Odds (Unified Betting Portfolios)

**Goal:** Learn from **opening** and **closing** odds/lines to spot value (especially underdogs and misleading prices), then build **unified bet portfolios** (mix of **singles** and **2‑leg parlays**) at two risk levels: **Low** and **High**.

> ⚠️ **Disclaimer:** This is a research/education project. Sports betting involves risk. Bet responsibly.

---

## TL;DR

- **Inputs**: Opening & closing odds/lines for **1X2**, **Over/Under 2.5**, and **Asian Handicap (AH)**.  
- **Models**:  
  - Moneyline (1X2): **Multinomial Logistic Regression**  
  - Totals (Over 2.5): **Gradient Boosting Classifier**  
  - AH EV (Home/Away): **Gradient Boosting Regressors**
- **Picks**: Rank by a Sharpe‑like ratio (**EV/σ**), apply **capped Kelly** stakes, combine **singles + 2‑leg parlays** with risk caps.  
- **Outputs**: `portfolio_low.csv`, `portfolio_high.csv` (+ one‑row summaries).  
- **Dashboard**: `advisor_dashboard.html` to visualize portfolios and show commands.

---

## Betting 101 (in 90 seconds)

- **Decimal odds** (e.g., 2.20): a \$1 stake returns \$2.20 if it wins (profit = \$1.20).  
- **1X2 / Moneyline**: outcomes `H` (home win), `D` (draw), `A` (away win).  
- **Totals (O/U 2.5 goals)**: bet on total goals **>2.5** (Over) or **<2.5** (Under).  
- **Asian Handicap (AH)**: spreads like `-0.25`, `+0.75`. Quarter‑lines split stakes across two half‑lines (e.g., `-0.25` = half on `0` and half on `-0.5`). A “push” returns your stake on exactly the handicap.  
- **Opening vs Closing odds**: **Opening** = posted earlier. **Closing (C)** = right before kickoff. **Price moves** Open→Close can signal information and value.

---

## Data Requirements (exact column names)

Every **training** file must include **all** of the following; otherwise it is **skipped**. The **recommender** fixture must also include them.

### 1X2 (Moneyline)
- **Open**: `B365H`, `B365D`, `B365A`  
- **Close**: `B365CH`, `B365CD`, `B365CA`

### Totals (2.5)
- **Open**: `B365>2.5`, `B365<2.5`  
- **Close**: `B365C>2.5`, `B365C<2.5`

### Asian Handicap
- **Open line & odds**: `AHh`, `B365AHH`, `B365AHA`  
- **Close line & odds**: `AHCh`, `B365CAHH`, `B365CAHA`

### Labels (training only)
- Moneyline: `FTR` ∈ {`H`,`D`,`A`}  
- Totals & AH: `FTHG`, `FTAG`

> `HomeTeam` / `AwayTeam` are optional (used for readable descriptions only).

---

## Intuition (why this can work)

- Book odds blend **bookmaker opinion** and **crowd money**. Mispricings—often on **underdogs**—do exist.  
- **Closing** odds are typically sharper than **opening**. The **movement** from Open→Close can carry useful information.  
- We **do not** use teams, leagues, or dates—this system learns purely from **market signals** (levels & moves), per your preference.  
- Portfolios combine **singles** (lower variance) with **2‑leg parlays** (higher variance) under **risk caps**.

---

## Features & Models

### Engineered features (from your columns)
- **1X2 gaps & changes**
  - \( \text{ml\_open\_gap} = \text{B365A} - \text{B365H} \)  
  - \( \text{ml\_close\_gap} = \text{B365CA} - \text{B365CH} \)  
  - \( \text{ml\_gap\_change} = \text{ml\_close\_gap} - \text{ml\_open\_gap} \)  
  - `ml_open_min/max`, `ml_close_min/max`  
  - Per‑side deltas/ratios:  
    - \( \text{ml\_delta\_H} = \text{B365CH} - \text{B365H} \) and \( \text{ml\_reld\_H} = \frac{\text{B365CH}}{\text{B365H}} - 1 \) (and similarly for `D`, `A`).
- **Totals movement**
  - \( \text{tot\_delta\_over} = \text{B365C>2.5} - \text{B365>2.5} \)  
  - \( \text{tot\_reld\_over} = \frac{\text{B365C>2.5}}{\text{B365>2.5}} - 1 \) (and similarly for Under).
- **AH movement**
  - \( \text{ah\_line\_change} = \text{AHCh} - \text{AHh} \)  
  - Odds deltas/ratios for home/away (e.g., \( \text{ah\_delta\_home\_odds} = \text{B365CAHH} - \text{B365AHH} \)).

### Algorithms
- **Moneyline (1X2)**: Multinomial **Logistic Regression** (softmax).  
- **Totals (Over 2.5)**: **Gradient Boosting Classifier**.  
- **AH EV (Home & Away)**: **Gradient Boosting Regressors** predicting **per‑unit profit**.

> Pipelines use **SimpleImputer** + **StandardScaler** for numeric features.

---

## Math (core formulas)

### Implied probability & overround
From decimal odds \( o \), naive implied probability:
\[
\hat{p} = \frac{1}{o}.
\]
(Books include margin; \(\sum \hat{p}_i > 1\). We primarily learn from **levels/moves** rather than explicit margin removal.)

### Moneyline (multinomial logistic)
For features \( x \) and classes \( k \in \{H, D, A\} \):
\[
P(y=k\,|\,x) = \frac{\exp(w_k^\top x)}{\sum_{j \in \{H,D,A\}} \exp(w_j^\top x)}.
\]

### Totals (binary classifier)
\[
P(\text{Over}\,|\,x) = \sigma(f(x)) = \frac{1}{1 + e^{-f(x)}}.
\]

### Expected value (single bet)
With probability \( p \) and decimal odds \( o \):
\[
\text{EV} = p\,(o-1) - (1-p).
\]

### Variance proxy (per \$1 bet)
Let \(\mu = \text{EV}\), win payoff \(o-1\), loss payoff \(-1\):
\[
\sigma^2 = p\,(o-1 - \mu)^2 + (1-p)\,(-1 - \mu)^2.
\]

### Scoring & selection
Rank by **Sharpe‑like** score:
\[
\text{score} = \frac{\mu}{\sigma + \varepsilon},
\]
keep \(\mu>0\), then allocate stakes subject to caps.

### Kelly (capped)
Let \( b=o-1 \), \( q=1-p \):
\[
f^* = \frac{b\,p - q}{b}.
\]
We cap \( f^* \) (per bet and by market) and apply smaller caps for **Low** risk mode.

### Parlay (2‑leg)
Assuming independence:
\[
p_{\text{parlay}} = p_1 p_2,\quad o_{\text{parlay}} = o_1 o_2,\quad
\text{EV}_{\text{parlay}} = p_{\text{parlay}}(o_{\text{parlay}}-1) - (1-p_{\text{parlay}}).
\]

### AH per‑unit profit (label)
For goal diff \( g = \text{FTHG}-\text{FTAG} \), line \( h \) (home perspective) and odds \( o \):  
Quarter‑lines split the unit into halves; profit:
\[
\text{profit}(g,h,o)=
\begin{cases}
o - 1 & \text{if win after handicap},\\
0     & \text{if push},\\
-1    & \text{if loss},
\end{cases}
\]
with half‑wins / half‑losses for quarter‑lines (e.g., \(h=\pm 0.25, \pm 0.75\)).

---

## Installation

```bash
pip install -r requirements.txt  # scikit-learn, pandas, numpy, etc.
```

---

## Usage

### 1) Train (skips files missing any required columns)
```bash
python train_model.py --data_dir ./data --models_dir models
```

### 2) Recommend unified portfolios
```bash
python recommend.py --fixture_file tonight.csv --models_dir models --bankroll 1000 --max_games 5 --max_picks 8 --debug
```
Artifacts:
- `portfolio_low.csv`, `portfolio_high.csv`
- `portfolio_low_summary.csv`, `portfolio_high_summary.csv`
- (with `--debug`) `recommend_debug.json` — coverage & candidate counts

### 3) Visualize
Open `advisor_dashboard.html`, upload the two portfolio CSVs (and optional summaries).

---

## How to read the portfolio files

Each row:
- `Type` — `Single` or `Parlay` (2‑leg)  
- `Market` — `Moneyline` | `Totals2.5` | `AsianHandicap`  
- `Description` — human‑readable matchup & selection  
- `Odds` — decimal odds used (prefer **closing**, fallback **opening**)  
- `p` — model probability (parlays multiply leg probs)  
- `EV_per_unit` — expected profit per \$1 stake  
- `Stake` — dollars to place  
- `StakePct` — stake/bankroll

The one‑row `*_summary.csv` files show bankroll, total stake, expected profit, rough risk (std), and number of picks.

---

## Worked Examples

**Single (Moneyline)**: If odds \(o=2.20\) and \(p=0.50\), then
\[
\text{EV} = 0.5\times(2.20-1) - 0.5 = 0.10.
\]
With a capped Kelly fraction \(f=0.04\) and bankroll \$1000 → stake \$40 → expected profit ≈ \$4.

**Parlay (2‑leg)**: Leg A \(o_1=1.85, p_1=0.58\); Leg B \(o_2=2.05, p_2=0.52\).  
\[
p_{\text{par}} = 0.58\times 0.52 = 0.3016,\quad o_{\text{par}} = 1.85\times 2.05 = 3.7925.
\]
\[
\text{EV}_{\text{par}} \approx 0.3016\times(3.7925-1) - 0.6984 \approx 0.143.
\]
Parlays have higher variance ⇒ the allocator caps total parlay exposure.

---

## Troubleshooting

- **Empty portfolios**: likely all \(\text{EV} \le 0\) or risk caps filtered them out. Check `recommend_debug.json`. Consider raising `--top_k_singles/--top_k_parlays`, `--max_picks`, or (carefully) relaxing caps.  
- **Fixture missing columns**: the recommender fails fast with a clear list of missing names.  
- **No models loaded**: ensure you ran training and that `models/` exists with `*.pkl` and `meta.json`.

---

## Roadmap

- Moneyline **class weighting** / probability **calibration**.  
- Overround‑adjusted implied probability movement features.  
- CV without dates (your preference) for more robust metrics.  
- Correlation‑aware parlay adjustments.  
- Mode‑specific min‑Sharpe thresholds.

---

## Ethics & Safety

Use this repo responsibly. For real money, start with **Low** risk mode and small stakes. Stop if it stops being fun.