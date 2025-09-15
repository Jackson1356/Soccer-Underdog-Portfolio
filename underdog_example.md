# Worked Example — Identifying and Staking an **Underdog**

**Bournemouth vs Manchester City**

**Closing 1X2 odds** (`B365CH`, `B365CD`, `B365CA`):  
Home (Bournemouth) = **4.40**, Draw = **3.75**, Away (Man City) = **1.80**

**Opening 1X2 odds** (`B365H`, `B365D`, `B365A`) for context:  
Home = **5.00**, Draw = **3.90**, Away = **1.65**

---

## Why this is an underdog spot

1) **Underdog by definition:** the home win price (4.40) is much higher than the away price (1.80).  
2) **Open → Close move favors the dog:** Home shortened **5.00 → 4.40**; Favorite drifted **1.65 → 1.80**. This is the kind of signal we learn from.

---

## Naïve implied probabilities (from closing odds)

$$
\hat p_H = \frac{1}{4.40} \approx 0.227, \qquad
\hat p_D = \frac{1}{3.75} \approx 0.267, \qquad
\hat p_A = \frac{1}{1.80} \approx 0.556
$$

(Sum \(>1\) due to overround — expected.)

---

## Model probabilities (illustrative)

$$
p_H = 0.27, \qquad p_D = 0.26, \qquad p_A = 0.47
$$

Model is more optimistic on the home side than \(1/o\) suggests (\(0.27\) vs \(0.227\)).

---

## Expected value (EV) of **Home** moneyline at 4.40

$$
\mathrm{EV} = p\,(o-1) - (1-p)
= 0.27 \times (4.40 - 1) - 0.73
= 0.27 \times 3.40 - 0.73
= 0.918 - 0.73
= \mathbf{0.188}
$$

So the expected profit is **\$0.188 per \$1 stake** (positive EV).

---

## Risk proxy & Sharpe-like score (per \$1)

Win payoff \(= o-1 = 3.40\); loss payoff \(= -1\); \(\mu=\mathrm{EV}=0.188\).

$$
\sigma^2 = p\,(3.40-\mu)^2 + (1-p)\,(-1-\mu)^2
\approx 0.27\times 3.212^2 + 0.73\times 1.188^2 \approx 3.816
\quad\Rightarrow\quad \sigma \approx 1.954
$$

$$
\text{score} = \frac{\mu}{\sigma} \approx \frac{0.188}{1.954} \approx \mathbf{0.096}
$$

---

## Kelly fraction (capped) & example stakes

$$
f^*=\frac{(o-1)\,p-(1-p)}{o-1}=\frac{\mu}{o-1}=\frac{0.188}{3.40}\approx \mathbf{0.055}
$$

With a \$1,000 bankroll:
- **Low‑risk mode** caps singles around **3%** → stake ≈ **\$30**.  
- **High‑risk mode** allows more; \(f^*\approx 5.5\%\) fits → stake ≈ **\$55**.

---

## Takeaway

This match qualifies as an **underdog opportunity** because (a) the home side is clearly priced as the underdog (**4.40 vs 1.80**), (b) **market movement** (Open→Close) favored the dog, and (c) the model’s probability for the home win **exceeds** its naïve implied probability enough to yield **positive EV**.