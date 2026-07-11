# Volatility Risk Premium Harvesting — A Comparative Study

A research project replicating and comparing three implementations of VRP
harvesting on SPX, with explicit tail-risk accounting. Framed as a
comparative study of known strategies, not a novel-alpha claim. Primary
audience: readers evaluating quantitative research work.

## Abstract

The study implements (A) a dollar-neutral VIX-futures calendar spread
with a 5-trading-day pre-expiry roll, (B) systematic SPX put-writing —
the published CBOE PUT index plus a Black-Scholes synthetic replication
and a put-spread variant — and (C) a VRP-signal-gated version of B, then
subjects every construction to bootstrap, VaR/ES, and an October-1987
stress test. Training window 2013–2018, test window 2019–2024, no
test-window tuning.

Headline results after a full engine audit (see the revision note):

- **Strategy A (short front / long second)** earns test Sharpe **+0.88**
  (train +0.07) with a ~20% max drawdown and a −29.5% single-day 1987
  stress loss. The carry is real but tail-dominated and extremely
  window-dependent.
- **Strategy B put-spread** (−0.30Δ short / −0.10Δ long) is the most
  robust construction: test Sharpe **+0.54**, max drawdown −9%,
  full-sample probabilistic Sharpe ratio 0.997, 1987 stress loss capped
  at −3.3% of capital.
- **The VRP gate (Strategy C) and all three tail-risk overlays fail to
  add out-of-sample value** once look-ahead bugs are removed. The
  train-optimal gate underperforms the ungated spread out of sample,
  and the VIX regime filter — which looked like a large improvement
  under a one-day signal leak — destroys performance when applied
  tradeably.

The strongest honest claim this study supports: **a hedged put-spread
writer captures a modest, statistically credible premium; everything
fancier that was tried failed to improve it out of sample.**

## Revision note (engine audit)

An earlier revision of this study reported materially different
results. A code audit found five defects, each fixed in the current
engine; the git history preserves the originals. In severity order:

1. **VX contract-splice PnL (Strategy A).** The continuous front/second
   series switched contracts at expiry, and daily PnL was computed by
   differencing the spliced series — booking the front/second calendar
   gap as phantom PnL once a month. In contango this systematically
   penalized the short-front direction. The corrected engine computes
   every daily return within a single contract and rolls the held pair
   5 trading days before expiry. **This inverted the study's original
   Phase 1 conclusion**: the spec's short-front direction, previously
   reported as losing 12–15%/yr, is the profitable one; the "flipped"
   long-front direction, previously celebrated, loses.
2. **Regime-filter look-ahead (Overlay 1).** The filter zeroed day-t
   returns using day-t closing values — an exit that would require
   trading at the previous close. Because the first day VIX crosses 30
   is systematically among the worst days for a short-vol book, the
   leak deleted exactly the crash days and accounted for more than the
   overlay's entire published improvement (test Sharpe +1.03 with the
   leak, +0.26 without).
3. **Missing contracts from holiday expiries.** The VX settlement rule
   ignored exchange holidays, so four mid-sample contracts (Mar 2014,
   Mar 2019, Mar 2022, Jun 2024) failed to download and the series
   silently promoted the next month to front. Early 2013 was also
   missing (the CDN publishes Settle=0 with the real mark in Close
   before 2013-05-20). Both fixed; the loader now hard-errors on any
   in-window gap.
4. **Tail-hedge double-charge (Overlay 3).** The hedge debit was
   subtracted on top of marks that already embedded it, so each hedge
   cost twice its premium. Overlay 3 is still a net drag, but roughly
   half the originally reported size.
5. **Same-day VRP gating (Strategy C).** The gate consumed the VRP of
   the month-open day itself (knowable only at that day's close) while
   entering at that same close. It now gates on the prior month-end
   value, as the methodology always claimed.

Smaller corrections: option transaction costs are charged on both legs
of the spread (gross premium, not net); the 1987 stress prices the
delta-targeted strikes the strategy actually trades (the old hardcoded
K=95/85 overstated the spread's stress loss ~2.6×); PUT-index alpha is
regressed on total-return SPX with a T-bill risk-free rate (price-only
SPX flattered alpha by ~2%/yr); Sortino uses the standard full-sample
downside deviation; crisis-window tables report window returns instead
of annualizing 20-day episodes.

## Strategies

- **Strategy A — VIX Term-Structure Carry.** Dollar-neutral short
  front-month VX, long second-month VX, rolled 5 trading days before
  expiry. Captures `(r_second − r_front) / 2` per day on gross capital.
- **Strategy B — Systematic Put-Writing.** Monthly −0.30Δ SPX puts,
  cash-secured; benchmarked against the CBOE PUT index. Spread variant
  buys a −0.10Δ wing.
- **Strategy C — Conditional VRP Harvester.** Strategy B, gated on the
  prior month-end `VRP_t = IV_t − RV_t ≥ threshold`.

Overlays (applied to C in Phase 4): VIX regime filter, realized-vol
position scaling, tail-hedge spend.

## Literature

- Bakshi & Kapadia (2003). Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium.
- Carr & Wu (2009). Variance Risk Premiums.
- Alexander, Korovilas (2013). Understanding ETNs on VIX Futures.
- Dew-Becker, Giglio, Le, Rodriguez (2017). The Price of Variance Risk.
- Israelov & Nielsen (2015). Covered Calls Uncovered (AQR).
- Bondarenko (2014). Why Are Put Options So Expensive?
- Bailey & López de Prado (2012). The Sharpe Ratio Efficient Frontier
  (probabilistic Sharpe ratio).
- Politis & Romano (1994). The Stationary Bootstrap.

## Data

- SPX, VIX spot: Yahoo Finance (`^GSPC`, `^VIX`); `^SP500TR` and `^IRX`
  for total-return benchmark and risk-free rate.
- VIX futures (VX) settlements: CBOE CDN per-contract historical CSVs
  at `cdn.cboe.com/data/us/futures/market_statistics/historical_data/`.
  Coverage 2013-present; pre-2013 contracts return HTTP 403. Expiries
  are holiday-adjusted (CFE rule); early-2013 files carry the mark in
  Close with Settle=0.
- CBOE benchmark indices (PUT, BXM): CBOE daily CSVs.

## Methodology notes

- Train 2013-01-01 → 2018-12-31. Test 2019-01-01 → 2024-12-31.
- The original spec targeted 2006 onward with train 2006–2016 / test
  2017–2024, which places Feb 2018 and Mar 2020 in the test set. CBOE
  per-contract VX history begins in 2013, forcing the reduction — and
  **moving Feb 2018 (Volmageddon) into the training window**, a
  deviation from the spec's stated requirement that the test set
  contain it. The test window retains COVID 2020 and the 2022 bear;
  the train window retains 2015 and Feb 2018.
- No test-window parameter tuning. Signals computed through day t
  trade no earlier than the close of day t; anything derived from day
  t's close applies to day t+1's return.
- Transaction costs: 1 bp per leg per VX roll event (4 legs trade per
  roll). Options: 5% of gross premium round-trip (both legs of a
  spread), half at open, half at close. Sensitivity sweep 1–30 bps for
  Strategy A below.
- Annualization: 252 trading days. Sharpe = geometric annualized
  return / annualized vol — slightly more conservative than the
  arithmetic convention (mean×252 / std×√252) for volatile series;
  applied uniformly, including to bootstrap paths.
- Bootstrap and VaR/ES in Phase 5 use the **full 2013–2024 sample**
  (labelled full-sample, not train or test).

## Reproduce

```bash
pip install -e '.[dev]'
python scripts/run_strategy_a.py
python scripts/sanity_vxx.py
python scripts/run_strategy_a_sweep.py
python scripts/run_strategy_b_putindex.py
python scripts/run_strategy_b_synthetic.py
python scripts/run_strategy_b_spread.py
python scripts/run_strategy_c.py
python scripts/run_strategy_c_sensitivity.py
python scripts/run_strategy_c_overlays.py
python scripts/run_tail_risk.py
```

Outputs land in `reports/` (gitignored — regenerate from source).

## Sanity-check gate

The daily-return correlation between Strategy A (short-front) and
`−VXX` is required to fall in `[0.5, 0.9]` — the splice-free
held-contract construction shares its dominant leg with −VXX, so the
correlation runs higher than the loosely-related shape trade the old
engine produced. Current value: **0.777** (in-band). A negative or
sub-0.3 correlation indicates a sign bug in the VX PnL computation or
a data-ingestion issue.

## Phase 1 results — Strategy A

### Spec-direction baseline (short front / long second, 1 bp per leg)

| window | Sharpe | ann. return | ann. vol | max DD | DD duration | skew |
|---|---|---|---|---|---|---|
| train (2013-2018) | +0.07 | +1.0% | 13.3% | −22.9% | 354 days | −3.4 |
| test (2019-2024)  | +0.88 | +11.6% | 13.2% | −19.7% | 362 days | −1.5 |

The spec's short-front calendar is a carry trade: the front month
decays proportionally faster than the second in contango, and the
short-front leg collects that differential. The train window is nearly
flat because Feb 2018 (front VX +113% in a day) and the 2015 spike
land inside it; the test window's steep contango regimes (2019,
2021, 2023–24) deliver the +0.88. **The 12-year gap between the two
windows is the honest headline: this trade's realized Sharpe is
regime-dependent to the point that six years of data can show either
~0 or ~0.9.**

### Direction comparison

| variant | train Sharpe | test Sharpe | train ret | test ret | train MDD | test MDD |
|---|---|---|---|---|---|---|
| **short front / long second (spec)** | **+0.07** | **+0.88** | +1.0% | +11.6% | −23% | −20% |
| long front / short second (flipped) | −0.24 | −0.94 | −3.2% | −12.4% | −35% | −58% |

The mirror direction loses the carry and keeps the (mirrored) tail.
An earlier revision of this study reported the opposite ordering; that
result was an artifact of the contract-splice bug (revision note #1)
and is retracted. The naive quant-retail framing — "short the front,
collect the roll-down" — survives correct accounting.

### Transaction-cost sensitivity (spec direction)

| tc (bps/leg/roll) | train Sharpe | test Sharpe | train MDD | test MDD |
|---|---|---|---|---|
| 1  | +0.07 | +0.88 | −23% | −20% |
| 5  | −0.00 | +0.80 | −24% | −20% |
| 10 | −0.09 | +0.70 | −25% | −20% |
| 20 | −0.26 | +0.50 | −27% | −20% |
| 30 | −0.43 | +0.31 | −34% | −21% |

Monotone in costs, as expected. At realistic institutional slippage
(1–5 bps) the test-window result survives; at 30 bps the training
window is deeply negative and the test window marginal — cost
assumptions matter more here than for the option strategies, because
the calendar trades 4 legs every month.

## Phase 2 — Strategy B results

### CBOE PUT index (canonical, published)

| window | Sharpe | ann. return | ann. vol | max DD | α vs SPX-TR (ann.) | β |
|---|---|---|---|---|---|---|
| train (2013-2018) | **+0.68** | 6.1% | 9.1% | −15.5% | −1.8% | 0.64 |
| test  (2019-2024) | **+0.70** | 9.9% | 14.1% | −28.9% | −1.6% | 0.62 |

Alpha is regressed on daily excess returns versus the S&P 500 **total
return** index with a T-bill risk-free rate. It is mildly negative in
both windows — put-writing delivers a different risk profile than
equities (β ≈ 0.6, upside-capped, crash-exposed), not CAPM alpha. An
earlier revision reported ≈0 alpha against price-only SPX; dividends
account for the difference. This is consistent with Israelov & Nielsen
(2015): option-writing returns are largely equity beta plus the
volatility risk premium, and the VRP portion shows up in the improved
Sharpe (0.70 vs SPX-TR's drawdown-heavier path), not in alpha.

### Synthetic Black-Scholes put-writer (replication)

Monthly −0.30Δ short put, VIX as the 30-day ATM IV proxy, 5% of gross
premium round-trip costs.

| window | Sharpe | ann. return | max DD | monthly corr vs PUT |
|---|---|---|---|---|
| train | +0.65 | 4.7% | −15.7% | **0.825** ✓ |
| test  | +0.28 | 3.1% | −26.5% | ″ |

**Sanity gate passes** (monthly correlation 0.825 ≥ 0.6). The synthetic
engine tracks the PUT index in shape and timing; 2019-2024
underperformance is consistent with the documented approximations (VIX
overstates ATM IV in crashes; calendar-month cycles misalign expiry
timing around COVID). The engine is a replication artifact; the PUT
index is the primary deliverable.

### Put-spread variant (short −0.30Δ / long −0.10Δ)

| variant | train Sharpe | test Sharpe | train ret | test ret | train MDD | test MDD |
|---|---|---|---|---|---|---|
| naked  | +0.65 | +0.28 | 4.7% | 3.1% | −15.7% | −26.5% |
| **spread** | **+1.22** | **+0.54** | 4.0% | 2.6% | **−4.1%** | **−9.2%** |

The spread gives up ~0.5–0.7% of annualized return for a 3× smaller
max drawdown and roughly double the Sharpe in both windows. Costs are
charged on both legs (5% of gross premium round-trip). This is the
strongest single construction in the study.

### Strategy A vs Strategy B side-by-side (test window, 2019-2024)

| strategy | Sharpe | ann. return | max DD | 1987 stress |
|---|---|---|---|---|
| Strategy A — short front / long second | +0.88 | 11.6% | −20% | −29.5% |
| Strategy B — PUT index (canonical) | +0.70 | 9.9% | −29% | n/a |
| Strategy B — synthetic spread | +0.54 | 2.6% | −9% | −3.3% |

Strategy A posts the best test-window Sharpe but carries the deepest
tail (a third of capital in a 1987-style day, and a flat-to-negative
training window). The spread is the only construction whose worst-case
single day is survivable by construction rather than by luck.

## Phase 3 — Strategy C results

Strategy C gates Strategy B on the **prior month-end** VRP signal
`VIX − RV20` (vol points): a position is taken in month N+1 only when
month N's closing VRP is at or above the threshold.

### Spec-default baseline (threshold = 2 vol points)

Gates ~27% of months.

| variant | gating | train Sharpe | test Sharpe | train MDD | test MDD | active % |
|---|---|---|---|---|---|---|
| naked  | off (B) | +0.65 | +0.28 | −15.7% | −26.5% | 100% |
| naked  | on  (C) | +0.73 | +0.27 | −7.4%  | −26.5% | 73%  |
| spread | off (B) | +1.22 | +0.54 | −4.1%  | −9.2%  | 100% |
| spread | on  (C) | +1.08 | +0.62 | −2.2%  | −9.1%  | 73%  |

### Threshold sensitivity (train-then-test)

Thresholds swept over `{no gate, −2 … +6}` vol points on 2013-2018
only — **"no gate" is an explicit candidate**, so the selection cannot
manufacture a gate where none is warranted. The train-maximizing choice
per variant is then evaluated once on 2019-2024.

| variant | train-optimal | train Sharpe @ chosen | test Sharpe @ chosen | ungated test Sharpe |
|---|---|---|---|---|
| naked  | +1.0 vp | +0.83 | +0.21 | +0.28 |
| spread | +1.0 vp | +1.21 | +0.49 | **+0.54** |

Train Sharpe is hump-shaped with a peak at +1 vol point for both
variants, beating no-gate in-sample (spread: 1.21 vs 1.18). Out of
sample the ordering reverses: **the train-selected gate underperforms
the ungated strategy for both variants.**

### Verdict

The VRP gate does not survive honest evaluation. It reduces training
drawdowns (it is, mechanically, a way to skip some months) but the
premium it forfeits exceeds the losses it avoids out of sample. An
earlier revision reported test Sharpe 0.81 for a gated spread; that
number rested on gating with the month-open day's own closing VRP (a
same-day leak) and on selecting a grid-boundary threshold that the
ungated strategy beat in train. Both defects are fixed, and the honest
conclusion is negative. Phase 4 nevertheless applies the overlays to
C(spread, thr=+1) — the train-selected configuration — because
test-window information cannot be used to re-select the base.

## Phase 4 — Tail-risk overlays

Three overlays from the project spec, applied to Strategy C (spread,
threshold +1 — the Phase 3 train-optimal). Parameters pinned to the
spec, not tuned. The regime filter trades at t+1: a stress close
triggers exit at the next session's close, so the trigger day's loss is
taken, as it would be live.

- **Overlay 1 — VIX regime filter.** Cash out when VIX > 30 or the VX
  term structure inverts; re-enter after 7 consecutive calm days.
- **Overlay 2 — Realized-vol position scaling.** Scale by
  `min(1, 0.10 / rv20)` of the strategy's own trailing vol (lagged).
- **Overlay 3 — Tail-hedge spend.** 15% of premium on a 5Δ 1-month put.

| configuration | train Sharpe | test Sharpe | train MDD | test MDD |
|---|---|---|---|---|
| base C(spread, thr=+1) | +1.21 | +0.49 | −2.2% | −9.1% |
| + O1 regime filter      | +0.22 | +0.26 | −4.7% | −4.4% |
| + O2 vol scale 10%      | +1.21 | +0.51 | −2.2% | −8.6% |
| + O3 tail hedge 15%     | +0.70 | +0.21 | −5.9% | −13.0% |
| all three combined      | +0.05 | −0.01 | −7.3% | −14.1% |

### Verdict

**No overlay adds out-of-sample value; the regime filter subtracts
most of it.** The filter halves the test max drawdown (−9.1% → −4.4%)
but at the cost of half the Sharpe, because every exit is one day late
by necessity — it eats the crash day, then sits out the vol-crush
recovery days that follow. An earlier revision reported the filter as
the study's best result (test Sharpe 1.03); that entire improvement
was the same-day exit leak (revision note #2). The vol-scaling overlay
is inert (the spread rarely exceeds 10% trailing vol). The tail hedge
remains a drag even after fixing its double-charged premium: 15% of
premium buys 5Δ puts that expire worthless in almost every month, and
the spread's long wing already covers the crash path — redundant
insurance, purchased monthly. Combining all three stacks three costs
on one risk.

The best construction identified by the study is therefore the plain
**Strategy B put-spread** — every gate and overlay tried on top of it
reduced out-of-sample performance.

## Phase 5 — Tail-risk analysis

Moving-block bootstrap (Kunsch 1989; block 40, 2000 paths), daily
VaR/ES, probabilistic Sharpe ratios, an October-1987 extrapolated
stress test, and block-size / method robustness. All computed on the
full 2013–2024 sample.

### Bootstrap Sharpe distributions and PSR

| construction | Sharpe p05 | p50 | p95 | full-sample | PSR |
|---|---|---|---|---|---|
| A short-front               | +0.01 | +0.47 | +0.97 | +0.46 | 0.957 |
| A long-front                | −1.01 | −0.59 | −0.19 | −0.60 | 0.033 |
| B PUT index                 | +0.18 | +0.69 | +1.40 | +0.67 | 0.990 |
| **B synth spread**          | **+0.34** | **+0.83** | **+1.35** | **+0.81** | **0.997** |
| C spread thr=+1             | +0.25 | +0.80 | +1.34 | +0.76 | 0.995 |
| C spread thr=+1 + O1        | −0.15 | +0.27 | +0.70 | +0.24 | 0.804 |

PSR is the probability the true Sharpe exceeds zero given the sample
length, skew, and kurtosis (Bailey & López de Prado 2012). The spread
constructions are statistically credible (PSR ≥ 0.99). A short-front's
5th percentile sits at zero — twelve years of data cannot rule out
that its carry is noise. The O1-filtered construction's interval spans
well into negative territory: the filter converts a credible strategy
into an incredible one.

### Daily 1%-VaR and Expected Shortfall

| construction | 1% VaR (daily) | 1% ES (daily) |
|---|---|---|
| A short-front               | −2.74% | −4.05% |
| A long-front                | −1.81% | −2.21% |
| B PUT index                 | −2.27% | −3.79% |
| B synth spread              | −0.88% | −1.14% |
| C spread thr=+1             | −0.76% | −1.08% |
| C spread thr=+1 + O1        | −0.56% | −0.75% |

The spread's structural wing cuts the daily tail by ~3× versus the
naked constructions. (The O1 row's small tail is bought with the
Sharpe destruction documented in Phase 4 — a thin left tail is not by
itself a recommendation.)

### Bootstrap robustness (headline C+O1 row)

| method | Sharpe p05 | p50 | p95 |
|---|---|---|---|
| MBB, block 20 | −0.15 | +0.27 | +0.72 |
| MBB, block 40 | −0.15 | +0.27 | +0.70 |
| MBB, block 60 | −0.15 | +0.28 | +0.73 |
| stationary (Politis-Romano, mean 40) | −0.18 | +0.24 | +0.68 |

Conclusions are insensitive to the block-size judgment call and to the
fixed-vs-random block-length choice.

### October 1987 stress test

Scenario (historical record, no tuning): SPX −20.5%, VIX-equivalent
+30 vol points, VX term structure inverts (front +30, second +20).
Option strikes are the same delta targets the strategies trade
(−0.30Δ ⇒ K≈96.9, −0.10Δ ⇒ K≈93.3 at S=100, IV=20%, 20 days left).

| construction | single-day PnL (% of capital) |
|---|---|
| A short-front               | **−29.5%** |
| A long-front (mirror gain)  | +29.5% |
| B synth naked −0.30Δ        | −17.6% |
| B synth spread              | **−3.3%** |
| C spread thr=+1             | −3.3% |
| C spread + O1               | −3.3% (day 0); O1 exits day 1+ |

A short-front loses nearly a third of capital in one day — not a
survivable event for a scaled allocator, and the number that should be
read alongside its +0.88 test Sharpe. The naked writer loses ~18%. The
spread's loss is capped near (K_short − K_long − net premium) by
construction at −3.3%. An earlier revision reported −8.7% for the
spread using hardcoded strikes ~4× wider apart than the strategy's
actual deltas. **On a surprise shock the regime filter provides no
day-0 protection** — VIX is below 30 the day before a black swan by
definition — so C+O1 takes the same hit; its exit occurs at the next
close.

### Honest limitations

1. **Sample thinness.** 2013–2024 contains at most four major
   short-vol events (2015, Feb 2018, Mar 2020, 2022) — two per window.
   Bootstrap intervals are in-distribution statements conditional on
   this sample, not population statistics. Strategy A's train/test
   Sharpe gap (+0.07 vs +0.88) is the cleanest illustration: the
   carry's profitability over any 6-year window is regime luck.
2. **Spec deviation on the split.** The spec required Feb 2018 in the
   test set; the 2013 data floor forces it into training. The test set
   retains COVID 2020 and 2022. A reader should treat train-window
   drawdown behavior around Feb 2018 as partially "seen" by every
   train-selected parameter.
3. **Multiple testing.** Two strategy-A directions, two B variants,
   ten C thresholds, and three overlays were evaluated; only the
   train-then-test protocol and the PSR/bootstrap machinery guard
   against selection bias, and they cannot fully undo it. The
   defensible claims are the ones that needed no selection: the spread
   beats the naked writer, and the gates/overlays failed out of sample.
4. **1987 extrapolation is approximate.** VIX did not exist in 1987;
   IV levels are back-fit. You could not exit short puts at theoretical
   prices on Oct 19 — treat the stress figures as lower-bound losses.
5. **BS marks in crises.** Strategy B/C marks at Black-Scholes
   theoretical prices; real spreads gap wide exactly when it matters.
   Live PnL under stress is worse than marked.
6. **This is short volatility.** The spread + honest accounting is the
   best construction found, but it is still short the left tail. The
   analysis quantifies how much premium survives honest tail-risk
   accounting; it does not make the tail go away.

## Repository notes

- `src/vrp/data/vx_futures.py` builds splice-free held-contract
  returns; never diff the market-designated `front_settle` column
  across a roll.
- All published numbers regenerate from the scripts above; `reports/`
  is gitignored by design.
