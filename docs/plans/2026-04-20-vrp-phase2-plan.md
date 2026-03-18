# VRP Phase 2 — Strategy B (Systematic Put-Writing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Strategy B — systematic monthly SPX put-writing — along two tracks: (1) the canonical CBOE PUT index as the ground-truth published backtest, (2) a Black-Scholes-based synthetic put-writer as a replication + pedagogical artifact, and (3) a put-spread variant using the synthetic engine. All three sit alongside the Phase 1 flipped-direction calendar as workable VRP constructions.

**Architecture:** Extend the existing `vrp` package with `vrp.util.bs` (Black-Scholes helpers) and `vrp.strategies.strategy_b` (synthetic put-writer engine supporting both naked and spread variants). Two new runner scripts: one for the PUT-index analysis, one for the synthetic engine + comparison. A sanity gate in the synthetic runner requires ≥0.6 monthly-return correlation against the PUT index.

**Tech Stack:** Python 3.11, pandas, numpy, scipy.stats (Black-Scholes uses `scipy.stats.norm`), matplotlib. No options data fetch — synthetic engine uses VIX as a 30-day ATM IV proxy, flagged as approximate.

**Branch:** `phase2-strategy-b`, based on `phase1-vrp` (Phase 1 code is not yet merged to main). New worktree at `.worktrees/phase2-strategy-b`.

**Train/Test Split:** Same as Phase 1. Train 2013-01-01 → 2018-12-31; Test 2019-01-01 → 2024-12-31. The CBOE PUT index covers back to 1986 so the full index history is available for context, but the split is held constant so Strategy B results compare directly to Strategy A.

**Commit strategy:** One commit per task.

---

## File Structure

```
src/vrp/
├── util/
│   └── bs.py                         # NEW — Black-Scholes price/delta/strike
└── strategies/
    └── strategy_b.py                 # NEW — synthetic monthly put-writer
scripts/
├── run_strategy_b_putindex.py        # NEW — CBOE PUT index analysis
├── run_strategy_b_synthetic.py       # NEW — BS put-writer + PUT-index comparison
└── run_strategy_b_spread.py          # NEW — put-spread variant comparison
tests/
├── test_bs.py                        # NEW
└── test_strategy_b.py                # NEW
```

---

## Task 1: Black-Scholes Utilities

**Files:**
- Create: `src/vrp/util/bs.py`
- Create: `tests/test_bs.py`

- [ ] **Step 1.1: Write failing test**

`tests/test_bs.py`:

```python
import math

import numpy as np
import pytest

from vrp.util.bs import bs_price, bs_delta, strike_from_delta


def test_bs_price_at_the_money_call_put_parity():
    # S = K, r = 0, T = 0.25, sigma = 0.20: ATM call and put should be equal
    S, K, T, sigma = 100.0, 100.0, 0.25, 0.20
    call = bs_price(S, K, T, sigma, r=0.0, option_type="call")
    put = bs_price(S, K, T, sigma, r=0.0, option_type="put")
    assert abs(call - put) < 1e-9


def test_bs_price_deep_itm_call_floor():
    # Deep ITM call (S=200, K=100, short-dated) should be ~ S - K = 100
    px = bs_price(200.0, 100.0, 0.01, 0.20, r=0.0, option_type="call")
    assert abs(px - 100.0) < 0.5


def test_bs_delta_signs():
    # Call delta in (0, 1), put delta in (-1, 0)
    call_d = bs_delta(100.0, 100.0, 0.25, 0.20, r=0.0, option_type="call")
    put_d = bs_delta(100.0, 100.0, 0.25, 0.20, r=0.0, option_type="put")
    assert 0 < call_d < 1
    assert -1 < put_d < 0
    # At-the-money with r=0, call and put deltas sum to approx 0
    assert abs(call_d + put_d) < 1e-6


def test_strike_from_delta_round_trip():
    # Pick a target delta for a put, recover the strike, price it, check
    target_delta = -0.30
    S, T, sigma = 100.0, 30 / 365.0, 0.20
    K = strike_from_delta(S, T, sigma, r=0.0, option_type="put",
                          target_delta=target_delta)
    recovered = bs_delta(S, K, T, sigma, r=0.0, option_type="put")
    assert abs(recovered - target_delta) < 1e-4


def test_strike_from_delta_out_of_range():
    # |target_delta| must be in (0, 1)
    with pytest.raises(ValueError):
        strike_from_delta(100, 0.25, 0.20, r=0, option_type="put", target_delta=-1.5)
```

- [ ] **Step 1.2: Run test — verify FAIL**

Run: `.venv/bin/pytest tests/test_bs.py -v`
Expected: `ModuleNotFoundError: No module named 'vrp.util.bs'`.

- [ ] **Step 1.3: Write implementation**

`src/vrp/util/bs.py`:

```python
"""Black-Scholes option pricing and Greeks.

European options on a non-dividend-paying underlying, risk-free rate
optionally zero (the default for the Strategy B backtest, since we're
analyzing a premium-collection strategy where the risk-free drift is
small relative to IV).

References:
- Black, Scholes (1973) "The Pricing of Options and Corporate Liabilities"
- Hull (2017) "Options, Futures, and Other Derivatives" (ch. 15, 17)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError(f"T ({T}) and sigma ({sigma}) must be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.0,
             option_type: str = "call") -> float:
    """European Black-Scholes option price."""
    if T <= 0:
        # At expiry, intrinsic value
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0,
             option_type: str = "call") -> float:
    """European Black-Scholes delta."""
    d1, _ = _d1_d2(S, K, T, sigma, r)
    if option_type == "call":
        return float(norm.cdf(d1))
    if option_type == "put":
        return float(norm.cdf(d1) - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def strike_from_delta(S: float, T: float, sigma: float, r: float,
                      option_type: str, target_delta: float) -> float:
    """Invert the delta formula: solve K such that delta(K) == target_delta.

    Uses the closed-form inversion:
        For a put with target_delta in (-1, 0):
            K = S * exp(-sigma * sqrt(T) * N^{-1}(1 + target_delta)
                       + (r + 0.5 sigma^2) T)
        For a call with target_delta in (0, 1):
            K = S * exp(-sigma * sqrt(T) * N^{-1}(target_delta)
                       + (r + 0.5 sigma^2) T)
    """
    if not (0.0 < abs(target_delta) < 1.0):
        raise ValueError(f"|target_delta| must be in (0, 1), got {target_delta}")
    if option_type == "put":
        if target_delta >= 0:
            raise ValueError("put target_delta must be negative")
        nd1 = norm.ppf(1.0 + target_delta)
    elif option_type == "call":
        if target_delta <= 0:
            raise ValueError("call target_delta must be positive")
        nd1 = norm.ppf(target_delta)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    K = S * math.exp(-sigma * math.sqrt(T) * nd1 + (r + 0.5 * sigma * sigma) * T)
    return float(K)
```

- [ ] **Step 1.4: Run test — verify PASS**

Run: `.venv/bin/pytest tests/test_bs.py -v`
Expected: 5 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add src/vrp/util/bs.py tests/test_bs.py
git commit -m "vrp: Black-Scholes price/delta/strike-from-delta (Phase 2 Task 1)"
```

---

## Task 2: CBOE PUT Index Analyzer

**Files:**
- Create: `scripts/run_strategy_b_putindex.py`

This is the canonical Strategy B deliverable. The CBOE PUT index is the published benchmark implementation of monthly ATM cash-secured put writing. We analyze it directly.

- [ ] **Step 2.1: Write the script**

`scripts/run_strategy_b_putindex.py`:

```python
"""Strategy B — canonical CBOE PUT index analysis.

PUT is the published CBOE S&P 500 PutWrite Index: monthly cash-secured
sales of at-the-money SPX puts. This script is the primary Strategy B
backtest — the PUT series already bakes in realistic execution and
transaction costs from CBOE's methodology.

Outputs:
    reports/strategy_b_putindex/metrics_train.json
    reports/strategy_b_putindex/metrics_test.json
    reports/strategy_b_putindex/regime_metrics_test.json
    reports/strategy_b_putindex/alpha_beta_vs_spx.json
    reports/strategy_b_putindex/equity_vs_spx.png
    reports/strategy_b_putindex/drawdown.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vrp.data.cboe_indices import load_cboe_index
from vrp.data.spx import load_spx
from vrp.report.metrics import summary, drawdown_series
from vrp.report.regimes import regime_metrics

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"


def _daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def _alpha_beta(strategy_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """Annualized alpha + beta of strategy on benchmark. OLS, no intercept
    restrictions. Returns are daily; alpha is annualized (×252)."""
    aligned = pd.concat([strategy_ret, bench_ret], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    x = aligned["x"].values
    y = aligned["y"].values
    # OLS: y = alpha_daily + beta * x
    x_mean, y_mean = x.mean(), y.mean()
    var_x = ((x - x_mean) ** 2).mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()
    beta = float(cov_xy / var_x) if var_x > 0 else float("nan")
    alpha_daily = float(y_mean - beta * x_mean)
    alpha_ann = alpha_daily * 252
    return {"alpha_annual": alpha_ann, "beta": beta,
            "n_days": int(len(aligned))}


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_b_putindex"
    out_dir.mkdir(parents=True, exist_ok=True)

    put = load_cboe_index("PUT")
    spx = load_spx(start="2012-01-01")["close"]

    put_ret = _daily_returns(put)
    spx_ret = _daily_returns(spx)

    train_put = put_ret.loc[TRAIN_START:TRAIN_END]
    test_put  = put_ret.loc[TEST_START:TEST_END]

    (out_dir / "metrics_train.json").write_text(json.dumps(summary(train_put), indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(summary(test_put), indent=2))
    (out_dir / "regime_metrics_test.json").write_text(
        json.dumps(regime_metrics(test_put), indent=2, default=str)
    )

    ab_train = _alpha_beta(train_put, spx_ret.loc[TRAIN_START:TRAIN_END])
    ab_test  = _alpha_beta(test_put,  spx_ret.loc[TEST_START:TEST_END])
    (out_dir / "alpha_beta_vs_spx.json").write_text(
        json.dumps({"train": ab_train, "test": ab_test}, indent=2)
    )

    # Normalize both series to 1.0 at the start of the train window
    common_start = max(put_ret.index.min(), spx_ret.index.min(),
                       pd.Timestamp(TRAIN_START))
    norm_put = (1 + put_ret).cumprod()
    norm_spx = (1 + spx_ret).cumprod()
    norm_put = norm_put / norm_put.loc[common_start:].iloc[0]
    norm_spx = norm_spx / norm_spx.loc[common_start:].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame({"PUT index": norm_put, "SPX": norm_spx}).loc[common_start:].plot(ax=ax)
    ax.set_title("Strategy B — CBOE PUT index vs SPX (normalized)")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red", label="out-of-sample")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "equity_vs_spx.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3))
    drawdown_series(put_ret).loc[common_start:].plot(ax=ax, color="red")
    ax.set_title("Strategy B — PUT index drawdown")
    ax.set_ylabel("Drawdown")
    fig.savefig(out_dir / "drawdown.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy B (PUT index) outputs written to {out_dir}")
    print("Train:", json.dumps(summary(train_put), indent=2))
    print("Test: ", json.dumps(summary(test_put),  indent=2))
    print("alpha/beta train:", ab_train)
    print("alpha/beta test: ", ab_test)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Run the script**

Run: `.venv/bin/python scripts/run_strategy_b_putindex.py`
Expected outputs:
- JSON + PNG files land in `reports/strategy_b_putindex/`
- Train Sharpe typically positive, around 0.5–1.0 (PUT has historically ridden a positive drift).
- Test Sharpe likely lower than train (COVID + 2022 bear); still plausibly positive but smaller.
- Beta to SPX should be ~0.5–0.7 (PUT has a put-write risk profile — captures most upside, absorbs downside).
- Annualized alpha vs SPX may be small (PUT does not systematically beat SPX on alpha; it delivers a different *risk profile*).

**Sanity bounds:** If train Sharpe > 2 or beta outside (0.2, 1.0), something is wrong — investigate.

- [ ] **Step 2.3: Commit**

```bash
git add scripts/run_strategy_b_putindex.py
git commit -m "vrp: Strategy B — CBOE PUT index canonical analysis (Phase 2 Task 2)"
```

---

## Task 3: Synthetic Put-Writer Engine

**Files:**
- Create: `src/vrp/strategies/strategy_b.py`
- Create: `tests/test_strategy_b.py`

### Design

The engine simulates monthly put-writing on an SPX price series, using VIX as the at-the-money 30-day IV proxy. Each month:

1. On the last trading day before a new monthly option cycle begins (first-of-month), pick a strike `K` such that the short-put has delta ≈ target_delta using `strike_from_delta(S_t, T=30/365, sigma=VIX_t/100, ...)`.
2. "Sell" the put at Black-Scholes premium `P_0`.
3. For each subsequent trading day until expiry, mark-to-market the short position at `-bs_price(S_t, K, T_t, VIX_t/100)` and record daily P&L as the change in mark plus the collected premium amortization.
4. On expiry: compute intrinsic value `max(K - S_expiry, 0)`, apply to close the position.
5. New cycle begins.

Simplifications:
- Risk-free rate = 0.
- VIX as IV proxy, flagged as approximate (VIX is a variance-swap construct, not strictly ATM IV; over/underestimates depending on smile and term-structure).
- Monthly option cycles align with calendar-month-ends rather than SPX monthly expiries (third Fridays). For a rough replication this is fine; the PUT-index correlation gate will surface meaningful divergence.
- Transaction cost: 5% of premium on open (bid-ask spread proxy), same on close. `tc_pct_of_premium=0.05` default.
- Position size: cash-secured → notional sold = 1 "contract" per unit capital. PnL expressed as a return on the capital (K * 1 per unit).

- [ ] **Step 3.1: Write failing test**

`tests/test_strategy_b.py`:

```python
import numpy as np
import pandas as pd
import pytest

from vrp.strategies.strategy_b import run_strategy_b


def _synth_spx_and_vix(n_days: int = 252):
    """Flat SPX with moderate IV — the synthetic put writer should net
    approximately the VRP (IV - RV) as premium collected."""
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    spx = pd.Series(100.0 + np.zeros(n_days), index=idx)
    vix = pd.Series(20.0 + np.zeros(n_days), index=idx)
    return spx, vix


def test_strategy_b_returns_series_shape():
    spx, vix = _synth_spx_and_vix(120)
    out = run_strategy_b(spx, vix, target_delta=-0.30)
    assert set(out.keys()) >= {"daily_return", "positions", "monthly_pnl"}
    assert len(out["daily_return"]) == len(spx)


def test_strategy_b_profitable_on_flat_underlying():
    # Flat SPX: puts expire worthless every month -> premium is collected.
    # Net PnL should be positive after TC.
    spx, vix = _synth_spx_and_vix(252)
    out = run_strategy_b(spx, vix, target_delta=-0.30,
                         tc_pct_of_premium=0.05)
    assert out["daily_return"].sum() > 0


def test_strategy_b_losses_on_crash():
    # SPX drops 20% in one day mid-month -> short puts go deep ITM ->
    # big negative PnL. Synthetic IV (VIX) stays flat for simplicity.
    n = 60
    idx = pd.bdate_range("2020-01-02", periods=n)
    spx = pd.Series(100.0, index=idx)
    spx.iloc[n // 2:] = 80.0
    vix = pd.Series(25.0, index=idx)
    out = run_strategy_b(spx, vix, target_delta=-0.30, tc_pct_of_premium=0.0)
    # Cumulative PnL should end up significantly negative
    assert (1 + out["daily_return"]).cumprod().iloc[-1] < 0.95


def test_strategy_b_invalid_delta_raises():
    spx, vix = _synth_spx_and_vix(60)
    with pytest.raises(ValueError):
        run_strategy_b(spx, vix, target_delta=0.30)  # positive delta for put
    with pytest.raises(ValueError):
        run_strategy_b(spx, vix, target_delta=-1.5)
```

- [ ] **Step 3.2: Run test — verify FAIL**

Run: `.venv/bin/pytest tests/test_strategy_b.py -v`
Expected: `ModuleNotFoundError: No module named 'vrp.strategies.strategy_b'`.

- [ ] **Step 3.3: Write implementation**

`src/vrp/strategies/strategy_b.py`:

```python
"""Strategy B — synthetic monthly put-writer on SPX.

Approximate Black-Scholes simulation of the CBOE PUT index methodology:
each calendar month, sell a 30-day SPX put at target delta, mark to
market daily, settle at month-end. This is a *pedagogical* replication.
Meaningful divergence from the PUT index is expected because:
- VIX is used as the ATM IV proxy (a variance-swap construct, not strict ATM IV).
- Monthly option cycles align to calendar-month ends rather than the
  third-Friday SPX monthly expiry used by CBOE.
- Risk-free rate is held at zero.

References:
- CBOE White Paper on PUT Index methodology.
- Bondarenko (2014). Why Are Put Options So Expensive?
- Israelov & Nielsen (2015). Covered Calls Uncovered (AQR). Implementation
  realism critique of naive option-writing backtests.

The correlation-against-PUT sanity gate (see scripts/run_strategy_b_synthetic.py)
requires monthly-return correlation ≥ 0.6. This engine is documented as
approximate and is *not* the primary Strategy B deliverable — that is the
CBOE PUT index analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from vrp.util.bs import bs_price, strike_from_delta


Variant = Literal["naked", "spread"]


@dataclass
class StrategyBConfig:
    target_delta: float = -0.30
    long_put_delta: Optional[float] = None  # e.g. -0.10 for spread
    maturity_days: int = 30
    tc_pct_of_premium: float = 0.05
    r: float = 0.0


def _month_starts(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the first trading date of each calendar month in idx."""
    df = pd.Series(index=idx, data=1)
    return list(df.groupby(df.index.to_period("M")).apply(lambda s: s.index[0]))


def _mark_leg(S_series: pd.Series, K: float, sigma_series: pd.Series,
              days_to_expiry: pd.Series, r: float) -> pd.Series:
    """Daily mark of a single long-put leg at quantity +1."""
    marks = []
    for dt, S, sigma_pct, dte in zip(S_series.index, S_series.values,
                                      sigma_series.values, days_to_expiry.values):
        T = max(float(dte), 0.0) / 365.0
        sigma = float(sigma_pct) / 100.0
        marks.append(bs_price(float(S), K, T, sigma, r=r, option_type="put"))
    return pd.Series(marks, index=S_series.index)


def run_strategy_b(spx: pd.Series, vix: pd.Series,
                   target_delta: float = -0.30,
                   long_put_delta: Optional[float] = None,
                   maturity_days: int = 30,
                   tc_pct_of_premium: float = 0.05,
                   r: float = 0.0) -> Dict[str, object]:
    """Simulate monthly short-put writing on SPX with VIX as IV proxy.

    Parameters
    ----------
    spx, vix:
        Aligned daily price and VIX-spot series.
    target_delta:
        Short-put delta target (e.g. -0.30). Must be in (-1, 0).
    long_put_delta:
        If not None, enables spread variant: buy a further-OTM put of
        this delta (e.g. -0.10). Must be between target_delta and 0.
    maturity_days:
        Time-to-expiry in calendar days at option open.
    tc_pct_of_premium:
        Round-trip transaction cost as a fraction of premium collected.
    r:
        Risk-free rate (annualized). Default 0.

    Returns
    -------
    Dict with keys ``daily_return``, ``positions`` (DataFrame), ``monthly_pnl``.
    """
    if not (-1.0 < target_delta < 0.0):
        raise ValueError(f"target_delta must be in (-1, 0), got {target_delta}")
    if long_put_delta is not None and not (target_delta < long_put_delta < 0.0):
        raise ValueError(
            f"long_put_delta ({long_put_delta}) must be between target_delta "
            f"({target_delta}) and 0"
        )

    cfg = StrategyBConfig(target_delta=target_delta,
                           long_put_delta=long_put_delta,
                           maturity_days=maturity_days,
                           tc_pct_of_premium=tc_pct_of_premium, r=r)

    # Align SPX and VIX on common index
    df = pd.concat([spx.rename("S"), vix.rename("vix")], axis=1).dropna()
    idx = df.index
    month_start_dates = _month_starts(idx)

    daily_pnl = pd.Series(0.0, index=idx)
    positions = []
    monthly_pnl = []

    for i, open_date in enumerate(month_start_dates):
        # Next cycle start defines the close date
        close_date = (month_start_dates[i + 1]
                      if i + 1 < len(month_start_dates) else idx[-1])
        cycle_idx = idx[(idx >= open_date) & (idx < close_date)]
        if len(cycle_idx) < 2:
            continue

        S0 = float(df.loc[open_date, "S"])
        sigma0 = float(df.loc[open_date, "vix"]) / 100.0
        T0 = cfg.maturity_days / 365.0

        K_short = strike_from_delta(S0, T0, sigma0, cfg.r,
                                    option_type="put",
                                    target_delta=cfg.target_delta)
        p_short_open = bs_price(S0, K_short, T0, sigma0, cfg.r, "put")

        K_long = None
        p_long_open = 0.0
        if cfg.long_put_delta is not None:
            K_long = strike_from_delta(S0, T0, sigma0, cfg.r,
                                        option_type="put",
                                        target_delta=cfg.long_put_delta)
            p_long_open = bs_price(S0, K_long, T0, sigma0, cfg.r, "put")

        premium_collected = p_short_open - p_long_open

        # Days to expiry within cycle
        dte = pd.Series(
            [max(cfg.maturity_days - (d - open_date).days, 0) for d in cycle_idx],
            index=cycle_idx,
        )

        short_marks = _mark_leg(df.loc[cycle_idx, "S"], K_short,
                                df.loc[cycle_idx, "vix"], dte, cfg.r)
        long_marks = pd.Series(0.0, index=cycle_idx)
        if K_long is not None:
            long_marks = _mark_leg(df.loc[cycle_idx, "S"], K_long,
                                    df.loc[cycle_idx, "vix"], dte, cfg.r)

        # Net position mark: +premium (received) minus (-short mark + long mark)
        # We are SHORT the short leg and LONG the long leg.
        # Equity contribution of position at time t =
        #   premium_collected + (-short_marks_t + long_marks_t) - (p_short_open - p_long_open)
        # which simplifies to: (-short_marks_t + long_marks_t) - (-p_short_open + p_long_open)
        #                    = (p_short_open - short_marks_t) + (long_marks_t - p_long_open)
        position_value = (p_short_open - short_marks) + (long_marks - p_long_open)

        # Apply transaction cost: half on open, half on close
        tc_open = cfg.tc_pct_of_premium * 0.5 * abs(premium_collected)
        tc_close = cfg.tc_pct_of_premium * 0.5 * abs(premium_collected)
        position_value.iloc[0] -= tc_open
        position_value.iloc[-1] -= tc_close

        # PnL is denominated on capital = K_short (cash-secured notional).
        # Convert to returns on K_short.
        position_return = position_value / K_short

        # Day-over-day change in equity contribution = daily PnL for this cycle
        cycle_daily_pnl = position_return.diff().fillna(position_return.iloc[0])
        daily_pnl.loc[cycle_idx] = cycle_daily_pnl.values

        positions.append({
            "open_date": open_date,
            "close_date": close_date,
            "S0": S0, "sigma0": sigma0,
            "K_short": K_short, "K_long": K_long,
            "p_short_open": p_short_open, "p_long_open": p_long_open,
            "premium_collected": premium_collected,
            "tc_total": tc_open + tc_close,
        })
        monthly_pnl.append({
            "month_start": open_date,
            "return": float(position_return.iloc[-1]),
        })

    positions_df = pd.DataFrame(positions)
    monthly_pnl_df = pd.DataFrame(monthly_pnl)

    return {
        "daily_return": daily_pnl,
        "positions": positions_df,
        "monthly_pnl": monthly_pnl_df,
    }
```

- [ ] **Step 3.4: Run test — verify PASS**

Run: `.venv/bin/pytest tests/test_strategy_b.py -v`
Expected: 4 tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add src/vrp/strategies/strategy_b.py tests/test_strategy_b.py
git commit -m "vrp: Strategy B — synthetic monthly put-writer engine (Phase 2 Task 3)"
```

---

## Task 4: Synthetic Engine Runner + PUT-Index Comparison

**Files:**
- Create: `scripts/run_strategy_b_synthetic.py`

- [ ] **Step 4.1: Write runner**

`scripts/run_strategy_b_synthetic.py`:

```python
"""Synthetic Strategy B (BS put-writer) run and PUT-index correlation gate.

Produces:
    reports/strategy_b_synthetic/metrics_train.json
    reports/strategy_b_synthetic/metrics_test.json
    reports/strategy_b_synthetic/put_index_correlation.json
    reports/strategy_b_synthetic/equity_vs_put_index.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.cboe_indices import load_cboe_index
from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.report.metrics import summary
from vrp.strategies.strategy_b import run_strategy_b

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_b_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx = load_spx(start=TRAIN_START, end=TEST_END)["close"]
    vix = load_vix(start=TRAIN_START, end=TEST_END)
    put = load_cboe_index("PUT")

    out = run_strategy_b(spx, vix, target_delta=-0.30,
                         tc_pct_of_premium=0.05)
    ret = out["daily_return"]

    train_ret = ret.loc[TRAIN_START:TRAIN_END]
    test_ret  = ret.loc[TEST_START:TEST_END]

    (out_dir / "metrics_train.json").write_text(json.dumps(summary(train_ret), indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(summary(test_ret), indent=2))

    # Monthly correlation vs PUT index
    put_ret = put.pct_change().dropna()
    combined = pd.concat([
        ret.rename("synth"),
        put_ret.rename("put"),
    ], axis=1).dropna()
    combined_monthly = combined.resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_corr = float(combined_monthly["synth"].corr(combined_monthly["put"]))
    daily_corr = float(combined["synth"].corr(combined["put"]))

    (out_dir / "put_index_correlation.json").write_text(
        json.dumps({"monthly_correlation": monthly_corr,
                    "daily_correlation": daily_corr,
                    "note": "Sanity gate requires monthly_correlation >= 0.6"},
                   indent=2)
    )

    # Equity vs PUT
    synth_equity = (1 + ret).cumprod()
    put_equity = (1 + put_ret).cumprod()
    common = synth_equity.index.intersection(put_equity.index)
    norm = pd.DataFrame({
        "synthetic BS writer": synth_equity.loc[common],
        "PUT index":           put_equity.loc[common],
    })
    norm = norm / norm.iloc[0]

    fig, ax = plt.subplots(figsize=(11, 4))
    norm.plot(ax=ax)
    ax.set_title(
        f"Strategy B — synthetic vs CBOE PUT (monthly corr = {monthly_corr:.2f})"
    )
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red", label="out-of-sample")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "equity_vs_put_index.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Synthetic Strategy B outputs written to {out_dir}")
    print("Train:", json.dumps(summary(train_ret), indent=2))
    print("Test: ", json.dumps(summary(test_ret),  indent=2))
    print(f"Monthly corr vs PUT: {monthly_corr:.3f}  (expected >= 0.6)")
    print(f"Daily corr vs PUT:   {daily_corr:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run**

Run: `.venv/bin/python scripts/run_strategy_b_synthetic.py`

**Sanity gate:** monthly correlation vs PUT index ≥ 0.6. If below 0.5, investigate the synthetic engine. If the synthetic is systematically worse-Sharpe than PUT by >1.5, that is *expected* (PUT uses real options, better execution); note but do not block on it.

- [ ] **Step 4.3: Commit**

```bash
git add scripts/run_strategy_b_synthetic.py
git commit -m "vrp: Strategy B synthetic runner + PUT-index correlation gate (Phase 2 Task 4)"
```

---

## Task 5: Put-Spread Variant Runner

**Files:**
- Create: `scripts/run_strategy_b_spread.py`

- [ ] **Step 5.1: Write runner**

`scripts/run_strategy_b_spread.py`:

```python
"""Strategy B — put-spread variant comparison.

Runs the synthetic engine twice:
1. Naked put (short -0.30Δ).
2. Put spread (short -0.30Δ, long -0.10Δ).

Spread truncates the left tail at the bought-put strike, at the cost of
a smaller net premium. This runner quantifies that tradeoff.

Outputs:
    reports/strategy_b_spread/comparison.json
    reports/strategy_b_spread/equity.png
    reports/strategy_b_spread/drawdown.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.report.metrics import summary, drawdown_series
from vrp.strategies.strategy_b import run_strategy_b

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"


def _windowed(ret: pd.Series) -> dict:
    return {
        "train": summary(ret.loc[TRAIN_START:TRAIN_END]),
        "test":  summary(ret.loc[TEST_START:TEST_END]),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_b_spread"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx = load_spx(start=TRAIN_START, end=TEST_END)["close"]
    vix = load_vix(start=TRAIN_START, end=TEST_END)

    naked = run_strategy_b(spx, vix, target_delta=-0.30,
                           long_put_delta=None, tc_pct_of_premium=0.05)
    spread = run_strategy_b(spx, vix, target_delta=-0.30,
                            long_put_delta=-0.10, tc_pct_of_premium=0.05)

    results = {
        "naked_put": _windowed(naked["daily_return"]),
        "put_spread": _windowed(spread["daily_return"]),
    }
    (out_dir / "comparison.json").write_text(json.dumps(results, indent=2))

    naked_eq = (1 + naked["daily_return"]).cumprod()
    spread_eq = (1 + spread["daily_return"]).cumprod()

    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame({"naked -0.30Δ": naked_eq, "spread -0.30Δ / -0.10Δ": spread_eq}).plot(ax=ax)
    ax.set_title("Strategy B — naked put vs put spread")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red", label="out-of-sample")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "equity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3))
    pd.DataFrame({
        "naked -0.30Δ":           drawdown_series(naked["daily_return"]),
        "spread -0.30Δ / -0.10Δ": drawdown_series(spread["daily_return"]),
    }).plot(ax=ax)
    ax.set_title("Strategy B — drawdown comparison")
    ax.set_ylabel("Drawdown")
    fig.savefig(out_dir / "drawdown.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy B spread comparison outputs written to {out_dir}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run**

Run: `.venv/bin/python scripts/run_strategy_b_spread.py`
Expected: both variants complete; spread typically has lower ann. return and shallower max drawdown than naked.

- [ ] **Step 5.3: Commit**

```bash
git add scripts/run_strategy_b_spread.py
git commit -m "vrp: Strategy B put-spread variant runner (Phase 2 Task 5)"
```

---

## Task 6: Phase 2 README Section

**Files:**
- Modify: `src/vrp/README.md` — add "Phase 2" section with results tables and framing.

- [ ] **Step 6.1: Edit README**

Add a `## Phase 2 — Strategy B Results` section to `src/vrp/README.md`, after the existing Phase 1 results, documenting:
- The three Phase 2 deliverables (PUT-index canonical, synthetic BS engine, put-spread variant).
- A results table matching the Phase 1 pattern:

```markdown
## Phase 2 — Strategy B Results

### CBOE PUT index (canonical)

| window | Sharpe | ann. return | ann. vol | max DD | alpha vs SPX | beta vs SPX |
|---|---|---|---|---|---|---|
| train (2013-2018) | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> |
| test  (2019-2024) | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> |

### Synthetic BS put-writer (replication, −0.30Δ)

| window | Sharpe | ann. return | max DD | monthly corr vs PUT |
|---|---|---|---|---|
| train | <fill> | <fill> | <fill> | <fill> |
| test  | <fill> | <fill> | <fill> | <fill> |

Sanity gate: monthly correlation ≥ 0.6 (met / not met — <fill>).

### Put-spread variant (−0.30Δ / +(−0.10Δ))

| variant | train Sharpe | test Sharpe | train MDD | test MDD |
|---|---|---|---|---|
| naked  | <fill> | <fill> | <fill> | <fill> |
| spread | <fill> | <fill> | <fill> | <fill> |
```

Fill the `<fill>` entries from the JSON outputs produced by Tasks 2, 4, and 5.

Add a short "Framing" subsection: how Strategy B (working VRP capture via premium collection) complements the flipped Strategy A (VX calendar, also working) — both are live candidates for Phase 4's meta-allocation.

- [ ] **Step 6.2: Commit**

```bash
git add src/vrp/README.md
git commit -m "vrp: Phase 2 README section — Strategy B results (Phase 2 Task 6)"
```

---

## Phase 2 Definition of Done

1. `.venv/bin/pytest` passes (25 Phase 1 tests + 5 bs tests + 4 strategy_b tests = 34 tests).
2. `scripts/run_strategy_b_putindex.py` produces the PUT-index canonical analysis.
3. `scripts/run_strategy_b_synthetic.py` produces the BS synthetic backtest, and monthly correlation vs PUT is ≥ 0.6. If below 0.6, STOP and investigate before declaring done.
4. `scripts/run_strategy_b_spread.py` produces the put-spread comparison.
5. README is updated with filled-in numbers.
6. All 6 task commits present with task-numbered messages.

---

## Self-Review Notes

- **Spec coverage:** Strategy B subpoints from the spec (monthly −0.30Δ, cash-secured, roll monthly, PUT benchmark, put-spread variant) are all addressed. Tasks 2 (PUT canonical), 3-4 (synthetic engine + runner), 5 (spread) map 1:1 to the spec bullets.
- **Placeholders:** All steps are concrete code or concrete commands. The README task has `<fill>` placeholders that the implementer fills from produced JSON — this is a directive, not a deferral.
- **Type consistency:** `run_strategy_b` signature is identical between T3's implementation and T4/T5's callers. Output dict keys (`daily_return`, `positions`, `monthly_pnl`) are consistent.
- **TDD discipline:** T1 (bs) and T3 (strategy_b) follow strict red/green. T2/T4/T5 are runner scripts, exercised via the produced reports — this is the right trade-off (runner tests would be tautological).
- **Scope:** Phase 2 covers only Strategy B per the spec's phased plan. Strategy C (conditional VRP harvester), overlays, and the bootstrap/1987 stress analysis are explicitly deferred to Phases 3–5.
