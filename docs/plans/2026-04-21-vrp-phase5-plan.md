# VRP Phase 5 — Tail-Risk Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Complete the spec's "Tail-Risk Analysis" block. Moving-block bootstrap of strategy returns (preserving vol clustering) to estimate tail distributions, an October 1987 stress test applied to each major construction, and a dedicated tail-risk writeup in the README that documents the full methodology and its honest limitations.

**Architecture:** New `vrp.util.bootstrap` for the moving-block resampler + tail statistics. New `vrp.analysis.stress_1987` for the extrapolated 1987 scenario applied to strategy outputs. One runner script that combines both analyses across the major constructions identified in Phases 1-4, plus the Phase 5 README section.

**Tech Stack:** Same as prior phases.

**Branch:** `phase5-tail-risk`, branched off `phase4-overlays`.

**Constructions to analyze:**

1. Strategy A baseline (short-front / long-second, per spec)
2. Strategy A flipped (long-front / short-second)
3. Strategy B PUT index (canonical)
4. Strategy B synthetic spread (−0.30/−0.10)
5. Strategy C spread at train-optimal threshold (−2 vol points)
6. **Strategy C spread + Overlay 1** (regime filter) — the Phase 4 winner

**Discipline:** bootstrap block size fixed at 40 (middle of the spec's 20-60 range). 1987 scenario parameters hand-picked from historical record, not tuned. No test-window-only analysis here — the bootstrap and stress test operate on the full backtest return series, since the point is distributional characterization, not parameter selection.

---

## File Structure

```
src/vrp/
├── util/
│   └── bootstrap.py                  # moving-block resampler + tail stats
└── analysis/
    ├── __init__.py
    └── stress_1987.py                # Oct 1987 extrapolated scenario
scripts/
└── run_tail_risk.py                  # combined runner: bootstrap + stress
tests/
├── test_bootstrap.py
└── test_stress_1987.py
```

---

## Task 1: Moving-Block Bootstrap Utility

**Files:**
- Create: `src/vrp/util/bootstrap.py`
- Create: `tests/test_bootstrap.py`

### Design

Moving-block bootstrap (MBB): pick random block starts, concatenate N_blocks blocks of length L, trim to target series length. Preserves within-block autocorrelation (and thus vol clustering) while making the path stationary in distribution.

Outputs per simulation: Sharpe, annualized return, annualized vol, max drawdown. User-level API:

- `bootstrap_paths(returns, block_size, n_paths) -> list of Series`
- `bootstrap_metrics(returns, block_size, n_paths) -> DataFrame` (one row per simulation, columns for each metric)
- `var_and_es(returns, alpha=0.01) -> (var, es)` on the *daily-return* distribution (non-bootstrapped)
- `confidence_intervals(metrics_df, alphas=(0.05, 0.95)) -> DataFrame`

- [ ] **Step 1.1: Write failing test**

`tests/test_bootstrap.py`:

```python
import numpy as np
import pandas as pd

from vrp.util.bootstrap import (
    bootstrap_paths,
    bootstrap_metrics,
    var_and_es,
    confidence_intervals,
)


def test_bootstrap_paths_length_matches_input():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-02", periods=500)
    ret = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    paths = bootstrap_paths(ret, block_size=40, n_paths=50, seed=1)
    assert len(paths) == 50
    for p in paths:
        assert len(p) == len(ret)


def test_bootstrap_metrics_columns():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 500))
    df = bootstrap_metrics(ret, block_size=40, n_paths=20, seed=1)
    for c in ("ann_return", "ann_vol", "sharpe", "max_drawdown"):
        assert c in df.columns
    assert len(df) == 20


def test_var_and_es():
    # At alpha=0.05, VaR on a standard-normal-ish series should be about -1.6 stds
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 10_000))
    v, es = var_and_es(ret, alpha=0.05)
    assert -0.02 < v < -0.01
    # ES should be worse (more negative) than VaR
    assert es < v


def test_confidence_intervals():
    df = pd.DataFrame({"x": np.linspace(-1, 1, 101)})
    ci = confidence_intervals(df, alphas=(0.05, 0.95))
    assert abs(ci.loc["x", "p05"] - (-0.9)) < 0.02
    assert abs(ci.loc["x", "p95"] - 0.9) < 0.02
```

- [ ] **Step 1.2: Run — verify FAIL**

Run: `.venv/bin/pytest tests/test_bootstrap.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 1.3: Write implementation**

`src/vrp/util/bootstrap.py`:

```python
"""Moving-block bootstrap and tail statistics.

Moving-block bootstrap (Kunsch 1989): resample blocks of consecutive
returns to generate simulated paths that preserve within-block
autocorrelation and volatility clustering. Breaks cross-block
autocorrelation, which is acceptable at block sizes ≥ 20 trading days
for the practical timescales of monthly-rebalance strategies.

References:
- Kunsch (1989) "The Jackknife and the Bootstrap for General Stationary
  Observations"
- Politis, Romano (1994) "The Stationary Bootstrap" — alternative with
  random block sizes; we use fixed-size MBB for simplicity.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from vrp.report.metrics import summary


def bootstrap_paths(returns: pd.Series, block_size: int, n_paths: int,
                    seed: int = 0) -> List[pd.Series]:
    """Generate n_paths moving-block-bootstrapped return paths."""
    rng = np.random.default_rng(seed)
    values = returns.dropna().values
    n = len(values)
    if block_size <= 0 or block_size > n:
        raise ValueError(f"block_size {block_size} must be in (0, {n}]")
    n_blocks = int(np.ceil(n / block_size))
    paths = []
    for _ in range(n_paths):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        chunks = [values[s:s + block_size] for s in starts]
        concatenated = np.concatenate(chunks)[:n]
        paths.append(pd.Series(concatenated))
    return paths


def bootstrap_metrics(returns: pd.Series, block_size: int, n_paths: int,
                      seed: int = 0) -> pd.DataFrame:
    """Run bootstrap, compute summary metrics per simulated path."""
    rows = []
    for p in bootstrap_paths(returns, block_size, n_paths, seed=seed):
        s = summary(p)
        rows.append({
            "ann_return": s["ann_return"],
            "ann_vol": s["ann_vol"],
            "sharpe": s["sharpe"],
            "max_drawdown": s["max_drawdown"],
        })
    return pd.DataFrame(rows)


def var_and_es(returns: pd.Series, alpha: float = 0.01) -> Tuple[float, float]:
    """1%-VaR and Expected Shortfall of the daily return distribution.

    VaR at alpha = the (alpha)-quantile (typically negative).
    ES = mean return below the VaR quantile.
    """
    r = returns.dropna().values
    v = float(np.quantile(r, alpha))
    es = float(r[r <= v].mean()) if (r <= v).any() else float("nan")
    return v, es


def confidence_intervals(metrics: pd.DataFrame,
                         alphas: Sequence[float] = (0.05, 0.95)) -> pd.DataFrame:
    """Per-column percentile bands across bootstrap simulations."""
    out = {}
    for col in metrics.columns:
        out[col] = {f"p{int(a*100):02d}": float(metrics[col].quantile(a))
                     for a in alphas}
        out[col]["mean"] = float(metrics[col].mean())
        out[col]["median"] = float(metrics[col].median())
    return pd.DataFrame(out).T
```

- [ ] **Step 1.4: Run — verify PASS**

Run: `.venv/bin/pytest tests/test_bootstrap.py -v`
Expected: 4 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add src/vrp/util/bootstrap.py tests/test_bootstrap.py
git commit -m "vrp: moving-block bootstrap + tail statistics (Phase 5 Task 1)"
```

---

## Task 2: October 1987 Stress Test

**Files:**
- Create: `src/vrp/analysis/__init__.py` (empty)
- Create: `src/vrp/analysis/stress_1987.py`
- Create: `tests/test_stress_1987.py`

### Design

Black Monday scenario:
- SPX: −20.5% single-day drop
- VIX-equivalent: spike from ~20 to ~50 (the historical back-fit)
- VX futures term structure inverts sharply: front +30 (from 20 to 50), second +20 (from 22 to 42)

Apply per construction on a *synthetic* single trading day:

- **Strategy A (calendar):** `PnL = sign_front * (1/front_0) * ΔV_front + sign_second * (1/second_0) * ΔV_second`, on gross capital of 2.
- **Strategy B/C (put-writer):** revalue the short put (and optional long leg) at post-crash S and IV, compute MTM change as % of K_short capital.
- **Overlays on top of C:** regime filter lags (no pre-crash signal under a surprise shock), so no cash-out happens on day 0; tail hedge MTM gains help. We compute net.

The point is illustrative, not predictive. Show each construction's single-day PnL under the extrapolated shock, plus what the drawdown would look like if the strategy was sitting at equilibrium before the event.

- [ ] **Step 2.1: Write failing test**

`tests/test_stress_1987.py`:

```python
import pandas as pd

from vrp.analysis.stress_1987 import (
    stress_calendar,
    stress_put_writer,
    STRESS_SPX_DROP,
    STRESS_VIX_JUMP,
)


def test_constants_match_spec():
    assert abs(STRESS_SPX_DROP - (-0.205)) < 1e-3
    assert STRESS_VIX_JUMP >= 20


def test_stress_calendar_short_front_hit():
    # Baseline Strategy A: short front, long second. Vol spike -> front
    # moves up more than second -> short leg loses, long leg gains less -> big loss.
    result = stress_calendar(front_0=20, second_0=22,
                              delta_front=30, delta_second=20,
                              direction="short_front")
    assert result["daily_pnl_return"] < -0.2  # at least -20% intraday


def test_stress_calendar_long_front_gain():
    # Flipped variant gains on the same shock, mirror of short_front.
    result = stress_calendar(front_0=20, second_0=22,
                              delta_front=30, delta_second=20,
                              direction="long_front")
    assert result["daily_pnl_return"] > 0.2


def test_stress_put_writer_short_heavily_negative():
    # Short -0.30Δ put with S=100, K~95, IV=20% pre-crash.
    # Post-crash: S=80, IV=50%. Put goes deep ITM.
    r = stress_put_writer(S0=100, K_short=95, sigma0_pct=20,
                           S_shock=80, sigma_shock_pct=50,
                           T_remaining_days=20,
                           K_long=None, premium_collected=2.0)
    assert r["pnl_return"] < -0.05  # at least 5% loss


def test_stress_put_writer_spread_less_bad():
    naked = stress_put_writer(S0=100, K_short=95, sigma0_pct=20,
                               S_shock=80, sigma_shock_pct=50,
                               T_remaining_days=20, K_long=None,
                               premium_collected=2.0)
    spread = stress_put_writer(S0=100, K_short=95, sigma0_pct=20,
                                S_shock=80, sigma_shock_pct=50,
                                T_remaining_days=20, K_long=85,
                                premium_collected=1.5)
    # Spread caps the loss; its |pnl_return| should be smaller
    assert spread["pnl_return"] > naked["pnl_return"]
```

- [ ] **Step 2.2: Run — verify FAIL**

Run: `.venv/bin/pytest tests/test_stress_1987.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 2.3: Write implementation**

`src/vrp/analysis/stress_1987.py`:

```python
"""October 1987 stress test — extrapolated single-day scenario.

Black Monday 1987 hard numbers (historical record):
- S&P 500: -20.5% single-day drop
- VIX did not exist in 1987; 30-day realized vol spiked from ~15 to ~60
  in the days around the event. Hull (2017) and academic
  reconstructions estimate a "VIX-equivalent" jump of +25 to +35 vol
  points on Oct 19.

We model the scenario as:
- Front-month VX: +30 vol points (larger response)
- Second-month VX: +20 vol points (smaller, typical term-structure
  compression in a stress event)
- SPX spot: -20.5%
- ATM IV (VIX): from ~20 to ~50

The goal is *illustrative*, not predictive. Strategy-A-style short-vol
constructions would have been wiped out; hedged put-spread
constructions would have been injured but survived.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional

from vrp.util.bs import bs_price


STRESS_SPX_DROP = -0.205
STRESS_VIX_JUMP = 30.0


def stress_calendar(front_0: float, second_0: float,
                    delta_front: float, delta_second: float,
                    direction: Literal["short_front", "long_front"] = "short_front",
                    gross_notional_per_leg: float = 1.0) -> Dict[str, float]:
    """Single-day PnL of a dollar-neutral VX calendar under the stress.

    Position sizes set at front_0/second_0; PnL uses delta_front/second.
    """
    front_qty = gross_notional_per_leg / front_0
    second_qty = gross_notional_per_leg / second_0
    if direction == "short_front":
        front_sign, second_sign = -1.0, +1.0
    else:
        front_sign, second_sign = +1.0, -1.0
    pnl = (front_sign * front_qty * delta_front
            + second_sign * second_qty * delta_second)
    gross = 2.0 * gross_notional_per_leg
    return {
        "direction": direction,
        "daily_pnl": float(pnl),
        "daily_pnl_return": float(pnl / gross),
        "front_change": float(delta_front),
        "second_change": float(delta_second),
    }


def stress_put_writer(S0: float, K_short: float, sigma0_pct: float,
                      S_shock: float, sigma_shock_pct: float,
                      T_remaining_days: int,
                      K_long: Optional[float],
                      premium_collected: float,
                      r: float = 0.0) -> Dict[str, float]:
    """Single-day MTM hit to a cash-secured put writer (with optional long leg).

    Pre-shock: short put at K_short, maybe long put at K_long, both with
    T_remaining_days on the clock, underlying S0 and IV sigma0.
    Post-shock: underlying S_shock, IV sigma_shock; same T_remaining_days
    (the shock is instantaneous).

    PnL is expressed as a return on K_short of gross capital.
    """
    T = T_remaining_days / 365.0
    sigma0 = sigma0_pct / 100.0
    sigma_shock = sigma_shock_pct / 100.0

    short_pre = bs_price(S0, K_short, T, sigma0, r, "put")
    short_post = bs_price(S_shock, K_short, T, sigma_shock, r, "put")
    # Short position: we collected short_pre in premium; current liability is short_post.
    short_pnl = -(short_post - short_pre)

    long_pnl = 0.0
    if K_long is not None:
        long_pre = bs_price(S0, K_long, T, sigma0, r, "put")
        long_post = bs_price(S_shock, K_long, T, sigma_shock, r, "put")
        long_pnl = long_post - long_pre

    total_pnl = short_pnl + long_pnl
    return {
        "short_pre": float(short_pre),
        "short_post": float(short_post),
        "long_pre": float(long_pre) if K_long is not None else 0.0,
        "long_post": float(long_post) if K_long is not None else 0.0,
        "premium_collected": float(premium_collected),
        "pnl": float(total_pnl),
        "pnl_return": float(total_pnl / K_short),
    }
```

- [ ] **Step 2.4: Run — verify PASS**

Run: `.venv/bin/pytest tests/test_stress_1987.py -v`
Expected: 4 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add src/vrp/analysis/__init__.py src/vrp/analysis/stress_1987.py tests/test_stress_1987.py
git commit -m "vrp: October 1987 extrapolated stress test (Phase 5 Task 2)"
```

---

## Task 3: Combined Tail-Risk Runner

**Files:**
- Create: `scripts/run_tail_risk.py`

- [ ] **Step 3.1: Write runner**

`scripts/run_tail_risk.py`:

```python
"""Phase 5 — tail-risk analysis across the major constructions.

For each construction, compute:
1. Moving-block bootstrap with block_size=40 and n_paths=2000, report
   percentile bands on Sharpe, ann. return, and max drawdown.
2. Daily 1%-VaR and Expected Shortfall.
3. October 1987 extrapolated stress: single-day PnL as % of capital.

Constructions:
- A short-front / long-second (spec)
- A long-front / short-second (flipped)
- B PUT index (canonical)
- B synthetic spread (-0.30/-0.10)
- C spread, threshold = -2
- C spread + Overlay 1 (regime filter)  [Phase 4 winner]

Outputs:
    reports/tail_risk/bootstrap_ci.json
    reports/tail_risk/var_es.json
    reports/tail_risk/stress_1987.json
    reports/tail_risk/bootstrap_sharpe_histogram.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.analysis.stress_1987 import (
    stress_calendar, stress_put_writer,
    STRESS_SPX_DROP, STRESS_VIX_JUMP,
)
from vrp.data.cboe_indices import load_cboe_index
from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.data.vx_futures import load_vx_continuous
from vrp.overlays.regime_filter import vix_regime_mask, apply_mask
from vrp.report.metrics import summary
from vrp.strategies.strategy_a import run_strategy_a
from vrp.strategies.strategy_b import run_strategy_b
from vrp.strategies.strategy_c import run_strategy_c
from vrp.util.bootstrap import (
    bootstrap_metrics, var_and_es, confidence_intervals,
)
from vrp.util.vol import close_to_close_rv
from vrp.util.vrp_signal import compute_vrp


BLOCK_SIZE = 40
N_PATHS = 2000


def _bootstrap_block(ret: pd.Series) -> dict:
    metrics = bootstrap_metrics(ret, block_size=BLOCK_SIZE,
                                  n_paths=N_PATHS, seed=42)
    ci = confidence_intervals(metrics, alphas=(0.05, 0.50, 0.95))
    return ci.to_dict(orient="index")


def _var_es_block(ret: pd.Series) -> dict:
    v1, es1 = var_and_es(ret, alpha=0.01)
    v5, es5 = var_and_es(ret, alpha=0.05)
    return {"var_1pct": v1, "es_1pct": es1,
             "var_5pct": v5, "es_5pct": es5}


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "tail_risk"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    spx = load_spx(start="2013-01-01", end="2024-12-31")["close"]
    vix = load_vix(start="2013-01-01", end="2024-12-31")
    vx = load_vx_continuous(start="2013-01-01", end="2024-12-31")
    rv = close_to_close_rv(spx, window=20)
    vrp = compute_vrp(vix, rv).dropna()
    put_index = load_cboe_index("PUT")
    put_daily_ret = put_index.pct_change().dropna()

    # ---- Build return series for each construction ------------------------
    construction_returns = {}

    a_short = run_strategy_a(vx, tc_bps_per_roll=1.0, direction="short_front")
    construction_returns["A_short_front"] = a_short["daily_return"]

    a_long = run_strategy_a(vx, tc_bps_per_roll=1.0, direction="long_front")
    construction_returns["A_long_front"] = a_long["daily_return"]

    construction_returns["B_PUT_index"] = put_daily_ret.loc["2013-01-01":"2024-12-31"]

    b_spread = run_strategy_b(spx, vix, target_delta=-0.30,
                               long_put_delta=-0.10, tc_pct_of_premium=0.05)
    construction_returns["B_synth_spread"] = b_spread["daily_return"]

    c_spread = run_strategy_c(spx, vix, vrp, threshold=-2.0,
                               target_delta=-0.30, long_put_delta=-0.10,
                               tc_pct_of_premium=0.05)
    construction_returns["C_spread_thr=-2"] = c_spread["daily_return"]

    mask = vix_regime_mask(vix, vx["front_settle"], vx["second_settle"])
    construction_returns["C_spread_thr=-2_plus_O1"] = apply_mask(
        c_spread["daily_return"], mask
    )

    # ---- Bootstrap confidence intervals -----------------------------------
    bootstrap_cis = {}
    var_es = {}
    full_summary = {}
    for name, ret in construction_returns.items():
        bootstrap_cis[name] = _bootstrap_block(ret)
        var_es[name] = _var_es_block(ret)
        full_summary[name] = summary(ret)
    (out_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap_cis, indent=2))
    (out_dir / "var_es.json").write_text(json.dumps(var_es, indent=2))
    (out_dir / "full_sample_summary.json").write_text(
        json.dumps(full_summary, indent=2)
    )

    # ---- 1987 stress ------------------------------------------------------
    # Reasonable pre-shock levels: front VX=20, second=22, S=100, K_short=95 (-0.30Δ),
    # K_long=85 (-0.10Δ), IV=20%, 20 days to expiry (mid-cycle).
    stress = {}
    stress["A_short_front"] = stress_calendar(
        front_0=20, second_0=22, delta_front=STRESS_VIX_JUMP,
        delta_second=STRESS_VIX_JUMP * 2 / 3, direction="short_front"
    )
    stress["A_long_front"] = stress_calendar(
        front_0=20, second_0=22, delta_front=STRESS_VIX_JUMP,
        delta_second=STRESS_VIX_JUMP * 2 / 3, direction="long_front"
    )
    stress["B_synth_naked"] = stress_put_writer(
        S0=100, K_short=95, sigma0_pct=20,
        S_shock=100 * (1 + STRESS_SPX_DROP),
        sigma_shock_pct=50,
        T_remaining_days=20, K_long=None, premium_collected=2.0,
    )
    stress["B_synth_spread"] = stress_put_writer(
        S0=100, K_short=95, sigma0_pct=20,
        S_shock=100 * (1 + STRESS_SPX_DROP),
        sigma_shock_pct=50,
        T_remaining_days=20, K_long=85, premium_collected=1.5,
    )
    # C_spread_thr=-2 and the same with O1 both use the spread structure.
    # Under a surprise crash, O1's regime filter cannot fire pre-event
    # (VIX < 30 yesterday), so day-0 PnL is identical to the ungated spread.
    # Filter kicks in on day 1; we approximate that reducing further losses
    # to 0 after day 0. For the single-day comparison, we take the B-spread
    # loss number.
    stress["C_spread_thr=-2"] = stress["B_synth_spread"]
    stress["C_spread_thr=-2_plus_O1"] = {
        **stress["B_synth_spread"],
        "note": "Day-0 PnL = spread loss; O1 cashes out day 1+ (approx).",
    }
    (out_dir / "stress_1987.json").write_text(
        json.dumps(stress, indent=2, default=str)
    )

    # ---- Figure: bootstrap Sharpe histograms ------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, (name, ret) in zip(axes.flat, construction_returns.items()):
        ms = bootstrap_metrics(ret, block_size=BLOCK_SIZE, n_paths=N_PATHS, seed=42)
        ax.hist(ms["sharpe"].dropna(), bins=40, alpha=0.7)
        ax.axvline(summary(ret)["sharpe"], linestyle="--", color="red",
                    label="in-sample")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Bootstrap Sharpe")
        ax.legend(fontsize=8)
    fig.suptitle(f"Bootstrap Sharpe distribution (block={BLOCK_SIZE}, n={N_PATHS})")
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_sharpe_histogram.png", dpi=140,
                 bbox_inches="tight")
    plt.close(fig)

    print(f"Tail-risk outputs written to {out_dir}")
    print("\nBootstrap Sharpe CIs (5%-95%):")
    for name, ci in bootstrap_cis.items():
        s = ci.get("sharpe", {})
        print(f"  {name:30s}  p05={s.get('p05', 0):+.3f} "
              f"median={s.get('p50', 0):+.3f} p95={s.get('p95', 0):+.3f}")
    print("\n1% VaR and ES (daily):")
    for name, vd in var_es.items():
        print(f"  {name:30s}  VaR={vd['var_1pct']:+.4f}  ES={vd['es_1pct']:+.4f}")
    print("\n1987 single-day stress:")
    for name, s in stress.items():
        key = "daily_pnl_return" if "daily_pnl_return" in s else "pnl_return"
        print(f"  {name:30s}  {key}={s[key]:+.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run**

Run: `.venv/bin/python scripts/run_tail_risk.py`
Expected: produces JSON + one figure. Prints bootstrap CIs, VaR/ES, and 1987 stress PnLs per construction.

- [ ] **Step 3.3: Commit**

```bash
git add scripts/run_tail_risk.py
git commit -m "vrp: Phase 5 tail-risk runner — bootstrap + 1987 stress (Phase 5 Task 3)"
```

---

## Task 4: Phase 5 README Section

**Files:**
- Modify: `src/vrp/README.md` — add a "Tail-Risk Analysis" section before Limitations.

- [ ] **Step 4.1: Edit README**

Add:

```markdown
## Tail-Risk Analysis (Phase 5)

This section delivers the spec's required tail-risk block: a moving-
block bootstrap of strategy returns (preserving vol clustering), an
October 1987 extrapolated stress test, and an explicit discussion of
what the backtest cannot tell us about the tails.

### Moving-Block Bootstrap

Method: fixed-block MBB (Kunsch 1989), block size 40 trading days
(middle of the spec's 20-60 range). 2000 simulations per construction.
Within a block, autocorrelation and vol clustering are preserved; across
blocks they are not, so the method is a lower bound on true tail risk
for a strategy whose vol events cluster at >40-day horizons.

Per-construction bootstrap Sharpe 5/50/95 percentiles:

| construction | Sharpe p05 | Sharpe p50 | Sharpe p95 | in-sample |
|---|---|---|---|---|
| A short-front       | <fill> | <fill> | <fill> | <fill> |
| A long-front        | <fill> | <fill> | <fill> | <fill> |
| B PUT index         | <fill> | <fill> | <fill> | <fill> |
| B synth spread      | <fill> | <fill> | <fill> | <fill> |
| C spread thr=-2     | <fill> | <fill> | <fill> | <fill> |
| C spread thr=-2 +O1 | <fill> | <fill> | <fill> | <fill> |

### Daily VaR and Expected Shortfall

At alpha = 1%, daily return distribution (full sample):

| construction | 1% VaR | 1% ES |
|---|---|---|
| A short-front       | <fill> | <fill> |
| A long-front        | <fill> | <fill> |
| B PUT index         | <fill> | <fill> |
| B synth spread      | <fill> | <fill> |
| C spread thr=-2     | <fill> | <fill> |
| C spread thr=-2 +O1 | <fill> | <fill> |

### October 1987 Stress Test

Scenario (hand-picked from historical record, not tuned): SPX -20.5%
intraday, VIX-equivalent +30 vol points, VX curve inverts (front +30,
second +20). Pre-shock: front=20, second=22, S=100, IV=20%, K_short=95,
K_long=85, 20 days remaining on the monthly cycle.

Single-day PnL as % of capital:

| construction | single-day PnL |
|---|---|
| A short-front       | <fill> |
| A long-front        | <fill> |
| B synth naked       | <fill> |
| B synth spread      | <fill> |
| C spread thr=-2     | <fill> |
| C spread + O1       | <fill> (same as spread on day 0; O1 cashes out day 1+) |

### Honest limitations

1. Our sample contains at most ~4 major short-vol-blow-up events
   (2015 flash crash, Feb 2018 Volmageddon, March 2020 COVID, 2022
   bear). That is nowhere near enough realizations to characterize
   the true tail — bootstrap CIs on Sharpe and MDD should be read as
   *in-distribution* tail estimates conditional on the observed
   sample, not as unbiased population statistics.
2. The 1987 scenario is extrapolated. Real 1987 parameters are not
   known with certainty (VIX did not exist; IV levels are back-fit
   estimates), and option-market microstructure in 1987 differed
   materially from the present (no VIX futures, no electronic
   execution, limited put depth).
3. The MBB breaks cross-block autocorrelation. For strategies whose
   drawdowns cluster at horizons >40 trading days (a plausible
   property of naive short-vol), bootstrap MDD CIs understate true
   tail-DD risk.
4. Transaction costs and liquidity constraints during crises are not
   modeled. Strategy B/C's BS-based mark-to-market assumes continuous
   hedging at theoretical prices; in 1987 put-option markets, you
   could not exit at theoretical prices.
5. This strategy is short volatility. It will lose money
   catastrophically in a vol spike. Everything above is a rigorous
   characterization of *how catastrophically*, not a claim of safety.
```

Fill `<fill>` from the JSON outputs of `scripts/run_tail_risk.py`.

- [ ] **Step 4.2: Commit**

```bash
git add src/vrp/README.md
git commit -m "vrp: Phase 5 README — tail-risk analysis and limitations (Phase 5 Task 4)"
```

---

## Phase 5 Definition of Done

1. `.venv/bin/pytest` passes (51 from Phase 4 + 4 bootstrap + 4 stress = 59 tests).
2. `scripts/run_tail_risk.py` produces JSON + figure.
3. README has the Tail-Risk Analysis section with filled numbers and limitations.
4. Four task commits present.

## Self-Review

- **Spec coverage:** all three required tail-risk items (block bootstrap, 1987 stress test, explicit short-vol disclosure) are addressed.
- **Placeholders:** all steps concrete; README has fill-from-JSON directives.
- **Type consistency:** `bootstrap_paths`/`bootstrap_metrics` output shapes consistent; stress functions return flat dicts.
- **TDD:** T1 and T2 strict red/green. T3 runner is exercised by produced JSON.
- **Scope:** Phase 5 is the finale. No Phase 6 is planned.
