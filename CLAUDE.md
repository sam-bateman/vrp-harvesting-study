# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A quant research project (resume/portfolio piece) replicating and comparing three volatility-risk-premium harvesting strategies on SPX, with explicit tail-risk accounting. It is framed as a replication + comparative study, **not** a novel-alpha claim. `README.md` is the results document — when a phase's results change, the README tables and narrative must be updated to match. The design spec lives in `docs/specs/`, phase plans in `docs/plans/`.

## Commands

```bash
pip install -e '.[dev]'          # install package + pytest/jupyter

pytest                           # run all tests (config in pyproject.toml)
pytest tests/test_bs.py          # single file
pytest tests/test_bs.py -k name  # single test

# Phase runners (each writes JSON/PNG into reports/<name>/, gitignored):
python scripts/run_strategy_a.py            # Phase 1 baseline
python scripts/sanity_vxx.py                # Phase 1 sanity gate
python scripts/run_strategy_a_sweep.py      # tc sensitivity
python scripts/run_strategy_b_putindex.py   # Phase 2 canonical (CBOE PUT)
python scripts/run_strategy_b_synthetic.py  # Phase 2 BS replication + gate
python scripts/run_strategy_b_spread.py     # Phase 2 put-spread variant
python scripts/run_strategy_c.py            # Phase 3 VRP gating
python scripts/run_strategy_c_sensitivity.py# Phase 3 train-then-test sweep
python scripts/run_strategy_c_overlays.py   # Phase 4 overlays
python scripts/run_tail_risk.py             # Phase 5 bootstrap + 1987 stress
```

There is no linter configured. Commit messages follow the existing style: `vrp: <what shipped>` (or `docs:` for docs-only), one commit per plan task.

## Architecture

Layered package under `src/vrp/` (setuptools `where = ["src"]`); scripts are thin runners that wire layers together and emit report artifacts.

- `vrp.data.*` — all I/O. `spx.py`/`vix.py` pull from yfinance (`^GSPC`, `^VIX`; `end` is inclusive — loaders adjust for yfinance's exclusive end); `benchmarks.py` adds `^SP500TR` and `^IRX` for alpha regressions; `vx_futures.py` stitches per-contract CBOE CDN CSVs (coverage 2013+; pre-2013 returns HTTP 403; `VX_CSV_OVERRIDE` env var points to local per-contract CSVs as a fallback). VX expiries are holiday-adjusted (CFE rule — Good Friday Aprils, Juneteenth 2024) and early-2013 rows fall back from Settle=0 to Close. The continuous frame carries two designations: spliced `front_settle`/`second_settle` (signals only — diffing them across a roll books phantom PnL) and splice-free `held_front_ret`/`held_second_ret` (PnL; pair rolls 5 trading days pre-expiry). `cboe_indices.py` loads PUT/BXM. Everything caches through `cache.py` — a parquet store under `data/vrp_cache/` keyed by string, with **no TTL/invalidation**: callers must vary the key (and bump the key prefix when loader semantics change, as done for `_v2`).
- `vrp.util.*` — pure math, no I/O: Black-Scholes (`bs.py`), realized vol (`vol.py`), annualization constants (`annualize.py`, 252 days everywhere), VRP signal (`vrp_signal.py`), moving-block bootstrap + VaR/ES (`bootstrap.py`).
- `vrp.strategies.*` — each strategy exposes a `run_strategy_*` function that consumes price Series and returns a dict containing a `daily_return` pandas Series (simple daily returns, trading-date index). Strategy C wraps `run_strategy_b` and gates months on the VRP signal; the naked/spread distinction is a Strategy B parameter (`long_put_delta=None` vs e.g. `-0.10`).
- `vrp.overlays.*` — post-hoc transforms on a strategy's daily-return series (regime-filter mask, realized-vol scaling, tail-hedge spend).
- `vrp.report.*` — metrics (Sharpe, MDD, drawdown duration, VaR…) and regime slicing over daily-return Series.
- `vrp.analysis.stress_1987` — closed-form single-day stress PnL for each construction.

Daily-return `pd.Series` is the universal interchange format between strategies, overlays, and reporting.

## Methodological rules (non-negotiable, from the design spec)

- **No lookahead bias.** Only information available at the decision point; signals computed over `[t-20, t]` trade at `t+1`.
- **Strict train/test split:** train 2013-01-01 → 2018-12-31, test 2019-01-01 → 2024-12-31. (The spec originally targeted 2006+, reduced because CBOE VX history starts in 2013.) **Never tune any parameter on the test window** — pick on train, evaluate test once, and report it even if unflattering.
- **Transaction costs modeled explicitly** (1 bp/leg per VX roll; 5% of premium round-trip for options).
- Every design choice cites a paper or is flagged as a judgment call needing sensitivity analysis.
- Honest reporting is the point: negative Sharpes and limitations stay in the README. Inflated numbers are red flags for the target audience, not selling points.

## Sanity gates

- Strategy A daily-return correlation vs `−VXX` must be in `[0.5, 0.9]` (currently 0.777). Negative or below 0.3 → sign bug in VX PnL or data-ingestion issue.
- Synthetic Strategy B monthly-return correlation vs the CBOE PUT index must be ≥ 0.6 (currently 0.825). The synthetic BS engine is a documented approximation; the PUT index is the canonical Strategy B deliverable.
- Timing convention everywhere: anything computed from day t's close applies no earlier than day t+1's return (regime-filter masks are pre-lagged inside `vix_regime_mask`; Strategy C gates on the prior month-end VRP).

## Testing conventions

Tests are pure-unit against `src/vrp` modules with synthetic inputs — no network access and no cache dependence. Keep it that way: anything touching yfinance/CBOE belongs in `vrp.data` behind the cache, exercised via scripts, not tests.
