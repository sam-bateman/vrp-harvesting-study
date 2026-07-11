"""Moving-block bootstrap and tail statistics.

Moving-block bootstrap (Kunsch 1989): resample blocks of consecutive
returns to generate simulated paths that preserve within-block
autocorrelation and volatility clustering. Breaks cross-block
autocorrelation, which is acceptable at block sizes >= 20 trading days
for the practical timescales of monthly-rebalance strategies.

References:
- Kunsch (1989) "The Jackknife and the Bootstrap for General Stationary
  Observations"
- Politis, Romano (1994) "The Stationary Bootstrap" — implemented in
  ``stationary_bootstrap_paths`` as a robustness check on the MBB's
  fixed block-size choice.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

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


def stationary_bootstrap_paths(returns: pd.Series, mean_block_size: int,
                               n_paths: int, seed: int = 0) -> List[pd.Series]:
    """Politis-Romano (1994) stationary bootstrap.

    Blocks have geometric random lengths with the given mean and wrap
    circularly, so no observation is underweighted and the resampled
    series is stationary. Unlike fixed-block MBB, the random block
    lengths avoid committing to a single dependence horizon — the
    complement to the MBB's fixed ``block_size``.
    """
    rng = np.random.default_rng(seed)
    values = returns.dropna().values
    n = len(values)
    if mean_block_size <= 0 or mean_block_size > n:
        raise ValueError(
            f"mean_block_size {mean_block_size} must be in (0, {n}]"
        )
    p = 1.0 / mean_block_size
    paths = []
    for _ in range(n_paths):
        idx = np.empty(0, dtype=int)
        while len(idx) < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            idx = np.concatenate([idx, (start + np.arange(length)) % n])
        paths.append(pd.Series(values[idx[:n]]))
    return paths


def bootstrap_metrics(returns: pd.Series, block_size: int, n_paths: int,
                      seed: int = 0, method: str = "mbb") -> pd.DataFrame:
    """Run bootstrap, compute summary metrics per simulated path.

    method: "mbb" (fixed-size moving block) or "stationary"
    (Politis-Romano, ``block_size`` is the mean block length).
    """
    if method == "mbb":
        paths = bootstrap_paths(returns, block_size, n_paths, seed=seed)
    elif method == "stationary":
        paths = stationary_bootstrap_paths(returns, block_size, n_paths,
                                           seed=seed)
    else:
        raise ValueError(f"unknown bootstrap method {method!r}")
    rows = []
    for p in paths:
        s = summary(p)
        rows.append({
            "ann_return": s["ann_return"],
            "ann_vol": s["ann_vol"],
            "sharpe": s["sharpe"],
            "max_drawdown": s["max_drawdown"],
        })
    return pd.DataFrame(rows)


def var_and_es(returns: pd.Series, alpha: float = 0.01) -> Tuple[float, float]:
    """1%-VaR and Expected Shortfall of the daily return distribution."""
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
