import numpy as np
import pandas as pd

from vrp.util.annualize import (
    TRADING_DAYS,
    ann_return,
    ann_vol,
    sharpe_ratio,
)


def test_trading_days_constant():
    assert TRADING_DAYS == 252


def test_ann_vol_sqrt_scaling():
    rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 10_000))
    ann = ann_vol(rets)
    assert abs(ann - 0.01 * np.sqrt(252)) < 5e-4


def test_ann_return_compounded():
    rets = pd.Series([0.001] * 252)
    expected = (1.001 ** 252) - 1
    assert abs(ann_return(rets) - expected) < 1e-9


def test_sharpe_zero_rf():
    rets = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 5_000))
    s = sharpe_ratio(rets, rf=0.0)
    mu_ann = ann_return(rets)
    sig_ann = ann_vol(rets)
    assert abs(s - mu_ann / sig_ann) < 1e-9


def test_sharpe_with_rf_subtracts_annual_rate():
    rets = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 5_000))
    s0 = sharpe_ratio(rets, rf=0.0)
    s2 = sharpe_ratio(rets, rf=0.02)
    assert abs((s0 - s2) * ann_vol(rets) - 0.02) < 1e-9


def test_sharpe_zero_vol_is_signed_infinite():
    # All-zero returns give an exactly-zero std (constant nonzero series
    # carry ~1e-19 float noise and never reach the degenerate branch).
    flat = pd.Series([0.0] * 252)
    assert sharpe_ratio(flat, rf=-0.02) == float("inf")
    # Zero-vol with negative excess return is -inf, not nan: the sign
    # of the excess return is known even when vol is degenerate.
    assert sharpe_ratio(flat, rf=0.02) == float("-inf")
