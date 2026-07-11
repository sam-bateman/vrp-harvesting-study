"""Benchmark series for alpha/beta attribution.

- ``load_spx_total_return``: S&P 500 Total Return index (^SP500TR).
  The CBOE PUT index is a total-return construct (premium reinvested,
  collateral earns T-bills), so regressing it against the price-only
  ^GSPC overstates alpha by roughly the dividend yield times beta.
- ``load_rf_daily``: daily risk-free rate proxy from ^IRX (13-week
  T-bill discount yield, annualized percent). The de-annualization is
  approximate (yield/252) — adequate for excess-return regressions at
  the precision this study reports.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import cache
from .spx import _inclusive_end

_TR_KEY_PREFIX = "spx_tr_daily"
_RF_KEY_PREFIX = "rf_irx_daily"


def load_spx_total_return(start: str = "2005-01-01",
                          end: str | None = None,
                          use_cache: bool = True) -> pd.Series:
    key = f"{_TR_KEY_PREFIX}__{start}__{end or 'live'}"
    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            return cached["spx_tr"]

    raw = yf.download("^SP500TR", start=start, end=_inclusive_end(end),
                      auto_adjust=False, progress=False, group_by="column")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].rename("spx_tr").dropna()
    s.index.name = "date"
    cache.save(key, s.to_frame())
    return s


def load_rf_daily(start: str = "2005-01-01",
                  end: str | None = None,
                  use_cache: bool = True) -> pd.Series:
    """Daily simple risk-free rate from the 13-week T-bill yield (^IRX)."""
    key = f"{_RF_KEY_PREFIX}__{start}__{end or 'live'}"
    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            return cached["rf_daily"]

    raw = yf.download("^IRX", start=start, end=_inclusive_end(end),
                      auto_adjust=False, progress=False, group_by="column")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    ann_pct = raw["Close"].dropna()
    s = (ann_pct / 100.0 / 252.0).rename("rf_daily")
    s.index.name = "date"
    cache.save(key, s.to_frame())
    return s
