import pandas as pd
import pytest

from vrp.strategies.strategy_b import run_strategy_b
from vrp.strategies.strategy_c import run_strategy_c


def _synth_spx_vix(n=252, spx=100.0, vix=20.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return (pd.Series(spx, index=idx),
            pd.Series(vix, index=idx))


def _synth_vrp_alternating(idx):
    vrp = pd.Series(0.0, index=idx)
    for i, d in enumerate(idx):
        vrp.iloc[i] = 5.0 if (d.month % 2 == 0) else -2.0
    return vrp


def test_strategy_c_matches_b_when_threshold_low():
    spx, vix = _synth_spx_vix()
    # VRP history must exist BEFORE each month-open for the gate to act;
    # extend it back so even the first month has a prior-month-end value.
    vrp_idx = pd.bdate_range("2019-12-02", periods=20).append(spx.index)
    vrp = pd.Series(10.0, index=vrp_idx)
    b = run_strategy_b(spx, vix, target_delta=-0.30)
    c = run_strategy_c(spx, vix, vrp, threshold=0.0, target_delta=-0.30)
    pd.testing.assert_series_equal(b["daily_return"], c["daily_return"],
                                   check_names=False)


def test_strategy_c_gates_on_prior_month_end_vrp_only():
    """The gate must consume the last VRP value STRICTLY BEFORE the
    month-open day. A high VRP appearing on the month-open day itself
    must not rescue a month whose prior-month-end VRP is below
    threshold (that information arrives only at that day's close)."""
    spx, vix = _synth_spx_vix(n=64)  # ~3 months
    vrp = pd.Series(-10.0, index=spx.index)
    month_open_2 = spx.index[spx.index.month == 2][0]
    vrp.loc[month_open_2] = +10.0  # visible only at that day's close
    c = run_strategy_c(spx, vix, vrp, threshold=0.0, target_delta=-0.30)
    gating = c["gating"].set_index("open_date")
    assert not gating.loc[month_open_2, "active"]
    # ... and the value the gate saw is the prior day's, not +10.
    assert gating.loc[month_open_2, "vrp"] == -10.0


def test_strategy_c_first_month_without_history_is_cash():
    spx, vix = _synth_spx_vix(n=64)
    vrp = pd.Series(10.0, index=spx.index)  # no history before day 1
    c = run_strategy_c(spx, vix, vrp, threshold=0.0, target_delta=-0.30)
    first_open = c["gating"]["open_date"].iloc[0]
    assert not c["gating"].set_index("open_date").loc[first_open, "active"]


def test_strategy_c_all_cash_when_threshold_high():
    spx, vix = _synth_spx_vix()
    vrp = pd.Series(0.0, index=spx.index)
    c = run_strategy_c(spx, vix, vrp, threshold=100.0, target_delta=-0.30)
    assert (c["daily_return"] == 0.0).all()
    assert c["active_months_fraction"] == 0.0


def test_strategy_c_active_fraction():
    spx, vix = _synth_spx_vix(n=252)
    vrp = _synth_vrp_alternating(spx.index)
    c = run_strategy_c(spx, vix, vrp, threshold=0.0, target_delta=-0.30)
    assert 0.3 < c["active_months_fraction"] < 0.7


def test_strategy_c_invalid_delta_raises():
    spx, vix = _synth_spx_vix()
    vrp = pd.Series(5.0, index=spx.index)
    with pytest.raises(ValueError):
        run_strategy_c(spx, vix, vrp, threshold=0.0, target_delta=0.30)
