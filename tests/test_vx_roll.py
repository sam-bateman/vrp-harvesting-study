import numpy as np
import pandas as pd
import pytest

from vrp.data.vx_futures import (
    build_continuous_from_panel,
    build_roll_calendar,
    vx_expiration,
)
from vrp.strategies.strategy_a import run_strategy_a


def test_vx_expiration_known_dates():
    # 3rd Friday of Feb 2018 is Feb 16; SOQ Wed = 30 days before = Jan 17
    assert vx_expiration(2018, 1) == pd.Timestamp("2018-01-17")
    # 3rd Friday of Jan 2020 is Jan 17; 30 days before = Dec 18 2019
    assert vx_expiration(2019, 12) == pd.Timestamp("2019-12-18")


def test_vx_expiration_holiday_adjustments():
    # Good Friday moves the April 3rd Friday: settlement is the business
    # day before the computed Wednesday. Confirmed against CBOE CDN files.
    assert vx_expiration(2014, 3) == pd.Timestamp("2014-03-18")
    assert vx_expiration(2019, 3) == pd.Timestamp("2019-03-19")
    assert vx_expiration(2022, 3) == pd.Timestamp("2022-03-15")
    # Juneteenth 2024 lands on the Wednesday itself.
    assert vx_expiration(2024, 6) == pd.Timestamp("2024-06-18")
    # Regular months are unaffected.
    assert vx_expiration(2020, 4) == pd.Timestamp("2020-04-15")


def test_build_roll_calendar_columns_and_order():
    cal = build_roll_calendar(start="2018-01-01", end="2018-06-30")
    assert list(cal.columns) == ["front_expiry", "second_expiry"]
    diffs = cal["front_expiry"].diff().dropna()
    assert (diffs >= pd.Timedelta(0)).all()
    assert (cal["second_expiry"] > cal["front_expiry"]).all()


def _synth_panel(n: int = 60, levels=(15.0, 17.0, 19.0)):
    """Three contracts at constant price levels over n business days.

    Expiries are spaced so the panel spans one roll of the held pair.
    Constant prices mean every within-contract daily return is exactly 0,
    so ANY nonzero held return exposes a contract splice.
    """
    idx = pd.bdate_range("2020-01-02", periods=n)
    e1 = idx[n // 2] + pd.Timedelta(days=1)     # front expires mid-sample
    e2 = e1 + pd.Timedelta(days=28)
    e3 = e2 + pd.Timedelta(days=35)
    panel = pd.DataFrame(
        {e1: levels[0], e2: levels[1], e3: levels[2]}, index=idx
    )
    return panel, (e1, e2, e3)


def test_no_splice_pnl_with_constant_contract_prices():
    """Regression for the roll-splice bug: constant per-contract prices
    must produce identically zero held returns, even across the roll."""
    panel, _ = _synth_panel()
    out = build_continuous_from_panel(panel, roll_days_before_expiry=5,
                                      holidays=set())
    held = out[["held_front_ret", "held_second_ret"]].iloc[1:]
    assert (held.abs() < 1e-12).all().all()
    # The spliced market series, by contrast, does jump at the roll:
    assert out["front_settle"].diff().abs().max() > 1.0
    # ... and there is exactly one roll in the window.
    assert out["is_roll_day"].sum() == 1


def test_roll_happens_five_trading_days_before_expiry():
    panel, (e1, _, _) = _synth_panel()
    out = build_continuous_from_panel(panel, roll_days_before_expiry=5,
                                      holidays=set())
    roll_date = out.index[out["is_roll_day"]][0]
    # After the roll, the held front is no longer the e1 contract.
    assert (out.loc[out.index >= roll_date, "held_front_expiry"] > e1).all()
    # The roll lands when 5 trading days remain before e1.
    remaining = np.busday_count(
        roll_date.to_datetime64().astype("datetime64[D]")
        + np.timedelta64(1, "D"),
        e1.to_datetime64().astype("datetime64[D]"),
    )
    assert remaining == 5


def _synth_vx_frame(n: int = 40, front_ret: float = -0.001,
                    second_ret: float = 0.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    df = pd.DataFrame({
        "held_front_ret": front_ret,
        "held_second_ret": second_ret,
        "is_roll_day": False,
    }, index=idx)
    df.iloc[0, df.columns.get_loc("held_front_ret")] = np.nan
    df.iloc[0, df.columns.get_loc("held_second_ret")] = np.nan
    return df


def test_strategy_a_positive_when_front_decays():
    df = _synth_vx_frame()
    result = run_strategy_a(df, tc_bps_per_roll=0)
    # Short front decays -> positive PnL on the short leg; second flat.
    assert result["daily_pnl"].sum() > 0


def test_strategy_a_daily_return_is_half_return_spread():
    df = _synth_vx_frame(front_ret=-0.002, second_ret=0.001)
    result = run_strategy_a(df, tc_bps_per_roll=0, direction="short_front")
    # daily_return = (r_second - r_front) / 2 on $2 gross capital
    expected = (0.001 - (-0.002)) / 2.0
    assert abs(result["daily_return"].iloc[5] - expected) < 1e-12


def test_strategy_a_tc_charged_once_per_roll_on_four_legs():
    df = _synth_vx_frame(front_ret=0.0, second_ret=0.0)
    df.iloc[10, df.columns.get_loc("is_roll_day")] = True
    result = run_strategy_a(df, tc_bps_per_roll=1.0)
    # 1 bp on 4x $1 legs = 4e-4 dollars on the roll day, else zero.
    assert abs(result["daily_pnl"].iloc[10] + 4e-4) < 1e-15
    assert result["daily_pnl"].drop(df.index[10]).abs().max() < 1e-15


def test_strategy_a_long_front_flips_pnl_sign():
    df = _synth_vx_frame(front_ret=-0.002, second_ret=0.001)
    short_front = run_strategy_a(df, tc_bps_per_roll=0,
                                 direction="short_front")
    long_front = run_strategy_a(df, tc_bps_per_roll=0,
                                direction="long_front")
    pd.testing.assert_series_equal(
        long_front["daily_pnl"], -short_front["daily_pnl"],
        check_names=False,
    )


def test_strategy_a_missing_columns_raises():
    idx = pd.bdate_range("2020-01-02", periods=10)
    legacy = pd.DataFrame({"front_settle": 15.0, "second_settle": 17.0},
                          index=idx)
    with pytest.raises(ValueError, match="missing columns"):
        run_strategy_a(legacy)


def test_strategy_a_invalid_direction_raises():
    df = _synth_vx_frame()
    with pytest.raises(ValueError, match="direction must be"):
        run_strategy_a(df, direction="sideways")  # type: ignore[arg-type]
