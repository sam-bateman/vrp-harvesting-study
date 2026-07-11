import pandas as pd

from vrp.overlays.regime_filter import vix_regime_mask, apply_mask


def _synth_data(n=100, vix_level=15.0, front=15.0, second=16.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return (pd.Series(vix_level, index=idx),
            pd.Series(front, index=idx),
            pd.Series(second, index=idx))


def test_mask_all_active_in_quiet_regime():
    vix, front, second = _synth_data(vix_level=15.0)
    mask = vix_regime_mask(vix, front, second)
    assert mask.all()


def test_mask_exit_is_lagged_one_day():
    """Stress first observable at close of day 20 -> day 20's own return
    is still earned (exit trades at that close); flat from day 21."""
    vix, front, second = _synth_data(n=50, vix_level=15.0)
    vix.iloc[20:30] = 35.0
    mask = vix_regime_mask(vix, front, second)
    assert mask.iloc[20]          # crash day itself cannot be dodged
    assert not mask.iloc[21:31].any()


def test_mask_reentry_is_lagged_one_day():
    """Calm confirmed at the close of the 7th calm day (day 36 when the
    spike ends after day 29) -> exposure resumes with day 37's return."""
    vix, front, second = _synth_data(n=50, vix_level=15.0)
    vix.iloc[20:30] = 35.0
    mask = vix_regime_mask(vix, front, second)
    assert not mask.iloc[31:37].any()
    assert mask.iloc[37:].all()


def test_mask_turns_off_on_backwardation():
    vix, front, second = _synth_data(n=50)
    front.iloc[10:20] = 20.0
    second.iloc[10:20] = 17.0
    mask = vix_regime_mask(vix, front, second)
    # Lagged one day: inversion observed at close 10 -> flat from day 11.
    assert mask.iloc[10]
    assert not mask.iloc[11:20].any()


def test_first_bar_is_exposed():
    vix, front, second = _synth_data(n=10, vix_level=40.0)
    mask = vix_regime_mask(vix, front, second)
    # No signal exists before the first close; the strategy starts
    # invested and exits at the first close (flat from bar 1).
    assert mask.iloc[0]
    assert not mask.iloc[1:].any()


def test_apply_mask_zeros_daily_returns():
    idx = pd.bdate_range("2020-01-02", periods=20)
    daily_return = pd.Series(0.01, index=idx)
    mask = pd.Series([True] * 10 + [False] * 10, index=idx)
    out = apply_mask(daily_return, mask)
    assert (out.iloc[:10] == 0.01).all()
    assert (out.iloc[10:] == 0.0).all()


def test_apply_mask_days_outside_mask_coverage_stay_invested():
    """Days before the mask's data coverage begins must not be forced to
    cash — the filter cannot act on days it cannot observe."""
    idx = pd.bdate_range("2020-01-02", periods=20)
    daily_return = pd.Series(0.01, index=idx)
    mask = pd.Series(True, index=idx[10:])  # coverage starts at bar 10
    out = apply_mask(daily_return, mask)
    assert (out == 0.01).all()
