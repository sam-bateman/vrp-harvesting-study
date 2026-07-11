"""Overlay 1 — VIX regime filter.

Go to cash when VIX spot > 30 OR VIX term structure inverts
(front-month VX > second-month VX). Re-enter when VIX < 25 AND term
structure is back in contango, confirmed for 7 consecutive trading days
(anti-whipsaw).

Timing: the stress/calm state is only observable at the close of day t,
so the earliest tradeable exit is at that close — which means day t's
own close-to-close return is still earned (or suffered). The mask
returned by ``vix_regime_mask`` is therefore lagged one bar: the state
decided at close t governs exposure to day t+1's return. The original
Phase 4 implementation zeroed day t itself, a one-day lookahead that
deleted exactly the crash days the filter is supposed to be unable to
dodge; that inflated the published Overlay 1 results.

Rule is hand-picked from the project spec; not tuned to data.
"""
from __future__ import annotations

import pandas as pd


HIGH_VIX = 30.0
LOW_VIX = 25.0
CONFIRMATION_DAYS = 7


def vix_regime_mask(vix_spot: pd.Series,
                    vx_front: pd.Series,
                    vx_second: pd.Series,
                    high_vix: float = HIGH_VIX,
                    low_vix: float = LOW_VIX,
                    confirm_days: int = CONFIRMATION_DAYS) -> pd.Series:
    """Daily boolean mask. True = exposed to that day's return.

    The mask at day t reflects the state decided at the close of day
    t-1 (tradeable, no lookahead). The first bar is True: the strategy
    starts invested, and no signal exists before the first close.
    """
    aligned = pd.concat([vix_spot.rename("vix"),
                          vx_front.rename("front"),
                          vx_second.rename("second")], axis=1).dropna()

    stress = (aligned["vix"] > high_vix) | (aligned["front"] > aligned["second"])
    calm = (aligned["vix"] < low_vix) & (aligned["front"] < aligned["second"])

    calm_streak = calm.astype(int).groupby(
        (~calm).cumsum()
    ).cumsum()

    # State as decided at each day's close.
    state = pd.Series(False, index=aligned.index)
    active = True
    for i, (is_stress, streak) in enumerate(zip(stress.values, calm_streak.values)):
        if active:
            if is_stress:
                active = False
        else:
            if streak >= confirm_days:
                active = True
        state.iloc[i] = active

    # Exposure to day t's return = state at close of t-1.
    mask = state.shift(1).astype("boolean").fillna(True).astype(bool)
    return mask


def apply_mask(daily_return: pd.Series, mask: pd.Series) -> pd.Series:
    """Zero out daily_return on bars where mask is False.

    Days absent from the mask index (e.g. before signal-data coverage
    begins) inherit the last known state, defaulting to exposed. The
    filter can only act on days it can observe; forcing unknown days to
    cash would silently delete strategy history.
    """
    m = mask.reindex(daily_return.index).ffill()
    m = m.astype("boolean").fillna(True).astype(bool)
    out = daily_return.where(m, 0.0)
    return out.rename(daily_return.name)
