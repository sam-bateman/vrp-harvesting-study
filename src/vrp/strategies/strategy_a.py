"""Strategy A — VIX futures term-structure calendar (dollar-neutral).

A dollar-neutral VIX-futures calendar spread with a 5-trading-day
pre-expiry roll. Two directions:

- ``short_front`` (the spec's direction): short front-month VX, long
  second-month VX. Captures ``(r_second − r_front) / 2`` daily on gross
  capital. Losing in practice — see the Phase 1 README for the analysis.
- ``long_front``: long front-month VX, short second-month VX. Captures
  ``(r_front − r_second) / 2`` daily.

PnL uses the *held-contract* return columns from
``vrp.data.vx_futures.load_vx_continuous`` (``held_front_ret`` /
``held_second_ret``): each daily return is computed within a single
contract and the held pair rolls ``roll_days_before_expiry`` trading
days before front expiry. Diffing the spliced continuous settle series
across a roll would book the front/second calendar gap as phantom PnL
once a month; that defect was present in the original Phase 1 engine
and inflated the long-front direction.

See:
- Whaley (2013) "Trading Volatility: At What Cost?"
- Alexander, Korovilas (2013) for a critique of naive VX calendars.

This implementation targets gross notional = $1 per side (so $2 gross),
renormalized daily. Interpret ``daily_pnl`` as dollars on that unit
capital, and ``daily_return`` as the daily return on $2 of gross capital.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import pandas as pd


ROLL_DAYS_BEFORE_EXPIRY = 5

Direction = Literal["short_front", "long_front"]


@dataclass
class StrategyAConfig:
    tc_bps_per_roll: float = 1.0
    gross_notional_per_leg: float = 1.0
    direction: Direction = "short_front"


def _leg_signs(direction: Direction) -> tuple[float, float]:
    """Return (front_sign, second_sign) for a given calendar direction."""
    if direction == "short_front":
        return -1.0, +1.0
    if direction == "long_front":
        return +1.0, -1.0
    raise ValueError(
        f"direction must be 'short_front' or 'long_front', got {direction!r}"
    )


def run_strategy_a(vx: pd.DataFrame,
                   tc_bps_per_roll: float = 1.0,
                   gross_notional_per_leg: float = 1.0,
                   direction: Direction = "short_front") -> Dict[str, pd.Series]:
    """Run Strategy A on a continuous VX frame from ``load_vx_continuous``.

    Parameters
    ----------
    vx:
        Index = date. Must include ``held_front_ret``, ``held_second_ret``
        and ``is_roll_day`` (the roll schedule is decided in the data
        layer via its ``roll_days_before_expiry`` parameter).
    tc_bps_per_roll:
        Basis points of slippage per leg traded at a roll. A roll closes
        two legs and opens two, so 4x the per-leg notional trades at each
        roll event. 1 bp is the baseline estimate.
    gross_notional_per_leg:
        Dollar notional per leg. Gross capital is ``2 * this`` by
        construction (dollar-neutral).
    direction:
        ``"short_front"`` (default, spec's direction) or ``"long_front"``.

    Returns
    -------
    Dict with keys ``daily_pnl``, ``daily_return``, ``positions``.
    """
    required = {"held_front_ret", "held_second_ret", "is_roll_day"}
    missing = required - set(vx.columns)
    if missing:
        raise ValueError(
            f"vx frame missing columns {sorted(missing)}; rebuild it with "
            f"vrp.data.vx_futures.load_vx_continuous"
        )

    cfg = StrategyAConfig(tc_bps_per_roll=tc_bps_per_roll,
                          gross_notional_per_leg=gross_notional_per_leg,
                          direction=direction)

    df = vx.copy().sort_index()
    front_sign, second_sign = _leg_signs(cfg.direction)

    leg = cfg.gross_notional_per_leg
    gross = 2.0 * leg
    daily_pnl = (front_sign * leg * df["held_front_ret"]
                 + second_sign * leg * df["held_second_ret"]).fillna(0.0)

    # Close 2 legs + open 2 legs at each roll: 4x per-leg notional trades.
    tc = df["is_roll_day"].astype(float) * (cfg.tc_bps_per_roll * 1e-4) * (
        4 * leg
    )
    daily_pnl = daily_pnl - tc

    daily_return = daily_pnl / gross

    positions = pd.DataFrame({
        "front_sign": front_sign,
        "second_sign": second_sign,
        "is_roll_day": df["is_roll_day"].astype(bool),
    }, index=df.index)

    return {
        "daily_pnl": daily_pnl,
        "daily_return": daily_return,
        "positions": positions,
    }
