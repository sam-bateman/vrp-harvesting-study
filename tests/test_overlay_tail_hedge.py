import pandas as pd

from vrp.overlays.tail_hedge import add_tail_hedge
from vrp.strategies.strategy_b import run_strategy_b


def _synth_spx_vix(n=252, spx_level=100.0, vix_level=20.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return (pd.Series(spx_level, index=idx),
            pd.Series(vix_level, index=idx))


def test_hedge_reduces_net_on_flat_underlying():
    spx, vix = _synth_spx_vix(252)
    strat = run_strategy_b(spx, vix, target_delta=-0.30,
                            tc_pct_of_premium=0.0)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    assert hedged["net_daily_return"].sum() < strat["daily_return"].sum()


def test_hedge_reduces_drawdown_on_crash():
    n = 60
    idx = pd.bdate_range("2020-01-02", periods=n)
    spx = pd.Series(100.0, index=idx)
    spx.iloc[n // 2:] = 80.0
    vix = pd.Series(25.0, index=idx)
    strat = run_strategy_b(spx, vix, target_delta=-0.30,
                            tc_pct_of_premium=0.0)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    strat_eq = (1 + strat["daily_return"]).cumprod()
    net_eq = (1 + hedged["net_daily_return"]).cumprod()
    low_day = strat_eq.idxmin()
    assert net_eq.loc[low_day] > strat_eq.loc[low_day]


def test_hedge_worthless_expiry_costs_exactly_the_premium():
    """Regression for the double-charged debit: on a flat underlying the
    hedge decays to ~worthless, so each cycle's total hedge PnL must be
    ~-hedge_spend (never ~-2x hedge_spend) on the K_short capital base."""
    spx, vix = _synth_spx_vix(64)  # ~3 monthly cycles
    strat = run_strategy_b(spx, vix, target_delta=-0.30,
                            tc_pct_of_premium=0.0)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    legs = hedged["hedge_legs"].set_index("open_date")
    positions = strat["positions"].set_index("open_date")
    hedge_ret = hedged["hedge_daily_return"]
    for open_date, leg in legs.iterrows():
        close_date = positions.loc[open_date, "close_date"]
        cycle = hedge_ret.loc[open_date:close_date - pd.Timedelta(days=1)]
        total = cycle.sum() * positions.loc[open_date, "K_short"]
        spend = leg["hedge_spend"]
        # Within 25% of one premium (BS decay isn't exactly zero at the
        # calendar-month close), and nowhere near two premia.
        assert -1.25 * spend < total < -0.5 * spend


def test_hedge_structure_keys():
    spx, vix = _synth_spx_vix(120)
    strat = run_strategy_b(spx, vix, target_delta=-0.30)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    assert set(hedged.keys()) >= {"net_daily_return", "hedge_daily_return",
                                    "hedge_legs"}
    assert len(hedged["net_daily_return"]) == len(strat["daily_return"])
