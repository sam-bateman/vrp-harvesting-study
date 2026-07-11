"""Phase 5 — tail-risk analysis across the major constructions.

For each construction, compute:
1. Moving-block bootstrap with block_size=40 and n_paths=2000.
2. Daily 1%-VaR and Expected Shortfall.
3. October 1987 extrapolated stress: single-day PnL as % of capital.

Outputs:
    reports/tail_risk/bootstrap_ci.json
    reports/tail_risk/var_es.json
    reports/tail_risk/stress_1987.json
    reports/tail_risk/full_sample_summary.json
    reports/tail_risk/bootstrap_sharpe_histogram.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.analysis.stress_1987 import (
    stress_calendar, stress_put_writer,
    STRESS_SPX_DROP, STRESS_VIX_JUMP,
)
from vrp.util.bs import bs_price, strike_from_delta
from vrp.data.cboe_indices import load_cboe_index
from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.data.vx_futures import load_vx_continuous
from vrp.overlays.regime_filter import vix_regime_mask, apply_mask
from vrp.report.metrics import probabilistic_sharpe_ratio, summary
from vrp.strategies.strategy_a import run_strategy_a
from vrp.strategies.strategy_b import run_strategy_b
from vrp.strategies.strategy_c import run_strategy_c
from vrp.util.bootstrap import (
    bootstrap_metrics, var_and_es, confidence_intervals,
)
from vrp.util.vol import close_to_close_rv
from vrp.util.vrp_signal import compute_vrp


BLOCK_SIZE = 40
N_PATHS = 2000


def _bootstrap_block(ret: pd.Series) -> dict:
    metrics = bootstrap_metrics(ret, block_size=BLOCK_SIZE,
                                  n_paths=N_PATHS, seed=42)
    ci = confidence_intervals(metrics, alphas=(0.05, 0.50, 0.95))
    return ci.to_dict(orient="index")


def _var_es_block(ret: pd.Series) -> dict:
    v1, es1 = var_and_es(ret, alpha=0.01)
    v5, es5 = var_and_es(ret, alpha=0.05)
    return {"var_1pct": v1, "es_1pct": es1,
             "var_5pct": v5, "es_5pct": es5}


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "tail_risk"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx = load_spx(start="2013-01-01", end="2024-12-31")["close"]
    vix = load_vix(start="2013-01-01", end="2024-12-31")
    vx = load_vx_continuous(start="2013-01-01", end="2024-12-31")
    rv = close_to_close_rv(spx, window=20)
    vrp = compute_vrp(vix, rv).dropna()
    put_index = load_cboe_index("PUT")
    put_daily_ret = put_index.pct_change().dropna()

    construction_returns = {}

    a_short = run_strategy_a(vx, tc_bps_per_roll=1.0, direction="short_front")
    construction_returns["A_short_front"] = a_short["daily_return"]

    a_long = run_strategy_a(vx, tc_bps_per_roll=1.0, direction="long_front")
    construction_returns["A_long_front"] = a_long["daily_return"]

    construction_returns["B_PUT_index"] = put_daily_ret.loc["2013-01-01":"2024-12-31"]

    b_spread = run_strategy_b(spx, vix, target_delta=-0.30,
                               long_put_delta=-0.10, tc_pct_of_premium=0.05)
    construction_returns["B_synth_spread"] = b_spread["daily_return"]

    c_spread = run_strategy_c(spx, vix, vrp, threshold=1.0,
                               target_delta=-0.30, long_put_delta=-0.10,
                               tc_pct_of_premium=0.05)
    construction_returns["C_spread_thr=+1"] = c_spread["daily_return"]

    mask = vix_regime_mask(vix, vx["front_settle"], vx["second_settle"])
    construction_returns["C_spread_thr=+1_plus_O1"] = apply_mask(
        c_spread["daily_return"], mask
    )

    bootstrap_cis = {}
    var_es = {}
    full_summary = {}
    psr = {}
    for name, ret in construction_returns.items():
        bootstrap_cis[name] = _bootstrap_block(ret)
        var_es[name] = _var_es_block(ret)
        full_summary[name] = summary(ret)
        psr[name] = probabilistic_sharpe_ratio(ret)
    (out_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap_cis, indent=2))
    (out_dir / "var_es.json").write_text(json.dumps(var_es, indent=2))
    (out_dir / "full_sample_summary.json").write_text(
        json.dumps(full_summary, indent=2)
    )
    (out_dir / "psr.json").write_text(json.dumps(psr, indent=2))

    # Bootstrap-method robustness on the headline construction: the MBB
    # block size is a judgment call, so sweep it and cross-check with the
    # stationary bootstrap (random geometric block lengths).
    headline = "C_spread_thr=+1_plus_O1"
    headline_ret = construction_returns[headline]
    method_sensitivity = {}
    for label, method, block in [("mbb_20", "mbb", 20),
                                 ("mbb_40", "mbb", 40),
                                 ("mbb_60", "mbb", 60),
                                 ("stationary_40", "stationary", 40)]:
        ms = bootstrap_metrics(headline_ret, block_size=block,
                               n_paths=N_PATHS, seed=42, method=method)
        ci = confidence_intervals(ms, alphas=(0.05, 0.50, 0.95))
        method_sensitivity[label] = ci.to_dict(orient="index")["sharpe"]
    (out_dir / "bootstrap_method_sensitivity.json").write_text(
        json.dumps({headline: method_sensitivity}, indent=2)
    )

    stress = {}
    stress["A_short_front"] = stress_calendar(
        front_0=20, second_0=22, delta_front=STRESS_VIX_JUMP,
        delta_second=STRESS_VIX_JUMP * 2 / 3, direction="short_front"
    )
    stress["A_long_front"] = stress_calendar(
        front_0=20, second_0=22, delta_front=STRESS_VIX_JUMP,
        delta_second=STRESS_VIX_JUMP * 2 / 3, direction="long_front"
    )
    # Strikes derived from the SAME delta targets the strategy trades
    # (-0.30d short / -0.10d long at S=100, IV=20%, T=30d), not ad-hoc
    # levels — the stress must hit the position the backtest holds.
    S0, sigma0_pct, T_open_days = 100.0, 20.0, 30
    T0 = T_open_days / 365.0
    sigma0 = sigma0_pct / 100.0
    K_short = strike_from_delta(S0, T0, sigma0, 0.0, "put", -0.30)
    K_long = strike_from_delta(S0, T0, sigma0, 0.0, "put", -0.10)
    prem_short = bs_price(S0, K_short, T0, sigma0, 0.0, "put")
    prem_long = bs_price(S0, K_long, T0, sigma0, 0.0, "put")
    stress["B_synth_naked"] = stress_put_writer(
        S0=S0, K_short=K_short, sigma0_pct=sigma0_pct,
        S_shock=S0 * (1 + STRESS_SPX_DROP),
        sigma_shock_pct=50,
        T_remaining_days=20, K_long=None,
        premium_collected=prem_short,
    )
    stress["B_synth_spread"] = stress_put_writer(
        S0=S0, K_short=K_short, sigma0_pct=sigma0_pct,
        S_shock=S0 * (1 + STRESS_SPX_DROP),
        sigma_shock_pct=50,
        T_remaining_days=20, K_long=K_long,
        premium_collected=prem_short - prem_long,
    )
    stress["C_spread_thr=+1"] = stress["B_synth_spread"]
    stress["C_spread_thr=+1_plus_O1"] = {
        **stress["B_synth_spread"],
        "note": "Day-0 PnL = spread loss; O1 cashes out day 1+ (approx).",
    }
    (out_dir / "stress_1987.json").write_text(
        json.dumps(stress, indent=2, default=str)
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, (name, ret) in zip(axes.flat, construction_returns.items()):
        ms = bootstrap_metrics(ret, block_size=BLOCK_SIZE, n_paths=N_PATHS, seed=42)
        ax.hist(ms["sharpe"].dropna(), bins=40, alpha=0.7)
        ax.axvline(summary(ret)["sharpe"], linestyle="--", color="red",
                    label="in-sample")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Bootstrap Sharpe")
        ax.legend(fontsize=8)
    fig.suptitle(f"Bootstrap Sharpe distribution (block={BLOCK_SIZE}, n={N_PATHS})")
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_sharpe_histogram.png", dpi=140,
                 bbox_inches="tight")
    plt.close(fig)

    print(f"Tail-risk outputs written to {out_dir}")
    print("\nBootstrap Sharpe CIs (5%/50%/95%):")
    for name, ci in bootstrap_cis.items():
        s = ci.get("sharpe", {})
        in_sample = full_summary[name]["sharpe"]
        print(f"  {name:30s}  p05={s.get('p05', 0):+.3f} "
              f"p50={s.get('p50', 0):+.3f} p95={s.get('p95', 0):+.3f}  "
              f"[in-sample={in_sample:+.3f}]")
    print("\n1% VaR and ES (daily):")
    for name, vd in var_es.items():
        print(f"  {name:30s}  VaR={vd['var_1pct']:+.4f}  ES={vd['es_1pct']:+.4f}")
    print("\nProbabilistic Sharpe ratio, P(true Sharpe > 0):")
    for name, p in psr.items():
        print(f"  {name:30s}  PSR={p:.3f}")
    print(f"\nBootstrap method sensitivity ({headline}, Sharpe p05/p50/p95):")
    for label, s in method_sensitivity.items():
        print(f"  {label:15s}  p05={s['p05']:+.3f} p50={s['p50']:+.3f} "
              f"p95={s['p95']:+.3f}")
    print("\n1987 single-day stress (PnL as % of capital):")
    for name, s in stress.items():
        key = "daily_pnl_return" if "daily_pnl_return" in s else "pnl_return"
        print(f"  {name:30s}  {key}={s[key]:+.3f}")


if __name__ == "__main__":
    main()
