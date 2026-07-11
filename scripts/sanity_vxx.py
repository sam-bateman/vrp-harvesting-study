"""Sanity check: Strategy A cumulative PnL should correlate positively with
the negative of VXX cumulative return in contango-heavy periods.

Strategy A is dollar-neutral short-front / long-second, while VXX holds a
~30-day rolling long blend — so -VXX is close to a pure short-front
carry position and Strategy A shares its dominant leg. With splice-free
held-contract returns the daily correlation runs higher than the old
spliced series showed; we expect 0.5-0.9. If it is negative or below
0.3, the roll logic or PnL sign in Strategy A is probably wrong;
investigate before moving on.

Because VXX has reverse splits and a 2019 issuance change, pre-2018
divergence is expected. Shape-level agreement across both windows is the
real signal.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from vrp.data.vx_futures import load_vx_continuous
from vrp.strategies.strategy_a import run_strategy_a


def main() -> None:
    vx = load_vx_continuous(start="2013-01-01", end="2024-12-31")
    result = run_strategy_a(vx)
    strat_equity = (1 + result["daily_return"]).cumprod()

    vxx = yf.download("VXX", start="2013-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)["Close"].dropna()
    if hasattr(vxx, "columns"):
        vxx = vxx.squeeze()

    # Correlate daily returns against -r_VXX directly; returns of (1/VXX)
    # would embed a convexity term (-r + r^2 - ...) that biases the gate.
    ret_pair = pd.DataFrame({
        "strategy_a": result["daily_return"],
        "neg_vxx": -vxx.pct_change(),
    }).dropna()
    corr = ret_pair.corr().iloc[0, 1]

    common = strat_equity.index.intersection(vxx.index)
    df = pd.DataFrame({
        "strategy_a": strat_equity.loc[common],
        "neg_vxx": (1 + ret_pair["neg_vxx"]).cumprod().reindex(common),
    })

    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    (df / df.iloc[0]).plot(ax=ax)
    ax.set_title(
        f"Strategy A vs 1/VXX (normalized). Daily-return corr = {corr:.2f}"
    )
    fig.savefig(out_dir / "sanity_vxx.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Daily return correlation Strategy A vs -VXX: {corr:.3f}")
    print("Expected: 0.5-0.9. If |corr| < 0.3 or negative, investigate.")


if __name__ == "__main__":
    main()
