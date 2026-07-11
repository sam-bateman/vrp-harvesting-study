import numpy as np
import pandas as pd

from vrp.util.bootstrap import (
    bootstrap_paths,
    bootstrap_metrics,
    stationary_bootstrap_paths,
    var_and_es,
    confidence_intervals,
)


def test_bootstrap_paths_length_matches_input():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-02", periods=500)
    ret = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    paths = bootstrap_paths(ret, block_size=40, n_paths=50, seed=1)
    assert len(paths) == 50
    for p in paths:
        assert len(p) == len(ret)


def test_bootstrap_metrics_columns():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 500))
    df = bootstrap_metrics(ret, block_size=40, n_paths=20, seed=1)
    for c in ("ann_return", "ann_vol", "sharpe", "max_drawdown"):
        assert c in df.columns
    assert len(df) == 20


def test_bootstrap_preserves_mean_and_vol():
    # Distributional property: resampled paths must reproduce the source
    # series' mean and vol on average (a shape-only test would pass even
    # if the sampler returned garbage values).
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0.0005, 0.01, 2000))
    paths = bootstrap_paths(ret, block_size=40, n_paths=200, seed=1)
    means = np.array([p.mean() for p in paths])
    stds = np.array([p.std() for p in paths])
    assert abs(means.mean() - ret.mean()) < 3 * ret.std() / np.sqrt(2000)
    assert abs(stds.mean() - ret.std()) < 0.001


def test_bootstrap_preserves_block_autocorrelation():
    # An AR(1) series keeps its lag-1 autocorrelation under block
    # resampling (within-block structure survives; iid resampling would
    # destroy it).
    rng = np.random.default_rng(1)
    n = 2000
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + e[i]
    ret = pd.Series(x)
    src_ac = ret.autocorr(1)
    paths = bootstrap_paths(ret, block_size=40, n_paths=100, seed=2)
    boot_ac = np.mean([p.autocorr(1) for p in paths])
    assert abs(boot_ac - src_ac) < 0.1


def test_stationary_bootstrap_paths_shape_and_moments():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0.0005, 0.01, 1500))
    paths = stationary_bootstrap_paths(ret, mean_block_size=40,
                                       n_paths=100, seed=3)
    assert len(paths) == 100
    assert all(len(p) == len(ret) for p in paths)
    stds = np.array([p.std() for p in paths])
    assert abs(stds.mean() - ret.std()) < 0.001


def test_bootstrap_metrics_stationary_method():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 500))
    df = bootstrap_metrics(ret, block_size=40, n_paths=20, seed=1,
                           method="stationary")
    assert len(df) == 20


def test_var_and_es():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 10_000))
    v, es = var_and_es(ret, alpha=0.05)
    assert -0.02 < v < -0.01
    assert es < v


def test_confidence_intervals():
    df = pd.DataFrame({"x": np.linspace(-1, 1, 101)})
    ci = confidence_intervals(df, alphas=(0.05, 0.95))
    assert abs(ci.loc["x", "p05"] - (-0.9)) < 0.02
    assert abs(ci.loc["x", "p95"] - 0.9) < 0.02
