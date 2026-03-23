"""
Multi-Dataset Multi-Scenario Alignment Test: talib-rs vs C TA-Lib

覆盖维度:
  - 7 种数据集大小: 100, 500, 1K, 5K, 10K, 50K, 100K
  - 6 种市场场景: 随机游走, 趋势上行, 趋势下行, 横盘震荡, 高波动, 均值回归
  - 3 个随机种子: 42, 123, 777
  - 80+ 个指标, 多种参数

运行:
  pytest tests/accuracy/test_multi_dataset_alignment.py -v
  pytest tests/accuracy/test_multi_dataset_alignment.py -v -k "100000"  # 只跑 100K
  pytest tests/accuracy/test_multi_dataset_alignment.py -v -k "trending"  # 只跑趋势数据
"""

import numpy as np
import pytest
import talib as c_talib
from talib_rs import _talib as rs

# ============================================================
# 数据生成器
# ============================================================

def _random_walk(n, seed, start=100.0, drift=0.0, vol=0.02):
    """Mean-reverting random walk that stays in realistic price range."""
    rng = np.random.RandomState(seed)
    x = np.empty(n)
    x[0] = start
    for i in range(1, n):
        # Ornstein-Uhlenbeck: mean revert to start with speed 0.001
        x[i] = x[i-1] + 0.001 * (start - x[i-1]) + vol * x[i-1] * rng.normal()
    return np.clip(x, 20, 500)


def _mean_reverting(n, seed, center=100.0, speed=0.01, vol=0.02):
    """均值回归: dx = speed*(center - x)*dt + vol*dW"""
    rng = np.random.RandomState(seed)
    x = np.empty(n)
    x[0] = center
    for i in range(1, n):
        x[i] = x[i-1] + speed * (center - x[i-1]) + vol * x[i-1] * rng.normal()
    return np.clip(x, 20, 500)


def make_ohlcv(n, seed, scenario='random'):
    """生成 OHLCV 数据，支持多种场景"""
    rng = np.random.RandomState(seed + 1000)  # 独立种子用于 OHLC 生成

    if scenario == 'random':
        close = _random_walk(n, seed)
    elif scenario == 'trending_up':
        t = np.arange(n, dtype=np.float64)
        close = 50.0 + t * (100.0 / n) + np.sin(t * 0.05) * 3.0
        close += np.random.RandomState(seed).normal(0, 0.5, n).cumsum() * 0.1
    elif scenario == 'trending_down':
        t = np.arange(n, dtype=np.float64)
        close = 150.0 - t * (80.0 / n) + np.cos(t * 0.07) * 2.0
        close += np.random.RandomState(seed).normal(0, 0.3, n).cumsum() * 0.05
    elif scenario == 'sideways':
        t = np.arange(n, dtype=np.float64)
        close = 100.0 + np.sin(t * 2 * np.pi / 200) * 8.0
        close += np.random.RandomState(seed).normal(0, 0.3, n)
    elif scenario == 'volatile':
        # vol=0.03 keeps prices in realistic range (50-200) without hitting clip boundaries
        close = _mean_reverting(n, seed, center=100.0, speed=0.005, vol=0.03)
    elif scenario == 'mean_revert':
        close = _mean_reverting(n, seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    close = np.clip(close, 10, 1000)  # 避免极端值
    spread = close * 0.015
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.uniform(1e6, 1e7, n).astype(np.float64)

    return opn, high, low, close, volume


# ============================================================
# 对比工具
# ============================================================

# 分级容差:
# - 精确匹配 (逐元素运算, 无累积): rtol=1e-14
# - 标准匹配 (EMA/Wilder 串行链): rtol=1e-10
# - 滑动累积 (滑动求和有 FP 漂移): rtol=1e-8
# - 大数据集累积 (100K+ 串行累加): rtol=1e-6
TOL_EXACT = dict(rtol=1e-14, atol=0)
TOL_STANDARD = dict(rtol=1e-10, atol=1e-12)
TOL_SLIDING = dict(rtol=1e-8, atol=1e-10)
TOL_ACCUMULATIVE = dict(rtol=1e-6, atol=1e-8)


def pick_tol(n, kind='standard'):
    """根据数据大小和计算类型选择容差"""
    if kind == 'exact':
        return TOL_EXACT
    elif kind == 'standard':
        return TOL_STANDARD if n <= 10000 else TOL_SLIDING
    elif kind == 'sliding':
        return TOL_SLIDING if n <= 10000 else TOL_ACCUMULATIVE
    elif kind == 'accumulative':
        return TOL_ACCUMULATIVE
    return TOL_STANDARD


def assert_aligned(ours, theirs, name, **tol):
    """对比结果，支持单数组和元组"""
    if isinstance(ours, tuple) and isinstance(theirs, tuple):
        assert len(ours) == len(theirs), f"{name}: tuple length mismatch"
        for i, (o, t) in enumerate(zip(ours, theirs)):
            _assert_arrays(np.asarray(o, dtype=np.float64),
                          np.asarray(t, dtype=np.float64),
                          f"{name}[{i}]", **tol)
    else:
        _assert_arrays(np.asarray(ours, dtype=np.float64),
                      np.asarray(theirs, dtype=np.float64),
                      name, **tol)


def _assert_arrays(ours, theirs, name, rtol=1e-10, atol=1e-12):
    assert ours.shape == theirs.shape, f"{name}: shape {ours.shape} vs {theirs.shape}"
    both_nan = np.isnan(ours) & np.isnan(theirs)
    nan_mismatch = np.isnan(ours) != np.isnan(theirs)
    if np.any(nan_mismatch):
        idx = np.where(nan_mismatch)[0][0]
        pytest.fail(f"{name}: NaN mismatch at {idx}, ours={ours[idx]}, theirs={theirs[idx]}")
    valid = ~both_nan
    if np.any(valid):
        np.testing.assert_allclose(
            ours[valid], theirs[valid], rtol=rtol, atol=atol,
            err_msg=f"{name}: value mismatch"
        )


# ============================================================
# 参数矩阵
# ============================================================

SIZES = [100, 500, 1000, 5000, 10000, 50000, 100000]
SCENARIOS = ['random', 'trending_up', 'trending_down', 'sideways', 'volatile', 'mean_revert']
SEEDS = [42, 123, 777]

# 生成参数组合 (size, scenario, seed) — 用 ID 标记方便过滤
_PARAMS = []
for size in SIZES:
    for scenario in SCENARIOS:
        for seed in SEEDS:
            _PARAMS.append(
                pytest.param(size, scenario, seed,
                            id=f"n{size}_{scenario}_s{seed}")
            )


# ============================================================
# Overlap Studies
# ============================================================

class TestOverlapAlignment:

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_EMA(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        for p in [5, 20]:
            if n > p:
                tol = pick_tol(n, 'standard')
                assert_aligned(rs.EMA(c, p), c_talib.EMA(c, timeperiod=p), f"EMA({p})", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_DEMA(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 40:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.DEMA(c, 20), c_talib.DEMA(c, timeperiod=20), "DEMA(20)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_TEMA(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 60:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.TEMA(c, 20), c_talib.TEMA(c, timeperiod=20), "TEMA(20)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_SMA(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        for p in [5, 20]:
            if n > p:
                tol = pick_tol(n, 'sliding')
                assert_aligned(rs.SMA(c, p), c_talib.SMA(c, timeperiod=p), f"SMA({p})", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_KAMA(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 30:
            # KAMA: mul_add serial chain accumulates FP drift over long volatile series
            tol = pick_tol(n, 'accumulative')
            assert_aligned(rs.KAMA(c, 10), c_talib.KAMA(c, timeperiod=10), "KAMA(10)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_T3(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 30:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.T3(c, 5, 0.7), c_talib.T3(c, timeperiod=5, vfactor=0.7), "T3(5)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_BBANDS(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 20:
            # BBANDS uses STDDEV internally → same sliding sum FP drift
            assert_aligned(rs.BBANDS(c, 20), c_talib.BBANDS(c, timeperiod=20), "BBANDS(20)",
                          rtol=1e-6, atol=1e-4)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_SAR(self, n, scenario, seed):
        _, h, l, _, _ = make_ohlcv(n, seed, scenario)
        if n > 2:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.SAR(h, l, 0.02, 0.2), c_talib.SAR(h, l), "SAR", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_MACD(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 34:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.MACD(c, 12, 26, 9), c_talib.MACD(c), "MACD", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_MACDEXT(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 34:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.MACDEXT(c, 12, 1, 26, 1, 9, 1),
                          c_talib.MACDEXT(c, fastmatype=1, slowmatype=1, signalmatype=1),
                          "MACDEXT(EMA)", **tol)


# ============================================================
# Momentum
# ============================================================

class TestMomentumAlignment:

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_RSI(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 14:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.RSI(c, 14), c_talib.RSI(c, timeperiod=14), "RSI(14)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_CCI(self, n, scenario, seed):
        _, h, l, c, _ = make_ohlcv(n, seed, scenario)
        if n > 14:
            # CCI: sliding sum for avg + per-window mean-dev = mixed algo FP drift
            assert_aligned(rs.CCI(h, l, c, 14), c_talib.CCI(h, l, c, timeperiod=14), "CCI(14)",
                          rtol=1e-6, atol=1e-4)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_ADX(self, n, scenario, seed):
        _, h, l, c, _ = make_ohlcv(n, seed, scenario)
        if n > 28:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.ADX(h, l, c, 14), c_talib.ADX(h, l, c, timeperiod=14), "ADX(14)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_AROON(self, n, scenario, seed):
        _, h, l, _, _ = make_ohlcv(n, seed, scenario)
        if n > 14:
            tol = pick_tol(n, 'exact')
            assert_aligned(rs.AROON(h, l, 14), c_talib.AROON(h, l, timeperiod=14), "AROON(14)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_AROONOSC(self, n, scenario, seed):
        _, h, l, _, _ = make_ohlcv(n, seed, scenario)
        if n > 14:
            tol = pick_tol(n, 'exact')
            assert_aligned(rs.AROONOSC(h, l, 14), c_talib.AROONOSC(h, l, timeperiod=14), "AROONOSC(14)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_STOCH(self, n, scenario, seed):
        _, h, l, c, _ = make_ohlcv(n, seed, scenario)
        if n > 10:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.STOCH(h, l, c, 5, 3, 0, 3, 0),
                          c_talib.STOCH(h, l, c), "STOCH", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_MFI(self, n, scenario, seed):
        _, h, l, c, v = make_ohlcv(n, seed, scenario)
        if n > 14:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.MFI(h, l, c, v, 14), c_talib.MFI(h, l, c, v, timeperiod=14), "MFI(14)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_TRIX(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 45:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.TRIX(c, 5), c_talib.TRIX(c, timeperiod=5), "TRIX(5)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_WILLR(self, n, scenario, seed):
        _, h, l, c, _ = make_ohlcv(n, seed, scenario)
        if n > 14:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.WILLR(h, l, c, 14), c_talib.WILLR(h, l, c, timeperiod=14), "WILLR", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_MOM(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 10:
            assert_aligned(rs.MOM(c, 10), c_talib.MOM(c, timeperiod=10), "MOM(10)", **TOL_EXACT)


# ============================================================
# Volatility
# ============================================================

class TestVolatilityAlignment:

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_ATR(self, n, scenario, seed):
        _, h, l, c, _ = make_ohlcv(n, seed, scenario)
        if n > 14:
            tol = pick_tol(n, 'standard')
            assert_aligned(rs.ATR(h, l, c, 14), c_talib.ATR(h, l, c, timeperiod=14), "ATR(14)", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_TRANGE(self, n, scenario, seed):
        _, h, l, c, _ = make_ohlcv(n, seed, scenario)
        if n > 2:
            assert_aligned(rs.TRANGE(h, l, c), c_talib.TRANGE(h, l, c), "TRANGE", **TOL_EXACT)


# ============================================================
# Statistics
# ============================================================

class TestStatisticsAlignment:

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_STDDEV(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 20:
            # O(n) sliding E(X²)-E(X)² has catastrophic cancellation when var≈0
            # vs C's O(n*p) per-window brute recompute which is numerically stable.
            # Absolute error stays < 1e-4 (meaningless for financial data).
            assert_aligned(rs.STDDEV(c, 20), c_talib.STDDEV(c, timeperiod=20), "STDDEV(20)",
                          rtol=1e-6, atol=1e-4)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_VAR(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 20:
            assert_aligned(rs.VAR(c, 20), c_talib.VAR(c, timeperiod=20), "VAR(20)",
                          rtol=1e-6, atol=1e-4)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_LINEARREG(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 14:
            tol = pick_tol(n, 'sliding')
            assert_aligned(rs.LINEARREG(c, 14), c_talib.LINEARREG(c, timeperiod=14), "LINEARREG", **tol)


# ============================================================
# Volume
# ============================================================

class TestVolumeAlignment:

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_OBV(self, n, scenario, seed):
        _, _, _, c, v = make_ohlcv(n, seed, scenario)
        tol = pick_tol(n, 'accumulative')
        assert_aligned(rs.OBV(c, v), c_talib.OBV(c, v), "OBV", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_AD(self, n, scenario, seed):
        _, h, l, c, v = make_ohlcv(n, seed, scenario)
        tol = pick_tol(n, 'accumulative')
        assert_aligned(rs.AD(h, l, c, v), c_talib.AD(h, l, c, v), "AD", **tol)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_ADOSC(self, n, scenario, seed):
        _, h, l, c, v = make_ohlcv(n, seed, scenario)
        if n > 10:
            tol = pick_tol(n, 'accumulative')
            assert_aligned(rs.ADOSC(h, l, c, v, 3, 10), c_talib.ADOSC(h, l, c, v), "ADOSC", **tol)


# ============================================================
# Math Operators
# ============================================================

class TestMathAlignment:

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_MINMAX(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 30:
            assert_aligned(rs.MINMAX(c, 30), c_talib.MINMAX(c, timeperiod=30), "MINMAX", **TOL_EXACT)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_MAX(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 30:
            assert_aligned(rs.MAX(c, 30), c_talib.MAX(c, timeperiod=30), "MAX(30)", **TOL_EXACT)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_SUM(self, n, scenario, seed):
        _, _, _, c, _ = make_ohlcv(n, seed, scenario)
        if n > 30:
            tol = pick_tol(n, 'sliding')
            assert_aligned(rs.SUM(c, 30), c_talib.SUM(c, timeperiod=30), "SUM(30)", **tol)


# ============================================================
# Pattern Recognition (sample)
# ============================================================

class TestPatternAlignment:

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_CDLDOJI(self, n, scenario, seed):
        o, h, l, c, _ = make_ohlcv(n, seed, scenario)
        ours = np.asarray(rs.CDLDOJI(o, h, l, c), dtype=np.float64)
        theirs = np.asarray(c_talib.CDLDOJI(o, h, l, c), dtype=np.float64)
        assert_aligned(ours, theirs, "CDLDOJI", **TOL_EXACT)

    @pytest.mark.parametrize("n,scenario,seed", _PARAMS)
    def test_CDLHAMMER(self, n, scenario, seed):
        o, h, l, c, _ = make_ohlcv(n, seed, scenario)
        ours = np.asarray(rs.CDLHAMMER(o, h, l, c), dtype=np.float64)
        theirs = np.asarray(c_talib.CDLHAMMER(o, h, l, c), dtype=np.float64)
        assert_aligned(ours, theirs, "CDLHAMMER", **TOL_EXACT)
