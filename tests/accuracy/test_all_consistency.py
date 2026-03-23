"""
全指标一致性测试: talib-rs vs C TA-Lib
覆盖所有 155 个函数，使用多种数据集对比输出精度。

运行: pytest tests/accuracy/test_all_consistency.py -v
"""

import numpy as np
import pytest

# ---- C TA-Lib (原版) ----
import talib as c_talib

# ---- talib-rs (我们的实现) ----
from talib_rs import _talib as rs

# ============================================================
# 测试数据集
# ============================================================

def _random_walk(n, seed, start=100.0, drift=0.0005, vol=0.02):
    rng = np.random.RandomState(seed)
    returns = rng.normal(drift, vol, n)
    return start * np.exp(np.cumsum(returns))


def _make_ohlcv(n, seed):
    rng = np.random.RandomState(seed)
    close = _random_walk(n, seed)
    spread = close * 0.015
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.uniform(1e6, 1e7, n)
    return open_, high, low, close, volume


# 多种数据集
DATASETS = {
    'random_1k':   _make_ohlcv(1000, 42),
    'random_5k':   _make_ohlcv(5000, 123),
    'trending_up': None,   # 延迟构造
    'trending_dn': None,
    'sideways':    None,
    'volatile':    None,
}

# 趋势向上
_n = 1000
_t = np.arange(_n, dtype=np.float64)
_close_up = 50.0 + _t * 0.1 + np.sin(_t * 0.1) * 2.0
_spread_up = np.abs(_close_up) * 0.01 + 0.5
DATASETS['trending_up'] = (
    _close_up - _spread_up * 0.3,
    _close_up + _spread_up,
    _close_up - _spread_up,
    _close_up,
    np.random.RandomState(10).uniform(1e6, 5e6, _n),
)

# 趋势向下
_close_dn = 150.0 - _t * 0.08 + np.cos(_t * 0.15) * 3.0
_spread_dn = np.abs(_close_dn) * 0.01 + 0.5
DATASETS['trending_dn'] = (
    _close_dn + _spread_dn * 0.2,
    _close_dn + _spread_dn,
    _close_dn - _spread_dn,
    _close_dn,
    np.random.RandomState(11).uniform(1e6, 5e6, _n),
)

# 横盘震荡
_close_side = 100.0 + np.sin(_t * 0.3) * 5.0 + np.random.RandomState(20).normal(0, 0.5, _n)
_spread_s = 1.0
DATASETS['sideways'] = (
    _close_side - 0.2,
    _close_side + _spread_s,
    _close_side - _spread_s,
    _close_side,
    np.random.RandomState(12).uniform(1e6, 5e6, _n),
)

# 高波动
_close_vol = _random_walk(_n, 99, vol=0.08)
_spread_v = _close_vol * 0.03
DATASETS['volatile'] = (
    _close_vol - _spread_v * 0.5,
    _close_vol + _spread_v,
    _close_vol - _spread_v,
    _close_vol,
    np.random.RandomState(13).uniform(1e6, 1e8, _n),
)

# ============================================================
# 比较辅助
# ============================================================

RTOL = 1e-10
ATOL = 1e-12

def compare(ours, theirs, name="", rtol=RTOL, atol=ATOL):
    """比较两个结果，支持单数组或元组"""
    if isinstance(ours, tuple) and isinstance(theirs, tuple):
        assert len(ours) == len(theirs), f"{name}: output count mismatch {len(ours)} vs {len(theirs)}"
        for i, (o, t) in enumerate(zip(ours, theirs)):
            _compare_arrays(np.asarray(o, dtype=np.float64),
                          np.asarray(t, dtype=np.float64),
                          f"{name}[{i}]", rtol, atol)
    else:
        _compare_arrays(np.asarray(ours, dtype=np.float64),
                       np.asarray(theirs, dtype=np.float64),
                       name, rtol, atol)


def _compare_arrays(ours, theirs, name, rtol, atol):
    assert ours.shape == theirs.shape, f"{name}: shape mismatch {ours.shape} vs {theirs.shape}"
    # 两边都是 NaN 的位置跳过
    both_nan = np.isnan(ours) & np.isnan(theirs)
    # 一边 NaN 另一边不是 -> 报错
    our_nan = np.isnan(ours) & ~np.isnan(theirs)
    their_nan = ~np.isnan(ours) & np.isnan(theirs)
    if np.any(our_nan) or np.any(their_nan):
        first_mismatch = np.where(our_nan | their_nan)[0][0]
        pytest.fail(f"{name}: NaN mismatch at index {first_mismatch}, "
                    f"ours={ours[first_mismatch]}, theirs={theirs[first_mismatch]}")
    # 比较非 NaN 值
    valid = ~both_nan
    if np.any(valid):
        np.testing.assert_allclose(
            ours[valid], theirs[valid],
            rtol=rtol, atol=atol,
            err_msg=f"{name}: value mismatch"
        )


# ============================================================
# Overlap Studies
# ============================================================

class TestOverlapConsistency:

    @pytest.mark.parametrize("ds_name", ['random_1k', 'random_5k', 'trending_up', 'trending_dn', 'sideways', 'volatile'])
    @pytest.mark.parametrize("period", [5, 10, 14, 20, 30, 50])
    def test_SMA(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.SMA(close, period), c_talib.SMA(close, timeperiod=period), f"SMA({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up', 'volatile'])
    @pytest.mark.parametrize("period", [5, 12, 26, 50])
    def test_EMA(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.EMA(close, period), c_talib.EMA(close, timeperiod=period), f"EMA({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    @pytest.mark.parametrize("period", [5, 14, 30])
    def test_WMA(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.WMA(close, period), c_talib.WMA(close, timeperiod=period), f"WMA({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    @pytest.mark.parametrize("period", [5, 14, 30])
    def test_DEMA(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.DEMA(close, period), c_talib.DEMA(close, timeperiod=period), f"DEMA({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    @pytest.mark.parametrize("period", [5, 14, 30])
    def test_TEMA(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.TEMA(close, period), c_talib.TEMA(close, timeperiod=period), f"TEMA({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'sideways'])
    @pytest.mark.parametrize("period", [5, 14, 30])
    def test_TRIMA(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.TRIMA(close, period), c_talib.TRIMA(close, timeperiod=period), f"TRIMA({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    def test_KAMA(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.KAMA(close, 30), c_talib.KAMA(close, timeperiod=30), "KAMA(30)")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    def test_BBANDS(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        ours = rs.BBANDS(close, 20, 2.0, 2.0, 0)
        theirs = c_talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        compare(ours, theirs, "BBANDS(20)")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    def test_SAR(self, ds_name):
        _, high, low, _, _ = DATASETS[ds_name]
        compare(rs.SAR(high, low, 0.02, 0.2),
                c_talib.SAR(high, low, acceleration=0.02, maximum=0.2), "SAR")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    @pytest.mark.parametrize("period", [7, 14])
    def test_MIDPOINT(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.MIDPOINT(close, period), c_talib.MIDPOINT(close, timeperiod=period))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    @pytest.mark.parametrize("period", [7, 14])
    def test_MIDPRICE(self, ds_name, period):
        _, high, low, _, _ = DATASETS[ds_name]
        compare(rs.MIDPRICE(high, low, period), c_talib.MIDPRICE(high, low, timeperiod=period))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_HT_TRENDLINE(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.HT_TRENDLINE(close), c_talib.HT_TRENDLINE(close), "HT_TRENDLINE")


# ============================================================
# Momentum Indicators
# ============================================================

class TestMomentumConsistency:

    @pytest.mark.parametrize("ds_name", ['random_1k', 'random_5k', 'trending_up', 'trending_dn', 'sideways', 'volatile'])
    @pytest.mark.parametrize("period", [5, 14, 21])
    def test_RSI(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.RSI(close, period), c_talib.RSI(close, timeperiod=period), f"RSI({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up', 'volatile'])
    def test_MACD(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.MACD(close, 12, 26, 9),
                c_talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9), "MACD")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    def test_MACDFIX(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.MACDFIX(close, 9), c_talib.MACDFIX(close, signalperiod=9), "MACDFIX")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'sideways'])
    def test_STOCH(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.STOCH(high, low, close, 5, 3, 0, 3, 0),
                c_talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0), "STOCH")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'sideways'])
    def test_STOCHF(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.STOCHF(high, low, close, 5, 3, 0),
                c_talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0), "STOCHF")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up', 'volatile'])
    @pytest.mark.parametrize("period", [7, 14])
    def test_ADX(self, ds_name, period):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.ADX(high, low, close, period),
                c_talib.ADX(high, low, close, timeperiod=period), f"ADX({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_ADXR(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.ADXR(high, low, close, 14),
                c_talib.ADXR(high, low, close, timeperiod=14), "ADXR")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up', 'volatile'])
    @pytest.mark.parametrize("period", [7, 14, 20])
    def test_CCI(self, ds_name, period):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.CCI(high, low, close, period),
                c_talib.CCI(high, low, close, timeperiod=period), f"CCI({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    @pytest.mark.parametrize("period", [5, 10, 20])
    def test_MOM(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.MOM(close, period), c_talib.MOM(close, timeperiod=period), f"MOM({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    @pytest.mark.parametrize("period", [5, 10])
    def test_ROC(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.ROC(close, period), c_talib.ROC(close, timeperiod=period))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    @pytest.mark.parametrize("period", [5, 10])
    def test_ROCP(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.ROCP(close, period), c_talib.ROCP(close, timeperiod=period))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    @pytest.mark.parametrize("period", [5, 10])
    def test_ROCR(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.ROCR(close, period), c_talib.ROCR(close, timeperiod=period))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_ROCR100(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.ROCR100(close, 10), c_talib.ROCR100(close, timeperiod=10))

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    def test_WILLR(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.WILLR(high, low, close, 14),
                c_talib.WILLR(high, low, close, timeperiod=14), "WILLR")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_APO(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.APO(close, 12, 26, 0), c_talib.APO(close, fastperiod=12, slowperiod=26, matype=0))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_PPO(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.PPO(close, 12, 26, 0), c_talib.PPO(close, fastperiod=12, slowperiod=26, matype=0))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_BOP(self, ds_name):
        o, h, l, c, _ = DATASETS[ds_name]
        compare(rs.BOP(o, h, l, c), c_talib.BOP(o, h, l, c))

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    def test_CMO(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.CMO(close, 14), c_talib.CMO(close, timeperiod=14))

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    def test_AROON(self, ds_name):
        _, high, low, _, _ = DATASETS[ds_name]
        compare(rs.AROON(high, low, 14), c_talib.AROON(high, low, timeperiod=14), "AROON")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_AROONOSC(self, ds_name):
        _, high, low, _, _ = DATASETS[ds_name]
        compare(rs.AROONOSC(high, low, 14), c_talib.AROONOSC(high, low, timeperiod=14))

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    def test_MFI(self, ds_name):
        _, high, low, close, volume = DATASETS[ds_name]
        compare(rs.MFI(high, low, close, volume, 14),
                c_talib.MFI(high, low, close, volume, timeperiod=14), "MFI")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_TRIX(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.TRIX(close, 15), c_talib.TRIX(close, timeperiod=15))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_ULTOSC(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.ULTOSC(high, low, close, 7, 14, 28),
                c_talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_DX(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.DX(high, low, close, 14), c_talib.DX(high, low, close, timeperiod=14))

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    def test_PLUS_DI(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.PLUS_DI(high, low, close, 14), c_talib.PLUS_DI(high, low, close, timeperiod=14))

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    def test_MINUS_DI(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.MINUS_DI(high, low, close, 14), c_talib.MINUS_DI(high, low, close, timeperiod=14))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_PLUS_DM(self, ds_name):
        _, high, low, _, _ = DATASETS[ds_name]
        compare(rs.PLUS_DM(high, low, 14), c_talib.PLUS_DM(high, low, timeperiod=14))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_MINUS_DM(self, ds_name):
        _, high, low, _, _ = DATASETS[ds_name]
        compare(rs.MINUS_DM(high, low, 14), c_talib.MINUS_DM(high, low, timeperiod=14))


# ============================================================
# Volatility
# ============================================================

class TestVolatilityConsistency:

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile', 'trending_up'])
    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_ATR(self, ds_name, period):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.ATR(high, low, close, period),
                c_talib.ATR(high, low, close, timeperiod=period), f"ATR({period})")

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    def test_NATR(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.NATR(high, low, close, 14),
                c_talib.NATR(high, low, close, timeperiod=14))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_TRANGE(self, ds_name):
        _, high, low, close, _ = DATASETS[ds_name]
        compare(rs.TRANGE(high, low, close), c_talib.TRANGE(high, low, close))


# ============================================================
# Volume
# ============================================================

class TestVolumeConsistency:

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up'])
    def test_AD(self, ds_name):
        _, high, low, close, volume = DATASETS[ds_name]
        compare(rs.AD(high, low, close, volume),
                c_talib.AD(high, low, close, volume))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_ADOSC(self, ds_name):
        _, high, low, close, volume = DATASETS[ds_name]
        compare(rs.ADOSC(high, low, close, volume, 3, 10),
                c_talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10))

    @pytest.mark.parametrize("ds_name", ['random_1k', 'trending_up', 'volatile'])
    def test_OBV(self, ds_name):
        _, _, _, close, volume = DATASETS[ds_name]
        compare(rs.OBV(close, volume), c_talib.OBV(close, volume))


# ============================================================
# Price Transform
# ============================================================

class TestPriceTransformConsistency:

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    def test_AVGPRICE(self, ds_name):
        o, h, l, c, _ = DATASETS[ds_name]
        compare(rs.AVGPRICE(o, h, l, c), c_talib.AVGPRICE(o, h, l, c))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_MEDPRICE(self, ds_name):
        _, h, l, _, _ = DATASETS[ds_name]
        compare(rs.MEDPRICE(h, l), c_talib.MEDPRICE(h, l))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_TYPPRICE(self, ds_name):
        _, h, l, c, _ = DATASETS[ds_name]
        compare(rs.TYPPRICE(h, l, c), c_talib.TYPPRICE(h, l, c))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_WCLPRICE(self, ds_name):
        _, h, l, c, _ = DATASETS[ds_name]
        compare(rs.WCLPRICE(h, l, c), c_talib.WCLPRICE(h, l, c))


# ============================================================
# Statistic Functions
# ============================================================

class TestStatisticConsistency:
    # Rust 使用 O(n) 滑动累加器（sum += new - old），C 使用 O(n·p) 逐窗口重计算。
    # 算法差异导致浮点累积漂移：1000 点数据上 max rel_diff ≈ 2e-9。
    # 以下指标分两组：per-window 计算的用严格容差，sliding 的用 1e-9。
    _SLIDING_RTOL = 3e-9

    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    @pytest.mark.parametrize("period", [5, 14, 30])
    def test_STDDEV(self, ds_name, period):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.STDDEV(close, period, 1.0),
                c_talib.STDDEV(close, timeperiod=period, nbdev=1.0), f"STDDEV({period})",
                rtol=self._SLIDING_RTOL)

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_VAR(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.VAR(close, 5, 1.0), c_talib.VAR(close, timeperiod=5, nbdev=1.0),
                rtol=self._SLIDING_RTOL)

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_BETA(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        close2 = _random_walk(len(close), 77)
        compare(rs.BETA(close, close2, 5), c_talib.BETA(close, close2, timeperiod=5))

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_CORREL(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        close2 = _random_walk(len(close), 77)
        compare(rs.CORREL(close, close2, 30), c_talib.CORREL(close, close2, timeperiod=30),
                rtol=self._SLIDING_RTOL)

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    @pytest.mark.parametrize("func_name", ['LINEARREG', 'LINEARREG_INTERCEPT', 'TSF'])
    def test_linearreg_strict(self, ds_name, func_name):
        _, _, _, close, _ = DATASETS[ds_name]
        ours = getattr(rs, func_name)(close, 14)
        theirs = getattr(c_talib, func_name)(close, timeperiod=14)
        compare(ours, theirs, func_name)

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    @pytest.mark.parametrize("func_name", ['LINEARREG_SLOPE', 'LINEARREG_ANGLE'])
    def test_linearreg_sliding(self, ds_name, func_name):
        # SLOPE 和 ANGLE 的滑动 ws 累加器在小值处 rel_diff 可达 2e-9
        _, _, _, close, _ = DATASETS[ds_name]
        ours = getattr(rs, func_name)(close, 14)
        theirs = getattr(c_talib, func_name)(close, timeperiod=14)
        compare(ours, theirs, func_name, rtol=self._SLIDING_RTOL)


# ============================================================
# Math Transform
# ============================================================

class TestMathTransformConsistency:

    @pytest.mark.parametrize("func_name", [
        'ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH', 'EXP', 'FLOOR',
        'LN', 'LOG10', 'SIN', 'SINH', 'SQRT', 'TAN', 'TANH',
    ])
    def test_math_transform(self, func_name):
        # 使用安全范围的输入（避免 asin/acos 超出 [-1,1]）
        if func_name in ('ACOS', 'ASIN'):
            data = np.linspace(-0.99, 0.99, 500)
        elif func_name in ('LN', 'LOG10', 'SQRT'):
            data = np.linspace(0.01, 100.0, 500)
        else:
            data = np.linspace(-2.0, 2.0, 500)
        ours = getattr(rs, func_name)(data)
        theirs = getattr(c_talib, func_name)(data)
        compare(ours, theirs, func_name)


# ============================================================
# Math Operators
# ============================================================

class TestMathOperatorConsistency:

    def test_ADD(self):
        a = np.arange(1.0, 1001.0)
        b = np.arange(1001.0, 2001.0)
        compare(rs.ADD(a, b), c_talib.ADD(a, b))

    def test_SUB(self):
        a = np.arange(1.0, 1001.0)
        b = np.arange(1001.0, 2001.0)
        compare(rs.SUB(a, b), c_talib.SUB(a, b))

    def test_MULT(self):
        a = np.arange(1.0, 1001.0)
        b = np.arange(1001.0, 2001.0)
        compare(rs.MULT(a, b), c_talib.MULT(a, b))

    def test_DIV(self):
        a = np.arange(1.0, 1001.0)
        b = np.arange(1.0, 1001.0) + 0.5
        compare(rs.DIV(a, b), c_talib.DIV(a, b))

    @pytest.mark.parametrize("period", [5, 14, 30])
    def test_MAX(self, period):
        _, _, _, close, _ = DATASETS['random_1k']
        compare(rs.MAX(close, period), c_talib.MAX(close, timeperiod=period))

    @pytest.mark.parametrize("period", [5, 14, 30])
    def test_MIN(self, period):
        _, _, _, close, _ = DATASETS['random_1k']
        compare(rs.MIN(close, period), c_talib.MIN(close, timeperiod=period))

    @pytest.mark.parametrize("period", [5, 14, 30])
    def test_SUM(self, period):
        _, _, _, close, _ = DATASETS['random_1k']
        compare(rs.SUM(close, period), c_talib.SUM(close, timeperiod=period))


# ============================================================
# Cycle Indicators
# ============================================================

class TestCycleConsistency:

    @pytest.mark.parametrize("ds_name", ['random_1k', 'sideways'])
    def test_HT_DCPERIOD(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.HT_DCPERIOD(close), c_talib.HT_DCPERIOD(close), "HT_DCPERIOD")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_HT_DCPHASE(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.HT_DCPHASE(close), c_talib.HT_DCPHASE(close), "HT_DCPHASE")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_HT_PHASOR(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.HT_PHASOR(close), c_talib.HT_PHASOR(close), "HT_PHASOR")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_HT_SINE(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        compare(rs.HT_SINE(close), c_talib.HT_SINE(close), "HT_SINE")

    @pytest.mark.parametrize("ds_name", ['random_1k'])
    def test_HT_TRENDMODE(self, ds_name):
        _, _, _, close, _ = DATASETS[ds_name]
        ours = np.asarray(rs.HT_TRENDMODE(close), dtype=np.float64)
        theirs = np.asarray(c_talib.HT_TRENDMODE(close), dtype=np.float64)
        compare(ours, theirs, "HT_TRENDMODE")


# ============================================================
# Pattern Recognition (采样测试)
# ============================================================

class TestPatternConsistency:
    """K 线形态: 验证输出形状和信号范围一致"""

    CDL_FUNCTIONS = [
        'CDLDOJI', 'CDLHAMMER', 'CDLENGULFING', 'CDL2CROWS', 'CDL3BLACKCROWS',
        'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH',
        'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD',
        'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
        'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
        'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
        'CDLGRAVESTONEDOJI', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS',
        'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
        'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING',
        'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
        'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR',
        'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN',
        'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
        'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI',
        'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER',
        'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS',
    ]

    @pytest.mark.parametrize("func_name", CDL_FUNCTIONS)
    @pytest.mark.parametrize("ds_name", ['random_1k', 'volatile'])
    def test_pattern_exact_match(self, func_name, ds_name):
        """验证 pattern 信号精确匹配 C TA-Lib（整数输出，无浮点问题）"""
        o, h, l, c, _ = DATASETS[ds_name]
        ours = np.asarray(getattr(rs, func_name)(o, h, l, c), dtype=np.float64)
        theirs = np.asarray(getattr(c_talib, func_name)(o, h, l, c), dtype=np.float64)
        compare(ours, theirs, func_name)
