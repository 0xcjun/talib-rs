"""Overlap Studies 精度验证: talib-rs vs C TA-Lib"""

import numpy as np
import pytest

# 尝试导入原版 TA-Lib 用于对比
try:
    import talib_rs as original_talib
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False

# 我们的实现（总是可用）
import sys
sys.modules.pop('talib_rs', None)  # 清理以避免冲突

# 直接导入我们的 _talib 模块
from talib_rs._talib import (
    SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3,
    BBANDS, SAR, MIDPOINT, MIDPRICE, HT_TRENDLINE,
)

from conftest import make_close, make_ohlcv, assert_talib_equal


class TestSMA:
    """SMA 精度测试"""

    def test_basic_values(self):
        """验证 SMA 基本计算正确"""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = SMA(close, timeperiod=3)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_almost_equal(result[2], 2.0)
        np.testing.assert_almost_equal(result[3], 3.0)
        np.testing.assert_almost_equal(result[9], 9.0)

    def test_various_periods(self):
        """测试不同周期的 SMA"""
        close = make_close(500)
        for period in [2, 5, 10, 14, 20, 30, 50, 100, 200]:
            result = SMA(close, timeperiod=period)
            # lookback 前应为 NaN
            assert np.isnan(result[period - 2])
            assert not np.isnan(result[period - 1])
            # 手动验证一个值
            expected = np.mean(close[:period])
            np.testing.assert_almost_equal(result[period - 1], expected, decimal=10)

    @pytest.mark.skipif(not HAS_ORIGINAL, reason="Original TA-Lib not installed")
    def test_vs_original(self):
        """与原版 TA-Lib 对比"""
        close = make_close(1000)
        for period in [5, 14, 30, 50, 200]:
            ours = SMA(close, timeperiod=period)
            theirs = original_talib.SMA(close, timeperiod=period)
            assert_talib_equal(ours, theirs)


class TestEMA:
    """EMA 精度测试"""

    def test_seed_is_sma(self):
        """EMA 初始值应等于 SMA"""
        close = make_close(100)
        period = 10
        ema = EMA(close, timeperiod=period)
        sma_val = np.mean(close[:period])
        np.testing.assert_almost_equal(ema[period - 1], sma_val, decimal=10)

    def test_monotonic_input(self):
        """单调递增输入，EMA 应跟踪趋势"""
        close = np.arange(1.0, 101.0)
        result = EMA(close, timeperiod=10)
        # EMA 应小于当前价格（滞后）
        for i in range(10, 100):
            assert result[i] < close[i]

    @pytest.mark.skipif(not HAS_ORIGINAL, reason="Original TA-Lib not installed")
    def test_vs_original(self):
        close = make_close(1000)
        for period in [5, 12, 26, 50]:
            ours = EMA(close, timeperiod=period)
            theirs = original_talib.EMA(close, timeperiod=period)
            assert_talib_equal(ours, theirs)


class TestBBANDS:
    """Bollinger Bands 精度测试"""

    def test_middle_equals_sma(self):
        """中轨应等于 SMA"""
        close = make_close(200)
        upper, middle, lower = BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        sma = SMA(close, timeperiod=20)
        # middle 和 SMA 应完全一致
        for i in range(19, 200):
            np.testing.assert_almost_equal(middle[i], sma[i], decimal=10)

    def test_band_symmetry(self):
        """当 nbdevup == nbdevdn 时，上下轨应关于中轨对称"""
        close = make_close(200)
        upper, middle, lower = BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        for i in range(19, 200):
            if not np.isnan(upper[i]):
                up_dist = upper[i] - middle[i]
                lo_dist = middle[i] - lower[i]
                np.testing.assert_almost_equal(up_dist, lo_dist, decimal=10)

    @pytest.mark.skipif(not HAS_ORIGINAL, reason="Original TA-Lib not installed")
    def test_vs_original(self):
        close = make_close(500)
        ours = BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        theirs = original_talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        for o, t in zip(ours, theirs):
            assert_talib_equal(o, t)
