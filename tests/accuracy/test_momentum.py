"""Momentum Indicators 精度验证: talib-rs vs C TA-Lib"""

import numpy as np
import pytest

try:
    import talib as original_talib
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False

from talib._talib import RSI, MACD, STOCH, ADX, CCI, MOM, ROC, WILLR, CMO, AROON
from conftest import make_close, make_ohlcv, assert_talib_equal


class TestRSI:
    """RSI 精度测试"""

    def test_range(self):
        """RSI 应在 [0, 100] 范围内"""
        close = make_close(500)
        result = RSI(close, timeperiod=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_uptrend_high(self):
        """持续上涨时 RSI 应接近 100"""
        close = np.arange(1.0, 51.0)
        result = RSI(close, timeperiod=14)
        assert result[20] > 95.0

    def test_downtrend_low(self):
        """持续下跌时 RSI 应接近 0"""
        close = np.arange(50.0, 0.0, -1.0)
        result = RSI(close, timeperiod=14)
        assert result[20] < 5.0

    @pytest.mark.skipif(not HAS_ORIGINAL, reason="Original TA-Lib not installed")
    def test_vs_original(self):
        close = make_close(1000)
        for period in [5, 14, 21]:
            ours = RSI(close, timeperiod=period)
            theirs = original_talib.RSI(close, timeperiod=period)
            assert_talib_equal(ours, theirs)


class TestMACD:
    """MACD 精度测试"""

    def test_histogram_equals_diff(self):
        """histogram 应等于 macd - signal"""
        close = make_close(200)
        macd_line, signal, hist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        for i in range(len(close)):
            if not np.isnan(macd_line[i]) and not np.isnan(signal[i]) and not np.isnan(hist[i]):
                np.testing.assert_almost_equal(hist[i], macd_line[i] - signal[i], decimal=10)

    @pytest.mark.skipif(not HAS_ORIGINAL, reason="Original TA-Lib not installed")
    def test_vs_original(self):
        close = make_close(500)
        ours = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        theirs = original_talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        for o, t in zip(ours, theirs):
            assert_talib_equal(o, t)


class TestADX:
    """ADX 精度测试"""

    def test_range(self):
        """ADX 应在 [0, 100] 范围内"""
        _, high, low, close, _ = make_ohlcv(200)
        result = ADX(high, low, close, timeperiod=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    @pytest.mark.skipif(not HAS_ORIGINAL, reason="Original TA-Lib not installed")
    def test_vs_original(self):
        _, high, low, close, _ = make_ohlcv(500)
        ours = ADX(high, low, close, timeperiod=14)
        theirs = original_talib.ADX(high, low, close, timeperiod=14)
        assert_talib_equal(ours, theirs)


class TestMOM:
    """Momentum 精度测试"""

    def test_basic(self):
        close = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
        result = MOM(close, timeperiod=3)
        np.testing.assert_almost_equal(result[3], 5.0 - 1.0)  # 4.0
        np.testing.assert_almost_equal(result[4], 8.0 - 2.0)  # 6.0
        np.testing.assert_almost_equal(result[5], 13.0 - 3.0) # 10.0


class TestROC:
    """Rate of Change 精度测试"""

    def test_basic(self):
        close = np.array([10.0, 20.0, 15.0, 30.0])
        result = ROC(close, timeperiod=1)
        np.testing.assert_almost_equal(result[1], 100.0)   # (20-10)/10*100
        np.testing.assert_almost_equal(result[2], -25.0)    # (15-20)/20*100
        np.testing.assert_almost_equal(result[3], 100.0)    # (30-15)/15*100
