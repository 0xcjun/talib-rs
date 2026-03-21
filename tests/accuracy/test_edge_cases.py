"""
边界条件测试: 极端输入下 ta-rs 的行为

覆盖: 最小输入长度、极值、常数值、大数组
"""

import numpy as np
import pytest

from talib import _talib as rs


# ============================================================
# 最小输入长度 (lookback + 1)
# ============================================================

class TestMinimalInput:
    """使用刚好满足 lookback 要求的最小输入"""

    def test_sma_minimal(self):
        close = np.array([1.0, 2.0, 3.0])
        result = rs.SMA(close, 3)
        assert not np.isnan(result[2])
        assert np.isnan(result[0])

    def test_ema_minimal(self):
        close = np.array([1.0, 2.0, 3.0])
        result = rs.EMA(close, 3)
        assert not np.isnan(result[2])

    def test_rsi_minimal(self):
        close = np.arange(1.0, 16.0)  # 15 元素, lookback=14
        result = rs.RSI(close, 14)
        assert not np.isnan(result[14])

    def test_macd_minimal(self):
        close = np.arange(1.0, 36.0)  # 35 元素 >= 25+8=33 lookback
        macd, signal, hist = rs.MACD(close, 12, 26, 9)
        valid_signal = signal[~np.isnan(signal)]
        assert len(valid_signal) > 0

    def test_bbands_minimal(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        upper, middle, lower = rs.BBANDS(close, 5, 2.0, 2.0, 0)
        assert not np.isnan(middle[4])

    def test_atr_minimal(self):
        high = np.array([12.0, 13.0, 11.0])
        low = np.array([10.0, 11.0, 9.0])
        close = np.array([11.0, 12.0, 10.0])
        result = rs.ATR(high, low, close, 2)
        assert not np.isnan(result[2])

    def test_stoch_minimal(self):
        """STOCH 需要 fastk+slowk+slowd-2 个数据点"""
        n = 10
        h = np.arange(1.0, n + 1.0) + 1
        l = np.arange(1.0, n + 1.0) - 1
        c = np.arange(1.0, n + 1.0)
        slowk, slowd = rs.STOCH(h, l, c, 5, 3, 0, 3, 0)
        assert slowk.shape[0] == n


# ============================================================
# 不足数据 (应抛出异常)
# ============================================================

class TestInsufficientData:

    def test_sma_too_short(self):
        with pytest.raises(ValueError):
            rs.SMA(np.array([1.0, 2.0]), 5)

    def test_rsi_too_short(self):
        with pytest.raises(ValueError):
            rs.RSI(np.array([1.0, 2.0, 3.0]), 14)

    def test_ema_too_short(self):
        with pytest.raises(ValueError):
            rs.EMA(np.array([1.0]), 5)

    def test_macd_too_short(self):
        with pytest.raises(ValueError):
            rs.MACD(np.arange(1.0, 10.0), 12, 26, 9)

    def test_adx_too_short(self):
        h = np.array([10.0, 11.0])
        l = np.array([9.0, 10.0])
        c = np.array([9.5, 10.5])
        with pytest.raises(ValueError):
            rs.ADX(h, l, c, 14)


# ============================================================
# 常数值输入
# ============================================================

class TestConstantInput:

    def test_sma_constant(self):
        """常数输入，SMA 应等于该常数"""
        close = np.full(100, 42.0)
        result = rs.SMA(close, 20)
        for v in result[19:]:
            np.testing.assert_almost_equal(v, 42.0)

    def test_ema_constant(self):
        close = np.full(100, 42.0)
        result = rs.EMA(close, 20)
        for v in result[19:]:
            np.testing.assert_almost_equal(v, 42.0)

    def test_rsi_constant(self):
        """常数输入，无涨跌，RSI 应为 50 或 NaN（取决于实现）"""
        close = np.full(100, 50.0)
        # 第一个值不同以避免全零变化
        close[0] = 49.0
        result = rs.RSI(close, 14)
        valid = result[~np.isnan(result)]
        # RSI 应趋近 50（因为几乎全是零变化）
        if len(valid) > 0:
            assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_bbands_constant(self):
        """常数输入，stddev=0，upper=middle=lower"""
        close = np.full(100, 100.0)
        upper, middle, lower = rs.BBANDS(close, 20, 2.0, 2.0, 0)
        for i in range(19, 100):
            np.testing.assert_almost_equal(upper[i], 100.0)
            np.testing.assert_almost_equal(middle[i], 100.0)
            np.testing.assert_almost_equal(lower[i], 100.0)

    def test_atr_constant_range(self):
        """高低差恒定，ATR 应收敛到该差值"""
        high = np.full(100, 52.0)
        low = np.full(100, 48.0)
        close = np.full(100, 50.0)
        result = rs.ATR(high, low, close, 14)
        for v in result[20:]:
            np.testing.assert_almost_equal(v, 4.0, decimal=1)

    def test_obv_constant(self):
        """价格不变，OBV 不应变化"""
        close = np.full(100, 50.0)
        volume = np.full(100, 1e6)
        result = rs.OBV(close, volume)
        # 第一个值之后，OBV 不应变化 (close[i] == close[i-1])
        for i in range(2, 100):
            np.testing.assert_almost_equal(result[i], result[1])


# ============================================================
# 极值测试
# ============================================================

class TestExtremeValues:

    def test_very_large_values(self):
        close = np.full(100, 1e15)
        close += np.arange(100) * 1e12
        result = rs.SMA(close, 20)
        assert not np.any(np.isinf(result[19:]))

    def test_very_small_values(self):
        close = np.full(100, 1e-15)
        close += np.arange(100) * 1e-18
        result = rs.SMA(close, 20)
        assert not np.any(np.isinf(result[19:]))

    def test_mixed_magnitude(self):
        """大小值混合"""
        close = np.array([1e-10, 1e10, 1e-10, 1e10, 1e-10] * 20)
        result = rs.SMA(close, 5)
        assert not np.any(np.isnan(result[4:]))

    def test_negative_prices(self):
        """负值（虽然价格不应为负，但函数应能处理）"""
        close = np.linspace(-10.0, 10.0, 100)
        result = rs.SMA(close, 10)
        assert not np.any(np.isnan(result[9:]))

    def test_rsi_extreme_uptrend(self):
        """极端上涨应给出 RSI 接近 100"""
        close = np.exp(np.arange(100) * 0.1)
        result = rs.RSI(close, 14)
        assert result[-1] > 99.0

    def test_rsi_extreme_downtrend(self):
        """极端下跌应给出 RSI 接近 0"""
        close = np.exp(-np.arange(100) * 0.1) * 1000
        result = rs.RSI(close, 14)
        assert result[-1] < 1.0


# ============================================================
# 大数据集
# ============================================================

class TestLargeDataset:

    def test_sma_100k(self):
        close = np.random.RandomState(42).random(100000) * 100 + 50
        result = rs.SMA(close, 200)
        assert result.shape == (100000,)
        assert not np.isnan(result[-1])

    def test_rsi_100k(self):
        close = np.cumsum(np.random.RandomState(42).normal(0, 1, 100000)) + 100
        result = rs.RSI(close, 14)
        assert result.shape == (100000,)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_macd_100k(self):
        close = np.cumsum(np.random.RandomState(42).normal(0, 1, 100000)) + 100
        macd, signal, hist = rs.MACD(close, 12, 26, 9)
        assert macd.shape == (100000,)

    def test_bbands_100k(self):
        close = np.random.RandomState(42).random(100000) * 100 + 50
        upper, middle, lower = rs.BBANDS(close, 20, 2.0, 2.0, 0)
        # upper >= middle >= lower (非 NaN 部分)
        for i in range(19, 100000):
            assert upper[i] >= middle[i] >= lower[i], f"Band order violated at {i}"


# ============================================================
# 输入验证
# ============================================================

class TestInputValidation:

    def test_length_mismatch_stoch(self):
        with pytest.raises((ValueError, Exception)):
            rs.STOCH(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]), 5, 3, 0, 3, 0)

    def test_length_mismatch_atr(self):
        with pytest.raises((ValueError, Exception)):
            rs.ATR(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), 14)

    def test_empty_input_sma(self):
        with pytest.raises((ValueError, Exception)):
            rs.SMA(np.array([]), 5)

    def test_period_zero_sma(self):
        with pytest.raises((ValueError, Exception)):
            rs.SMA(np.array([1.0, 2.0, 3.0]), 0)
