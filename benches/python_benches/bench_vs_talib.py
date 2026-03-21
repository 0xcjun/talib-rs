"""
性能基准测试: talib-rs vs 原版 C TA-Lib

使用方法:
    pip install pytest-benchmark
    pytest benches/python_benches/bench_vs_talib.py -v
"""

import numpy as np
import pytest

# talib-rs 实现
from talib._talib import (
    SMA as RS_SMA, EMA as RS_EMA, RSI as RS_RSI,
    MACD as RS_MACD, BBANDS as RS_BBANDS, ATR as RS_ATR,
    ADX as RS_ADX, STOCH as RS_STOCH, CCI as RS_CCI,
    OBV as RS_OBV, CDLENGULFING as RS_CDLENGULFING,
)

# 尝试导入原版
try:
    import talib as orig
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False

# 测试数据
np.random.seed(42)
N = 10000
CLOSE = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, N)))
SPREAD = CLOSE * 0.02
HIGH = CLOSE + np.random.uniform(0, 1, N) * SPREAD
LOW = CLOSE - np.random.uniform(0, 1, N) * SPREAD
OPEN = LOW + np.random.uniform(0, 1, N) * (HIGH - LOW)
VOLUME = np.random.uniform(1e6, 1e7, N)


# ============================================================
# talib-rs benchmarks (always run)
# ============================================================

class TestTaRsBenchmarks:
    """talib-rs 性能基准"""

    def test_sma_10k(self, benchmark):
        benchmark(RS_SMA, CLOSE, 20)

    def test_ema_10k(self, benchmark):
        benchmark(RS_EMA, CLOSE, 20)

    def test_rsi_10k(self, benchmark):
        benchmark(RS_RSI, CLOSE, 14)

    def test_macd_10k(self, benchmark):
        benchmark(RS_MACD, CLOSE, 12, 26, 9)

    def test_bbands_10k(self, benchmark):
        benchmark(RS_BBANDS, CLOSE, 20, 2.0, 2.0, 0)

    def test_atr_10k(self, benchmark):
        benchmark(RS_ATR, HIGH, LOW, CLOSE, 14)

    def test_adx_10k(self, benchmark):
        benchmark(RS_ADX, HIGH, LOW, CLOSE, 14)

    def test_stoch_10k(self, benchmark):
        benchmark(RS_STOCH, HIGH, LOW, CLOSE, 5, 3, 0, 3, 0)

    def test_cci_10k(self, benchmark):
        benchmark(RS_CCI, HIGH, LOW, CLOSE, 14)

    def test_obv_10k(self, benchmark):
        benchmark(RS_OBV, CLOSE, VOLUME)

    def test_cdlengulfing_10k(self, benchmark):
        benchmark(RS_CDLENGULFING, OPEN, HIGH, LOW, CLOSE)


# ============================================================
# Original TA-Lib benchmarks (only when available)
# ============================================================

@pytest.mark.skipif(not HAS_ORIGINAL, reason="Original TA-Lib not installed")
class TestOriginalBenchmarks:
    """原版 C TA-Lib 性能基准"""

    def test_sma_10k(self, benchmark):
        benchmark(orig.SMA, CLOSE, timeperiod=20)

    def test_ema_10k(self, benchmark):
        benchmark(orig.EMA, CLOSE, timeperiod=20)

    def test_rsi_10k(self, benchmark):
        benchmark(orig.RSI, CLOSE, timeperiod=14)

    def test_macd_10k(self, benchmark):
        benchmark(orig.MACD, CLOSE, fastperiod=12, slowperiod=26, signalperiod=9)

    def test_bbands_10k(self, benchmark):
        benchmark(orig.BBANDS, CLOSE, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)

    def test_atr_10k(self, benchmark):
        benchmark(orig.ATR, HIGH, LOW, CLOSE, timeperiod=14)

    def test_adx_10k(self, benchmark):
        benchmark(orig.ADX, HIGH, LOW, CLOSE, timeperiod=14)

    def test_stoch_10k(self, benchmark):
        benchmark(orig.STOCH, HIGH, LOW, CLOSE, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    def test_cci_10k(self, benchmark):
        benchmark(orig.CCI, HIGH, LOW, CLOSE, timeperiod=14)

    def test_obv_10k(self, benchmark):
        benchmark(orig.OBV, CLOSE, VOLUME)

    def test_cdlengulfing_10k(self, benchmark):
        benchmark(orig.CDLENGULFING, OPEN, HIGH, LOW, CLOSE)


# ============================================================
# 大数据量测试 (100K)
# ============================================================

N_LARGE = 100000
CLOSE_LARGE = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, N_LARGE)))


class TestTaRsLargeBenchmarks:
    """talib-rs 大数据量基准 (100K 条)"""

    def test_sma_100k(self, benchmark):
        benchmark(RS_SMA, CLOSE_LARGE, 20)

    def test_ema_100k(self, benchmark):
        benchmark(RS_EMA, CLOSE_LARGE, 20)

    def test_rsi_100k(self, benchmark):
        benchmark(RS_RSI, CLOSE_LARGE, 14)

    def test_macd_100k(self, benchmark):
        benchmark(RS_MACD, CLOSE_LARGE, 12, 26, 9)
