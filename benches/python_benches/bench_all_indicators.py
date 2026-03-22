"""
全指标性能基准: talib-rs vs C TA-Lib
覆盖所有类别的指标，10K 数据量

运行: pytest benches/python_benches/bench_all_indicators.py -v --benchmark-sort=name --benchmark-columns=mean,stddev
"""

import numpy as np
import pytest

from talib_rs import _talib as rs

try:
    import talib as c_talib
    HAS_C = True
except ImportError:
    HAS_C = False

# ---- 测试数据 ----
np.random.seed(42)
N = 10000
CLOSE = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, N)))
SPREAD = CLOSE * 0.015
HIGH = CLOSE + np.random.uniform(0, 1, N) * SPREAD
LOW = CLOSE - np.random.uniform(0, 1, N) * SPREAD
OPEN = LOW + np.random.uniform(0, 1, N) * (HIGH - LOW)
VOLUME = np.random.uniform(1e6, 1e7, N)
CLOSE2 = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.025, N)))


# ============================================================
# talib-rs 性能基准
# ============================================================

class TestRsBenchAll:
    """talib-rs 全指标基准 (10K)"""

    # -- Overlap --
    def test_SMA(self, benchmark): benchmark(rs.SMA, CLOSE, 20)
    def test_EMA(self, benchmark): benchmark(rs.EMA, CLOSE, 20)
    def test_WMA(self, benchmark): benchmark(rs.WMA, CLOSE, 20)
    def test_DEMA(self, benchmark): benchmark(rs.DEMA, CLOSE, 20)
    def test_TEMA(self, benchmark): benchmark(rs.TEMA, CLOSE, 20)
    def test_TRIMA(self, benchmark): benchmark(rs.TRIMA, CLOSE, 20)
    def test_KAMA(self, benchmark): benchmark(rs.KAMA, CLOSE, 30)
    def test_T3(self, benchmark): benchmark(rs.T3, CLOSE, 5, 0.7)
    def test_BBANDS(self, benchmark): benchmark(rs.BBANDS, CLOSE, 20, 2.0, 2.0, 0)
    def test_SAR(self, benchmark): benchmark(rs.SAR, HIGH, LOW, 0.02, 0.2)
    def test_MIDPOINT(self, benchmark): benchmark(rs.MIDPOINT, CLOSE, 14)
    def test_MIDPRICE(self, benchmark): benchmark(rs.MIDPRICE, HIGH, LOW, 14)
    def test_HT_TRENDLINE(self, benchmark): benchmark(rs.HT_TRENDLINE, CLOSE)

    # -- Momentum --
    def test_RSI(self, benchmark): benchmark(rs.RSI, CLOSE, 14)
    def test_MACD(self, benchmark): benchmark(rs.MACD, CLOSE, 12, 26, 9)
    def test_MACDEXT(self, benchmark): benchmark(rs.MACDEXT, CLOSE, 12, 1, 26, 1, 9, 1)
    def test_MACDFIX(self, benchmark): benchmark(rs.MACDFIX, CLOSE, 9)
    def test_STOCH(self, benchmark): benchmark(rs.STOCH, HIGH, LOW, CLOSE, 5, 3, 0, 3, 0)
    def test_STOCHF(self, benchmark): benchmark(rs.STOCHF, HIGH, LOW, CLOSE, 5, 3, 0)
    def test_ADX(self, benchmark): benchmark(rs.ADX, HIGH, LOW, CLOSE, 14)
    def test_ADXR(self, benchmark): benchmark(rs.ADXR, HIGH, LOW, CLOSE, 14)
    def test_CCI(self, benchmark): benchmark(rs.CCI, HIGH, LOW, CLOSE, 14)
    def test_MOM(self, benchmark): benchmark(rs.MOM, CLOSE, 10)
    def test_ROC(self, benchmark): benchmark(rs.ROC, CLOSE, 10)
    def test_ROCP(self, benchmark): benchmark(rs.ROCP, CLOSE, 10)
    def test_ROCR(self, benchmark): benchmark(rs.ROCR, CLOSE, 10)
    def test_ROCR100(self, benchmark): benchmark(rs.ROCR100, CLOSE, 10)
    def test_WILLR(self, benchmark): benchmark(rs.WILLR, HIGH, LOW, CLOSE, 14)
    def test_APO(self, benchmark): benchmark(rs.APO, CLOSE, 12, 26, 0)
    def test_PPO(self, benchmark): benchmark(rs.PPO, CLOSE, 12, 26, 0)
    def test_BOP(self, benchmark): benchmark(rs.BOP, OPEN, HIGH, LOW, CLOSE)
    def test_CMO(self, benchmark): benchmark(rs.CMO, CLOSE, 14)
    def test_AROON(self, benchmark): benchmark(rs.AROON, HIGH, LOW, 14)
    def test_AROONOSC(self, benchmark): benchmark(rs.AROONOSC, HIGH, LOW, 14)
    def test_MFI(self, benchmark): benchmark(rs.MFI, HIGH, LOW, CLOSE, VOLUME, 14)
    def test_TRIX(self, benchmark): benchmark(rs.TRIX, CLOSE, 15)
    def test_ULTOSC(self, benchmark): benchmark(rs.ULTOSC, HIGH, LOW, CLOSE, 7, 14, 28)
    def test_DX(self, benchmark): benchmark(rs.DX, HIGH, LOW, CLOSE, 14)
    def test_PLUS_DI(self, benchmark): benchmark(rs.PLUS_DI, HIGH, LOW, CLOSE, 14)
    def test_MINUS_DI(self, benchmark): benchmark(rs.MINUS_DI, HIGH, LOW, CLOSE, 14)
    def test_PLUS_DM(self, benchmark): benchmark(rs.PLUS_DM, HIGH, LOW, 14)
    def test_MINUS_DM(self, benchmark): benchmark(rs.MINUS_DM, HIGH, LOW, 14)

    # -- Volatility --
    def test_ATR(self, benchmark): benchmark(rs.ATR, HIGH, LOW, CLOSE, 14)
    def test_NATR(self, benchmark): benchmark(rs.NATR, HIGH, LOW, CLOSE, 14)
    def test_TRANGE(self, benchmark): benchmark(rs.TRANGE, HIGH, LOW, CLOSE)

    # -- Volume --
    def test_AD(self, benchmark): benchmark(rs.AD, HIGH, LOW, CLOSE, VOLUME)
    def test_ADOSC(self, benchmark): benchmark(rs.ADOSC, HIGH, LOW, CLOSE, VOLUME, 3, 10)
    def test_OBV(self, benchmark): benchmark(rs.OBV, CLOSE, VOLUME)

    # -- Price Transform --
    def test_AVGPRICE(self, benchmark): benchmark(rs.AVGPRICE, OPEN, HIGH, LOW, CLOSE)
    def test_MEDPRICE(self, benchmark): benchmark(rs.MEDPRICE, HIGH, LOW)
    def test_TYPPRICE(self, benchmark): benchmark(rs.TYPPRICE, HIGH, LOW, CLOSE)
    def test_WCLPRICE(self, benchmark): benchmark(rs.WCLPRICE, HIGH, LOW, CLOSE)

    # -- Statistic --
    def test_STDDEV(self, benchmark): benchmark(rs.STDDEV, CLOSE, 20, 1.0)
    def test_VAR(self, benchmark): benchmark(rs.VAR, CLOSE, 20, 1.0)
    def test_BETA(self, benchmark): benchmark(rs.BETA, CLOSE, CLOSE2, 5)
    def test_CORREL(self, benchmark): benchmark(rs.CORREL, CLOSE, CLOSE2, 30)
    def test_LINEARREG(self, benchmark): benchmark(rs.LINEARREG, CLOSE, 14)
    def test_LINEARREG_SLOPE(self, benchmark): benchmark(rs.LINEARREG_SLOPE, CLOSE, 14)
    def test_TSF(self, benchmark): benchmark(rs.TSF, CLOSE, 14)

    # -- Math Transform --
    def test_SQRT(self, benchmark): benchmark(rs.SQRT, CLOSE)
    def test_LN(self, benchmark): benchmark(rs.LN, CLOSE)
    def test_SIN(self, benchmark): benchmark(rs.SIN, CLOSE)
    def test_EXP(self, benchmark):
        data = np.linspace(0, 5, N)  # 避免溢出
        benchmark(rs.EXP, data)

    # -- Math Operators --
    def test_ADD(self, benchmark): benchmark(rs.ADD, CLOSE, CLOSE2)
    def test_SUB(self, benchmark): benchmark(rs.SUB, CLOSE, CLOSE2)
    def test_MULT(self, benchmark): benchmark(rs.MULT, CLOSE, CLOSE2)
    def test_DIV(self, benchmark): benchmark(rs.DIV, CLOSE, CLOSE2)
    def test_MAX(self, benchmark): benchmark(rs.MAX, CLOSE, 30)
    def test_MIN(self, benchmark): benchmark(rs.MIN, CLOSE, 30)
    def test_SUM(self, benchmark): benchmark(rs.SUM, CLOSE, 30)

    # -- Cycle --
    def test_HT_DCPERIOD(self, benchmark): benchmark(rs.HT_DCPERIOD, CLOSE)
    def test_HT_DCPHASE(self, benchmark): benchmark(rs.HT_DCPHASE, CLOSE)
    def test_HT_SINE(self, benchmark): benchmark(rs.HT_SINE, CLOSE)
    def test_HT_TRENDMODE(self, benchmark): benchmark(rs.HT_TRENDMODE, CLOSE)

    # -- Pattern (sample) --
    def test_CDLDOJI(self, benchmark): benchmark(rs.CDLDOJI, OPEN, HIGH, LOW, CLOSE)
    def test_CDLHAMMER(self, benchmark): benchmark(rs.CDLHAMMER, OPEN, HIGH, LOW, CLOSE)
    def test_CDLENGULFING(self, benchmark): benchmark(rs.CDLENGULFING, OPEN, HIGH, LOW, CLOSE)
    def test_CDL3BLACKCROWS(self, benchmark): benchmark(rs.CDL3BLACKCROWS, OPEN, HIGH, LOW, CLOSE)
    def test_CDLMORNINGSTAR(self, benchmark): benchmark(rs.CDLMORNINGSTAR, OPEN, HIGH, LOW, CLOSE)


# ============================================================
# C TA-Lib 性能基准 (对照组)
# ============================================================

@pytest.mark.skipif(not HAS_C, reason="C TA-Lib not installed")
class TestCBenchAll:
    """C TA-Lib 全指标基准 (10K)"""

    # -- Overlap --
    def test_SMA(self, benchmark): benchmark(c_talib.SMA, CLOSE, timeperiod=20)
    def test_EMA(self, benchmark): benchmark(c_talib.EMA, CLOSE, timeperiod=20)
    def test_WMA(self, benchmark): benchmark(c_talib.WMA, CLOSE, timeperiod=20)
    def test_DEMA(self, benchmark): benchmark(c_talib.DEMA, CLOSE, timeperiod=20)
    def test_TEMA(self, benchmark): benchmark(c_talib.TEMA, CLOSE, timeperiod=20)
    def test_TRIMA(self, benchmark): benchmark(c_talib.TRIMA, CLOSE, timeperiod=20)
    def test_KAMA(self, benchmark): benchmark(c_talib.KAMA, CLOSE, timeperiod=30)
    def test_T3(self, benchmark): benchmark(c_talib.T3, CLOSE, timeperiod=5, vfactor=0.7)
    def test_BBANDS(self, benchmark): benchmark(c_talib.BBANDS, CLOSE, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    def test_SAR(self, benchmark): benchmark(c_talib.SAR, HIGH, LOW, acceleration=0.02, maximum=0.2)
    def test_MIDPOINT(self, benchmark): benchmark(c_talib.MIDPOINT, CLOSE, timeperiod=14)
    def test_MIDPRICE(self, benchmark): benchmark(c_talib.MIDPRICE, HIGH, LOW, timeperiod=14)
    def test_HT_TRENDLINE(self, benchmark): benchmark(c_talib.HT_TRENDLINE, CLOSE)

    # -- Momentum --
    def test_RSI(self, benchmark): benchmark(c_talib.RSI, CLOSE, timeperiod=14)
    def test_MACD(self, benchmark): benchmark(c_talib.MACD, CLOSE, fastperiod=12, slowperiod=26, signalperiod=9)
    def test_MACDEXT(self, benchmark): benchmark(c_talib.MACDEXT, CLOSE, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9, signalmatype=1)
    def test_MACDFIX(self, benchmark): benchmark(c_talib.MACDFIX, CLOSE, signalperiod=9)
    def test_STOCH(self, benchmark): benchmark(c_talib.STOCH, HIGH, LOW, CLOSE, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    def test_STOCHF(self, benchmark): benchmark(c_talib.STOCHF, HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3, fastd_matype=0)
    def test_ADX(self, benchmark): benchmark(c_talib.ADX, HIGH, LOW, CLOSE, timeperiod=14)
    def test_ADXR(self, benchmark): benchmark(c_talib.ADXR, HIGH, LOW, CLOSE, timeperiod=14)
    def test_CCI(self, benchmark): benchmark(c_talib.CCI, HIGH, LOW, CLOSE, timeperiod=14)
    def test_MOM(self, benchmark): benchmark(c_talib.MOM, CLOSE, timeperiod=10)
    def test_ROC(self, benchmark): benchmark(c_talib.ROC, CLOSE, timeperiod=10)
    def test_WILLR(self, benchmark): benchmark(c_talib.WILLR, HIGH, LOW, CLOSE, timeperiod=14)
    def test_APO(self, benchmark): benchmark(c_talib.APO, CLOSE, fastperiod=12, slowperiod=26, matype=0)
    def test_PPO(self, benchmark): benchmark(c_talib.PPO, CLOSE, fastperiod=12, slowperiod=26, matype=0)
    def test_BOP(self, benchmark): benchmark(c_talib.BOP, OPEN, HIGH, LOW, CLOSE)
    def test_CMO(self, benchmark): benchmark(c_talib.CMO, CLOSE, timeperiod=14)
    def test_AROON(self, benchmark): benchmark(c_talib.AROON, HIGH, LOW, timeperiod=14)
    def test_MFI(self, benchmark): benchmark(c_talib.MFI, HIGH, LOW, CLOSE, VOLUME, timeperiod=14)
    def test_TRIX(self, benchmark): benchmark(c_talib.TRIX, CLOSE, timeperiod=15)
    def test_ULTOSC(self, benchmark): benchmark(c_talib.ULTOSC, HIGH, LOW, CLOSE, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    def test_DX(self, benchmark): benchmark(c_talib.DX, HIGH, LOW, CLOSE, timeperiod=14)
    def test_PLUS_DI(self, benchmark): benchmark(c_talib.PLUS_DI, HIGH, LOW, CLOSE, timeperiod=14)
    def test_MINUS_DI(self, benchmark): benchmark(c_talib.MINUS_DI, HIGH, LOW, CLOSE, timeperiod=14)

    # -- Volatility --
    def test_ATR(self, benchmark): benchmark(c_talib.ATR, HIGH, LOW, CLOSE, timeperiod=14)
    def test_NATR(self, benchmark): benchmark(c_talib.NATR, HIGH, LOW, CLOSE, timeperiod=14)
    def test_TRANGE(self, benchmark): benchmark(c_talib.TRANGE, HIGH, LOW, CLOSE)

    # -- Volume --
    def test_AD(self, benchmark): benchmark(c_talib.AD, HIGH, LOW, CLOSE, VOLUME)
    def test_ADOSC(self, benchmark): benchmark(c_talib.ADOSC, HIGH, LOW, CLOSE, VOLUME, fastperiod=3, slowperiod=10)
    def test_OBV(self, benchmark): benchmark(c_talib.OBV, CLOSE, VOLUME)

    # -- Price Transform --
    def test_AVGPRICE(self, benchmark): benchmark(c_talib.AVGPRICE, OPEN, HIGH, LOW, CLOSE)
    def test_MEDPRICE(self, benchmark): benchmark(c_talib.MEDPRICE, HIGH, LOW)
    def test_TYPPRICE(self, benchmark): benchmark(c_talib.TYPPRICE, HIGH, LOW, CLOSE)

    # -- Statistic --
    def test_STDDEV(self, benchmark): benchmark(c_talib.STDDEV, CLOSE, timeperiod=20, nbdev=1.0)
    def test_VAR(self, benchmark): benchmark(c_talib.VAR, CLOSE, timeperiod=20, nbdev=1.0)
    def test_BETA(self, benchmark): benchmark(c_talib.BETA, CLOSE, CLOSE2, timeperiod=5)
    def test_CORREL(self, benchmark): benchmark(c_talib.CORREL, CLOSE, CLOSE2, timeperiod=30)
    def test_LINEARREG(self, benchmark): benchmark(c_talib.LINEARREG, CLOSE, timeperiod=14)

    # -- Math --
    def test_SQRT(self, benchmark): benchmark(c_talib.SQRT, CLOSE)
    def test_LN(self, benchmark): benchmark(c_talib.LN, CLOSE)
    def test_ADD(self, benchmark): benchmark(c_talib.ADD, CLOSE, CLOSE2)
    def test_MAX(self, benchmark): benchmark(c_talib.MAX, CLOSE, timeperiod=30)
    def test_SUM(self, benchmark): benchmark(c_talib.SUM, CLOSE, timeperiod=30)

    # -- Cycle --
    def test_HT_DCPERIOD(self, benchmark): benchmark(c_talib.HT_DCPERIOD, CLOSE)
    def test_HT_SINE(self, benchmark): benchmark(c_talib.HT_SINE, CLOSE)
    def test_HT_TRENDMODE(self, benchmark): benchmark(c_talib.HT_TRENDMODE, CLOSE)

    # -- Pattern --
    def test_CDLDOJI(self, benchmark): benchmark(c_talib.CDLDOJI, OPEN, HIGH, LOW, CLOSE)
    def test_CDLENGULFING(self, benchmark): benchmark(c_talib.CDLENGULFING, OPEN, HIGH, LOW, CLOSE)
    def test_CDLMORNINGSTAR(self, benchmark): benchmark(c_talib.CDLMORNINGSTAR, OPEN, HIGH, LOW, CLOSE)
