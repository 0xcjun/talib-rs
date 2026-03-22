"""
Stream API -- computes only the latest value of each indicator.

Compatible with the original ``talib.stream`` module. Each function accepts
the same arguments as the regular (batch) API but returns only the most
recent output value(s) instead of a full array.

Usage:
    from talib_rs import stream
    latest_sma = stream.SMA(close, timeperiod=20)   # single float
    macd, signal, hist = stream.MACD(close)          # tuple of floats
"""

from __future__ import annotations

import numpy as np
import talib_rs._talib as _talib


def _last(result):
    """
    Extract the last non-NaN value from a result.
    If the result is a tuple of arrays, return a tuple of last values.
    If it is a single array, return a single float.
    """
    if isinstance(result, tuple):
        return tuple(_scalar(arr) for arr in result)
    return _scalar(result)


def _scalar(arr):
    """Return the last element of an array as a Python float/int."""
    a = np.asarray(arr)
    if a.size == 0:
        return float('nan')
    return float(a[-1])


def _stream_wrapper(func_name):
    """Create a stream wrapper for a given _talib function."""
    func = getattr(_talib, func_name)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return _last(result)
    wrapper.__name__ = func_name
    wrapper.__qualname__ = func_name
    wrapper.__doc__ = (
        f"Stream version of {func_name}. Returns only the latest value(s).\n"
        f"Accepts the same arguments as talib.{func_name}."
    )
    return wrapper


# ============================================================
# Overlap Studies
# ============================================================

SMA = _stream_wrapper('SMA')
EMA = _stream_wrapper('EMA')
WMA = _stream_wrapper('WMA')
DEMA = _stream_wrapper('DEMA')
TEMA = _stream_wrapper('TEMA')
TRIMA = _stream_wrapper('TRIMA')
KAMA = _stream_wrapper('KAMA')
T3 = _stream_wrapper('T3')
MAMA = _stream_wrapper('MAMA')
BBANDS = _stream_wrapper('BBANDS')
SAR = _stream_wrapper('SAR')
SAREXT = _stream_wrapper('SAREXT')
MIDPOINT = _stream_wrapper('MIDPOINT')
MIDPRICE = _stream_wrapper('MIDPRICE')
MAVP = _stream_wrapper('MAVP')
HT_TRENDLINE = _stream_wrapper('HT_TRENDLINE')

# ============================================================
# Momentum Indicators
# ============================================================

RSI = _stream_wrapper('RSI')
MACD = _stream_wrapper('MACD')
MACDEXT = _stream_wrapper('MACDEXT')
MACDFIX = _stream_wrapper('MACDFIX')
STOCH = _stream_wrapper('STOCH')
STOCHF = _stream_wrapper('STOCHF')
STOCHRSI = _stream_wrapper('STOCHRSI')
ADX = _stream_wrapper('ADX')
ADXR = _stream_wrapper('ADXR')
CCI = _stream_wrapper('CCI')
MOM = _stream_wrapper('MOM')
ROC = _stream_wrapper('ROC')
ROCP = _stream_wrapper('ROCP')
ROCR = _stream_wrapper('ROCR')
ROCR100 = _stream_wrapper('ROCR100')
WILLR = _stream_wrapper('WILLR')
APO = _stream_wrapper('APO')
PPO = _stream_wrapper('PPO')
BOP = _stream_wrapper('BOP')
CMO = _stream_wrapper('CMO')
AROON = _stream_wrapper('AROON')
AROONOSC = _stream_wrapper('AROONOSC')
MFI = _stream_wrapper('MFI')
TRIX = _stream_wrapper('TRIX')
ULTOSC = _stream_wrapper('ULTOSC')
DX = _stream_wrapper('DX')
PLUS_DI = _stream_wrapper('PLUS_DI')
MINUS_DI = _stream_wrapper('MINUS_DI')
PLUS_DM = _stream_wrapper('PLUS_DM')
MINUS_DM = _stream_wrapper('MINUS_DM')

# ============================================================
# Volatility
# ============================================================

ATR = _stream_wrapper('ATR')
NATR = _stream_wrapper('NATR')
TRANGE = _stream_wrapper('TRANGE')

# ============================================================
# Volume
# ============================================================

AD = _stream_wrapper('AD')
ADOSC = _stream_wrapper('ADOSC')
OBV = _stream_wrapper('OBV')

# ============================================================
# Price Transform
# ============================================================

AVGPRICE = _stream_wrapper('AVGPRICE')
MEDPRICE = _stream_wrapper('MEDPRICE')
TYPPRICE = _stream_wrapper('TYPPRICE')
WCLPRICE = _stream_wrapper('WCLPRICE')

# ============================================================
# Statistic Functions
# ============================================================

STDDEV = _stream_wrapper('STDDEV')
VAR = _stream_wrapper('VAR')
BETA = _stream_wrapper('BETA')
CORREL = _stream_wrapper('CORREL')
LINEARREG = _stream_wrapper('LINEARREG')
LINEARREG_SLOPE = _stream_wrapper('LINEARREG_SLOPE')
LINEARREG_INTERCEPT = _stream_wrapper('LINEARREG_INTERCEPT')
LINEARREG_ANGLE = _stream_wrapper('LINEARREG_ANGLE')
TSF = _stream_wrapper('TSF')

# ============================================================
# Math Transform
# ============================================================

ACOS = _stream_wrapper('ACOS')
ASIN = _stream_wrapper('ASIN')
ATAN = _stream_wrapper('ATAN')
CEIL = _stream_wrapper('CEIL')
COS = _stream_wrapper('COS')
COSH = _stream_wrapper('COSH')
EXP = _stream_wrapper('EXP')
FLOOR = _stream_wrapper('FLOOR')
LN = _stream_wrapper('LN')
LOG10 = _stream_wrapper('LOG10')
SIN = _stream_wrapper('SIN')
SINH = _stream_wrapper('SINH')
SQRT = _stream_wrapper('SQRT')
TAN = _stream_wrapper('TAN')
TANH = _stream_wrapper('TANH')

# ============================================================
# Math Operators
# ============================================================

ADD = _stream_wrapper('ADD')
SUB = _stream_wrapper('SUB')
MULT = _stream_wrapper('MULT')
DIV = _stream_wrapper('DIV')
MAX = _stream_wrapper('MAX')
MAXINDEX = _stream_wrapper('MAXINDEX')
MIN = _stream_wrapper('MIN')
MININDEX = _stream_wrapper('MININDEX')
SUM = _stream_wrapper('SUM')

# ============================================================
# Cycle Indicators
# ============================================================

HT_DCPERIOD = _stream_wrapper('HT_DCPERIOD')
HT_DCPHASE = _stream_wrapper('HT_DCPHASE')
HT_PHASOR = _stream_wrapper('HT_PHASOR')
HT_SINE = _stream_wrapper('HT_SINE')
HT_TRENDMODE = _stream_wrapper('HT_TRENDMODE')

# ============================================================
# Pattern Recognition
# ============================================================

CDLDOJI = _stream_wrapper('CDLDOJI')
CDLHAMMER = _stream_wrapper('CDLHAMMER')
CDLENGULFING = _stream_wrapper('CDLENGULFING')
CDL2CROWS = _stream_wrapper('CDL2CROWS')
CDL3BLACKCROWS = _stream_wrapper('CDL3BLACKCROWS')
CDL3INSIDE = _stream_wrapper('CDL3INSIDE')
CDL3LINESTRIKE = _stream_wrapper('CDL3LINESTRIKE')
CDL3OUTSIDE = _stream_wrapper('CDL3OUTSIDE')
CDL3STARSINSOUTH = _stream_wrapper('CDL3STARSINSOUTH')
CDL3WHITESOLDIERS = _stream_wrapper('CDL3WHITESOLDIERS')
CDLABANDONEDBABY = _stream_wrapper('CDLABANDONEDBABY')
CDLADVANCEBLOCK = _stream_wrapper('CDLADVANCEBLOCK')
CDLBELTHOLD = _stream_wrapper('CDLBELTHOLD')
CDLBREAKAWAY = _stream_wrapper('CDLBREAKAWAY')
CDLCLOSINGMARUBOZU = _stream_wrapper('CDLCLOSINGMARUBOZU')
CDLCONCEALBABYSWALL = _stream_wrapper('CDLCONCEALBABYSWALL')
CDLCOUNTERATTACK = _stream_wrapper('CDLCOUNTERATTACK')
CDLDARKCLOUDCOVER = _stream_wrapper('CDLDARKCLOUDCOVER')
CDLDOJISTAR = _stream_wrapper('CDLDOJISTAR')
CDLDRAGONFLYDOJI = _stream_wrapper('CDLDRAGONFLYDOJI')
CDLEVENINGDOJISTAR = _stream_wrapper('CDLEVENINGDOJISTAR')
CDLEVENINGSTAR = _stream_wrapper('CDLEVENINGSTAR')
CDLGAPSIDESIDEWHITE = _stream_wrapper('CDLGAPSIDESIDEWHITE')
CDLGRAVESTONEDOJI = _stream_wrapper('CDLGRAVESTONEDOJI')
CDLHANGINGMAN = _stream_wrapper('CDLHANGINGMAN')
CDLHARAMI = _stream_wrapper('CDLHARAMI')
CDLHARAMICROSS = _stream_wrapper('CDLHARAMICROSS')
CDLHIGHWAVE = _stream_wrapper('CDLHIGHWAVE')
CDLHIKKAKE = _stream_wrapper('CDLHIKKAKE')
CDLHIKKAKEMOD = _stream_wrapper('CDLHIKKAKEMOD')
CDLHOMINGPIGEON = _stream_wrapper('CDLHOMINGPIGEON')
CDLIDENTICAL3CROWS = _stream_wrapper('CDLIDENTICAL3CROWS')
CDLINNECK = _stream_wrapper('CDLINNECK')
CDLINVERTEDHAMMER = _stream_wrapper('CDLINVERTEDHAMMER')
CDLKICKING = _stream_wrapper('CDLKICKING')
CDLKICKINGBYLENGTH = _stream_wrapper('CDLKICKINGBYLENGTH')
CDLLADDERBOTTOM = _stream_wrapper('CDLLADDERBOTTOM')
CDLLONGLEGGEDDOJI = _stream_wrapper('CDLLONGLEGGEDDOJI')
CDLLONGLINE = _stream_wrapper('CDLLONGLINE')
CDLMARUBOZU = _stream_wrapper('CDLMARUBOZU')
CDLMATCHINGLOW = _stream_wrapper('CDLMATCHINGLOW')
CDLMATHOLD = _stream_wrapper('CDLMATHOLD')
CDLMORNINGDOJISTAR = _stream_wrapper('CDLMORNINGDOJISTAR')
CDLMORNINGSTAR = _stream_wrapper('CDLMORNINGSTAR')
CDLONNECK = _stream_wrapper('CDLONNECK')
CDLPIERCING = _stream_wrapper('CDLPIERCING')
CDLRICKSHAWMAN = _stream_wrapper('CDLRICKSHAWMAN')
CDLRISEFALL3METHODS = _stream_wrapper('CDLRISEFALL3METHODS')
CDLSEPARATINGLINES = _stream_wrapper('CDLSEPARATINGLINES')
CDLSHOOTINGSTAR = _stream_wrapper('CDLSHOOTINGSTAR')
CDLSHORTLINE = _stream_wrapper('CDLSHORTLINE')
CDLSPINNINGTOP = _stream_wrapper('CDLSPINNINGTOP')
CDLSTALLEDPATTERN = _stream_wrapper('CDLSTALLEDPATTERN')
CDLSTICKSANDWICH = _stream_wrapper('CDLSTICKSANDWICH')
CDLTAKURI = _stream_wrapper('CDLTAKURI')
CDLTASUKIGAP = _stream_wrapper('CDLTASUKIGAP')
CDLTHRUSTING = _stream_wrapper('CDLTHRUSTING')
CDLTRISTAR = _stream_wrapper('CDLTRISTAR')
CDLUNIQUE3RIVER = _stream_wrapper('CDLUNIQUE3RIVER')
CDLUPSIDEGAP2CROWS = _stream_wrapper('CDLUPSIDEGAP2CROWS')
CDLXSIDEGAP3METHODS = _stream_wrapper('CDLXSIDEGAP3METHODS')
