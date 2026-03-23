"""
Real-World Data Alignment Test: talib-rs vs C TA-Lib

Uses actual market data to validate numerical alignment:
  1. 数字币1小时行情 (crypto hourly OHLCV, 1.5M rows, 100+ symbols)
  2. 全A日线 (A-share daily price, 8.4M rows, 5000+ symbols)

Tolerances: same as C TA-Lib comparison, NO relaxation for passing.
  - exact:  rtol=1e-14, atol=0      (element-wise, no accumulation)
  - tight:  rtol=1e-10, atol=1e-12  (serial EMA/Wilder chains)
  - sliding: rtol=1e-8, atol=1e-10  (sliding sum algorithms)

Run:
  pytest tests/accuracy/test_real_data_alignment.py -v
  pytest tests/accuracy/test_real_data_alignment.py -v -k crypto
  pytest tests/accuracy/test_real_data_alignment.py -v -k ashare
"""

import numpy as np
import pytest
import talib as c_talib
from talib_rs import _talib as rs

# ============================================================
# Data loading
# ============================================================

try:
    import pandas as pd
    _CRYPTO_PATH = '/Volumes/jun/数字币1小时行情_210101_230101.feather'
    _ASHARE_PATH = '/Volumes/jun/全A日线测试_20170101_20250429.feather'

    _crypto_df = pd.read_feather(_CRYPTO_PATH)
    _ashare_df = pd.read_feather(_ASHARE_PATH)

    # Group crypto by symbol, pick diverse samples (different price ranges + lengths)
    _crypto_groups = dict(list(_crypto_df.groupby('symbol')))
    _crypto_symbols = sorted(_crypto_groups.keys())

    # Pick 10 representative crypto symbols with varying characteristics
    _crypto_sample_symbols = []
    if len(_crypto_symbols) >= 10:
        step = len(_crypto_symbols) // 10
        _crypto_sample_symbols = [_crypto_symbols[i * step] for i in range(10)]
    else:
        _crypto_sample_symbols = _crypto_symbols

    # Group A-share by symbol, pick samples
    _ashare_groups = dict(list(_ashare_df.groupby('symbol')))
    _ashare_symbols = sorted(_ashare_groups.keys())
    _ashare_sample_symbols = []
    if len(_ashare_symbols) >= 20:
        step = len(_ashare_symbols) // 20
        _ashare_sample_symbols = [_ashare_symbols[i * step] for i in range(20)]
    else:
        _ashare_sample_symbols = _ashare_symbols

    DATA_AVAILABLE = True
except Exception as e:
    DATA_AVAILABLE = False
    _DATA_ERROR = str(e)


def _get_crypto_ohlcv(symbol):
    """Get OHLCV arrays for a crypto symbol."""
    df = _crypto_groups[symbol].sort_values('dt').reset_index(drop=True)
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['vol'].values.astype(np.float64)
    return o, h, l, c, v


def _get_ashare_price(symbol):
    """Get price array for an A-share symbol."""
    df = _ashare_groups[symbol].sort_values('dt').reset_index(drop=True)
    return df['price'].values.astype(np.float64)


# ============================================================
# Comparison engine
# ============================================================

def _compare(ours, theirs, name, rtol, atol):
    """Compare results. Returns (pass, max_diff, description)."""
    if isinstance(ours, tuple) and isinstance(theirs, tuple):
        assert len(ours) == len(theirs), f"{name}: tuple length mismatch"
        max_d = 0.0
        for i, (o, t) in enumerate(zip(ours, theirs)):
            oa = np.asarray(o, dtype=np.float64)
            ta = np.asarray(t, dtype=np.float64)
            d = _compare_arrays(oa, ta, f"{name}[{i}]", rtol, atol)
            max_d = max(max_d, d)
        return max_d
    else:
        oa = np.asarray(ours, dtype=np.float64)
        ta = np.asarray(theirs, dtype=np.float64)
        return _compare_arrays(oa, ta, name, rtol, atol)


def _compare_arrays(ours, theirs, name, rtol, atol):
    """Compare two arrays. Raises on mismatch. Returns max abs diff."""
    assert ours.shape == theirs.shape, f"{name}: shape {ours.shape} vs {theirs.shape}"

    # NaN positions must match exactly
    our_nan = np.isnan(ours)
    their_nan = np.isnan(theirs)
    nan_mismatch = our_nan != their_nan
    if np.any(nan_mismatch):
        idx = np.where(nan_mismatch)[0][0]
        pytest.fail(f"{name}: NaN mismatch at index {idx}, "
                    f"ours={ours[idx]}, theirs={theirs[idx]}")

    valid = ~our_nan
    if not np.any(valid):
        return 0.0

    ov = ours[valid]
    tv = theirs[valid]
    abs_diff = np.abs(ov - tv)
    max_abs = float(np.max(abs_diff))

    # Check tolerance
    np.testing.assert_allclose(
        ov, tv, rtol=rtol, atol=atol,
        err_msg=f"{name}: value mismatch"
    )
    return max_abs


# ============================================================
# Indicator definitions
# ============================================================

# (name, min_n, rtol, atol, needs_ohlcv, call_fn)
# call_fn: (lib, o, h, l, c, v) -> result  OR  (lib, price) -> result

def _ohlcv_indicator(name, min_n, rtol, atol, fn):
    return (name, min_n, rtol, atol, True, fn)

def _price_indicator(name, min_n, rtol, atol, fn):
    return (name, min_n, rtol, atol, False, fn)

# Exact tolerance (element-wise, no accumulation)
E_RTOL, E_ATOL = 1e-14, 0
# Tight tolerance (EMA/Wilder serial chains)
T_RTOL, T_ATOL = 1e-10, 1e-12
# Sliding tolerance (sliding sum with FP drift)
S_RTOL, S_ATOL = 1e-8, 1e-10

INDICATORS = [
    # ---- Overlap ----
    _price_indicator('SMA_20',   21, S_RTOL, S_ATOL, lambda lib, p: lib.SMA(p, 20)),
    _price_indicator('EMA_20',   21, T_RTOL, T_ATOL, lambda lib, p: lib.EMA(p, 20)),
    _price_indicator('WMA_20',   21, T_RTOL, T_ATOL, lambda lib, p: lib.WMA(p, 20)),
    _price_indicator('DEMA_20',  41, T_RTOL, T_ATOL, lambda lib, p: lib.DEMA(p, 20)),
    _price_indicator('TEMA_20',  61, T_RTOL, T_ATOL, lambda lib, p: lib.TEMA(p, 20)),
    _price_indicator('TRIMA_20', 21, T_RTOL, T_ATOL, lambda lib, p: lib.TRIMA(p, 20)),
    _price_indicator('KAMA_30',  31, T_RTOL, T_ATOL, lambda lib, p: lib.KAMA(p, 30)),
    _price_indicator('T3_5',     31, T_RTOL, T_ATOL, lambda lib, p: lib.T3(p, 5, 0.7)),
    _price_indicator('BBANDS_20', 21, S_RTOL, S_ATOL, lambda lib, p: lib.BBANDS(p, 20)),
    _price_indicator('MIDPOINT_14', 15, E_RTOL, E_ATOL, lambda lib, p: lib.MIDPOINT(p, 14)),
    _ohlcv_indicator('SAR', 3, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.SAR(h, l, 0.02, 0.2)),
    _ohlcv_indicator('SAREXT', 3, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.SAREXT(h, l)),
    _ohlcv_indicator('MIDPRICE_14', 15, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.MIDPRICE(h, l, 14)),
    _price_indicator('HT_TRENDLINE', 64, T_RTOL, T_ATOL,
                     lambda lib, p: lib.HT_TRENDLINE(p)),

    # ---- Momentum ----
    _price_indicator('RSI_14',   15, T_RTOL, T_ATOL, lambda lib, p: lib.RSI(p, 14)),
    _price_indicator('MACD',     34, T_RTOL, T_ATOL,
                     lambda lib, p: lib.MACD(p, 12, 26, 9)),
    _price_indicator('MACDFIX_9', 34, T_RTOL, T_ATOL,
                     lambda lib, p: lib.MACDFIX(p, 9)),
    _price_indicator('MACDEXT_EMA', 34, T_RTOL, T_ATOL,
                     lambda lib, p: lib.MACDEXT(p, 12, 1, 26, 1, 9, 1)),
    _ohlcv_indicator('STOCH', 10, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.STOCH(h, l, c, 5, 3, 0, 3, 0)),
    _ohlcv_indicator('STOCHF', 10, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.STOCHF(h, l, c, 5, 3, 0)),
    _ohlcv_indicator('ADX_14', 28, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.ADX(h, l, c, 14)),
    _ohlcv_indicator('ADXR_14', 52, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.ADXR(h, l, c, 14)),
    _ohlcv_indicator('CCI_14', 15, S_RTOL, S_ATOL,
                     lambda lib, o, h, l, c, v: lib.CCI(h, l, c, 14)),
    _price_indicator('CMO_14',   15, T_RTOL, T_ATOL, lambda lib, p: lib.CMO(p, 14)),
    _price_indicator('MOM_10',   11, E_RTOL, E_ATOL, lambda lib, p: lib.MOM(p, 10)),
    _price_indicator('ROC_10',   11, T_RTOL, T_ATOL, lambda lib, p: lib.ROC(p, 10)),
    _price_indicator('ROCP_10',  11, E_RTOL, E_ATOL, lambda lib, p: lib.ROCP(p, 10)),
    _price_indicator('ROCR_10',  11, E_RTOL, E_ATOL, lambda lib, p: lib.ROCR(p, 10)),
    _price_indicator('ROCR100_10', 11, E_RTOL, E_ATOL, lambda lib, p: lib.ROCR100(p, 10)),
    _ohlcv_indicator('WILLR_14', 15, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.WILLR(h, l, c, 14)),
    _price_indicator('APO', 27, S_RTOL, S_ATOL,
                     lambda lib, p: lib.APO(p, 12, 26, 0)),
    _price_indicator('PPO', 27, S_RTOL, S_ATOL,
                     lambda lib, p: lib.PPO(p, 12, 26, 0)),
    _ohlcv_indicator('BOP', 2, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.BOP(o, h, l, c)),
    _ohlcv_indicator('AROON_14', 15, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.AROON(h, l, 14)),
    _ohlcv_indicator('AROONOSC_14', 15, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.AROONOSC(h, l, 14)),
    _ohlcv_indicator('MFI_14', 15, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.MFI(h, l, c, v, 14)),
    _price_indicator('TRIX_15',  46, T_RTOL, T_ATOL, lambda lib, p: lib.TRIX(p, 15)),
    _ohlcv_indicator('ULTOSC', 29, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.ULTOSC(h, l, c, 7, 14, 28)),
    _ohlcv_indicator('DX_14', 28, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.DX(h, l, c, 14)),
    _ohlcv_indicator('PLUS_DI_14', 15, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.PLUS_DI(h, l, c, 14)),
    _ohlcv_indicator('MINUS_DI_14', 15, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.MINUS_DI(h, l, c, 14)),
    _ohlcv_indicator('PLUS_DM_14', 15, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.PLUS_DM(h, l, 14)),
    _ohlcv_indicator('MINUS_DM_14', 15, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.MINUS_DM(h, l, 14)),

    # ---- Volatility ----
    _ohlcv_indicator('ATR_14', 15, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.ATR(h, l, c, 14)),
    _ohlcv_indicator('NATR_14', 15, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.NATR(h, l, c, 14)),
    _ohlcv_indicator('TRANGE', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.TRANGE(h, l, c)),

    # ---- Volume ----
    _ohlcv_indicator('AD', 2, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.AD(h, l, c, v)),
    _ohlcv_indicator('ADOSC', 11, T_RTOL, T_ATOL,
                     lambda lib, o, h, l, c, v: lib.ADOSC(h, l, c, v, 3, 10)),
    _ohlcv_indicator('OBV', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.OBV(c, v)),

    # ---- Price Transform ----
    _ohlcv_indicator('AVGPRICE', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.AVGPRICE(o, h, l, c)),
    _ohlcv_indicator('MEDPRICE', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.MEDPRICE(h, l)),
    _ohlcv_indicator('TYPPRICE', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.TYPPRICE(h, l, c)),
    _ohlcv_indicator('WCLPRICE', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: lib.WCLPRICE(h, l, c)),

    # ---- Statistics ----
    _price_indicator('STDDEV_20', 21, S_RTOL, S_ATOL,
                     lambda lib, p: lib.STDDEV(p, 20)),
    _price_indicator('VAR_20', 21, S_RTOL, S_ATOL,
                     lambda lib, p: lib.VAR(p, 20)),
    _price_indicator('LINEARREG_14', 15, S_RTOL, S_ATOL,
                     lambda lib, p: lib.LINEARREG(p, 14)),
    _price_indicator('LINEARREG_SLOPE_14', 15, S_RTOL, S_ATOL,
                     lambda lib, p: lib.LINEARREG_SLOPE(p, 14)),
    _price_indicator('TSF_14', 15, S_RTOL, S_ATOL,
                     lambda lib, p: lib.TSF(p, 14)),
    _price_indicator('CORREL_30', 31, S_RTOL, S_ATOL,
                     lambda lib, p: lib.CORREL(p, np.roll(p, 5), 30)),
    _price_indicator('BETA_5', 6, S_RTOL, S_ATOL,
                     lambda lib, p: lib.BETA(p, np.roll(p, 3), 5)),

    # ---- Math Operators ----
    _price_indicator('MAX_30', 31, E_RTOL, E_ATOL, lambda lib, p: lib.MAX(p, 30)),
    _price_indicator('MIN_30', 31, E_RTOL, E_ATOL, lambda lib, p: lib.MIN(p, 30)),
    _price_indicator('SUM_30', 31, S_RTOL, S_ATOL, lambda lib, p: lib.SUM(p, 30)),
    _price_indicator('SQRT',    2, E_RTOL, E_ATOL, lambda lib, p: lib.SQRT(p)),

    # ---- Cycle ----
    _price_indicator('HT_DCPERIOD', 64, T_RTOL, T_ATOL,
                     lambda lib, p: lib.HT_DCPERIOD(p)),
    _price_indicator('HT_SINE', 64, T_RTOL, T_ATOL,
                     lambda lib, p: lib.HT_SINE(p)),

    # ---- Pattern (sample) ----
    _ohlcv_indicator('CDLDOJI', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: np.asarray(lib.CDLDOJI(o, h, l, c), dtype=np.float64)),
    _ohlcv_indicator('CDLHAMMER', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: np.asarray(lib.CDLHAMMER(o, h, l, c), dtype=np.float64)),
    _ohlcv_indicator('CDLENGULFING', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: np.asarray(lib.CDLENGULFING(o, h, l, c), dtype=np.float64)),
    _ohlcv_indicator('CDL3BLACKCROWS', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: np.asarray(lib.CDL3BLACKCROWS(o, h, l, c), dtype=np.float64)),
    _ohlcv_indicator('CDLMORNINGSTAR', 2, E_RTOL, E_ATOL,
                     lambda lib, o, h, l, c, v: np.asarray(lib.CDLMORNINGSTAR(o, h, l, c), dtype=np.float64)),
]

# ============================================================
# Tests
# ============================================================

skipif_no_data = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason=f"Real data files not available"
)


# ---- Crypto OHLCV tests ----

_crypto_params = []
if DATA_AVAILABLE:
    for symbol in _crypto_sample_symbols:
        for ind in INDICATORS:
            name, min_n, rtol, atol, needs_ohlcv, fn = ind
            _crypto_params.append(
                pytest.param(symbol, name, min_n, rtol, atol, needs_ohlcv, fn,
                            id=f"crypto_{symbol}_{name}")
            )


@skipif_no_data
@pytest.mark.parametrize("symbol,name,min_n,rtol,atol,needs_ohlcv,fn", _crypto_params)
def test_crypto_alignment(symbol, name, min_n, rtol, atol, needs_ohlcv, fn):
    o, h, l, c, v = _get_crypto_ohlcv(symbol)
    n = len(c)
    if n < min_n:
        pytest.skip(f"Insufficient data: {n} < {min_n}")

    if needs_ohlcv:
        ours = fn(rs, o, h, l, c, v)
        theirs = fn(c_talib, o, h, l, c, v)
    else:
        ours = fn(rs, c)
        theirs = fn(c_talib, c)

    _compare(ours, theirs, f"{name}@{symbol}", rtol, atol)


# ---- A-share price-only tests ----

_ashare_params = []
if DATA_AVAILABLE:
    price_only_indicators = [(ind) for ind in INDICATORS if not ind[4]]  # needs_ohlcv=False
    for symbol in _ashare_sample_symbols:
        for ind in price_only_indicators:
            name, min_n, rtol, atol, needs_ohlcv, fn = ind
            _ashare_params.append(
                pytest.param(symbol, name, min_n, rtol, atol, fn,
                            id=f"ashare_{symbol}_{name}")
            )


@skipif_no_data
@pytest.mark.parametrize("symbol,name,min_n,rtol,atol,fn", _ashare_params)
def test_ashare_alignment(symbol, name, min_n, rtol, atol, fn):
    price = _get_ashare_price(symbol)
    n = len(price)
    if n < min_n:
        pytest.skip(f"Insufficient data: {n} < {min_n}")

    # Skip if price has zeros (can cause div-by-zero in ROCP etc.)
    if np.any(price <= 0):
        pytest.skip(f"Non-positive prices in {symbol}")

    ours = fn(rs, price)
    theirs = fn(c_talib, price)

    _compare(ours, theirs, f"{name}@{symbol}", rtol, atol)
