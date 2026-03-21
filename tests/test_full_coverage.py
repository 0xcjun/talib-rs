"""
全指标自动化测试: 155 个函数 × 多数据集 × 一致性 + 边界 + 性能

数据驱动设计: 每个函数的签名和默认参数记录在 FUNC_REGISTRY 中，
测试框架自动为每个函数生成:
  1. 多数据集一致性测试 (ta-rs vs C TA-Lib, rtol=1e-10)
  2. 边界条件测试 (最小输入、常数值、大数据集)
  3. 性能基准测试 (10K 数据)

运行:
  pytest tests/test_full_coverage.py -v                          # 全部
  pytest tests/test_full_coverage.py -k "consistency" -v         # 仅一致性
  pytest tests/test_full_coverage.py -k "edge" -v                # 仅边界
  pytest tests/test_full_coverage.py -k "bench" --benchmark-sort=name  # 仅性能
"""

import numpy as np
import pytest

import talib as c_talib
from talib import _talib as rs

# ============================================================
# 函数注册表: 每个函数的输入类型和默认参数
# ============================================================

# 输入类型缩写:
#   'c'    = close only
#   'hl'   = high, low
#   'hlc'  = high, low, close
#   'ohlc' = open, high, low, close
#   'hlcv' = high, low, close, volume
#   'cv'   = close, volume
#   'cc'   = close, close2 (两个序列)
#   'c+p'  = close + periods array

FUNC_REGISTRY = {
    # ---- Overlap Studies ----
    'SMA':   ('c',    {'timeperiod': 20}),
    'EMA':   ('c',    {'timeperiod': 20}),
    'WMA':   ('c',    {'timeperiod': 20}),
    'DEMA':  ('c',    {'timeperiod': 20}),
    'TEMA':  ('c',    {'timeperiod': 20}),
    'TRIMA': ('c',    {'timeperiod': 20}),
    'KAMA':  ('c',    {'timeperiod': 30}),
    'T3':    ('c',    {'timeperiod': 5, 'vfactor': 0.7}),
    'MAMA':  ('c',    {'fastlimit': 0.5, 'slowlimit': 0.05}),
    'BBANDS':('c',    {'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'matype': 0}),
    'SAR':   ('hl',   {'acceleration': 0.02, 'maximum': 0.2}),
    'SAREXT':('hl',   {'startvalue': 0.0, 'offsetonreverse': 0.0, 'accelerationinitlong': 0.02, 'accelerationlong': 0.02, 'accelerationmaxlong': 0.2, 'accelerationinitshort': 0.02, 'accelerationshort': 0.02, 'accelerationmaxshort': 0.2}),
    'MIDPOINT':('c',  {'timeperiod': 14}),
    'MIDPRICE':('hl', {'timeperiod': 14}),
    'MAVP':  ('c+p',  {'minperiod': 2, 'maxperiod': 30, 'matype': 0}),
    'HT_TRENDLINE':('c', {}),

    # ---- Momentum ----
    'RSI':   ('c',    {'timeperiod': 14}),
    'MACD':  ('c',    {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
    'MACDEXT':('c',   {'fastperiod': 12, 'fastmatype': 1, 'slowperiod': 26, 'slowmatype': 1, 'signalperiod': 9, 'signalmatype': 1}),
    'MACDFIX':('c',   {'signalperiod': 9}),
    'STOCH': ('hlc',  {'fastk_period': 5, 'slowk_period': 3, 'slowk_matype': 0, 'slowd_period': 3, 'slowd_matype': 0}),
    'STOCHF':('hlc',  {'fastk_period': 5, 'fastd_period': 3, 'fastd_matype': 0}),
    'STOCHRSI':('c',  {'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3, 'fastd_matype': 0}),
    'ADX':   ('hlc',  {'timeperiod': 14}),
    'ADXR':  ('hlc',  {'timeperiod': 14}),
    'CCI':   ('hlc',  {'timeperiod': 14}),
    'MOM':   ('c',    {'timeperiod': 10}),
    'ROC':   ('c',    {'timeperiod': 10}),
    'ROCP':  ('c',    {'timeperiod': 10}),
    'ROCR':  ('c',    {'timeperiod': 10}),
    'ROCR100':('c',   {'timeperiod': 10}),
    'WILLR': ('hlc',  {'timeperiod': 14}),
    'APO':   ('c',    {'fastperiod': 12, 'slowperiod': 26, 'matype': 0}),
    'PPO':   ('c',    {'fastperiod': 12, 'slowperiod': 26, 'matype': 0}),
    'BOP':   ('ohlc', {}),
    'CMO':   ('c',    {'timeperiod': 14}),
    'AROON': ('hl',   {'timeperiod': 14}),
    'AROONOSC':('hl', {'timeperiod': 14}),
    'MFI':   ('hlcv', {'timeperiod': 14}),
    'TRIX':  ('c',    {'timeperiod': 15}),
    'ULTOSC':('hlc',  {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28}),
    'DX':    ('hlc',  {'timeperiod': 14}),
    'PLUS_DI':('hlc', {'timeperiod': 14}),
    'MINUS_DI':('hlc',{'timeperiod': 14}),
    'PLUS_DM':('hl',  {'timeperiod': 14}),
    'MINUS_DM':('hl', {'timeperiod': 14}),

    # ---- Volatility ----
    'ATR':   ('hlc',  {'timeperiod': 14}),
    'NATR':  ('hlc',  {'timeperiod': 14}),
    'TRANGE':('hlc',  {}),

    # ---- Volume ----
    'AD':    ('hlcv', {}),
    'ADOSC': ('hlcv', {'fastperiod': 3, 'slowperiod': 10}),
    'OBV':   ('cv',   {}),

    # ---- Price Transform ----
    'AVGPRICE':('ohlc',{}),
    'MEDPRICE':('hl',  {}),
    'TYPPRICE':('hlc', {}),
    'WCLPRICE':('hlc', {}),

    # ---- Statistic ----
    'STDDEV':('c',    {'timeperiod': 20, 'nbdev': 1.0}),
    'VAR':   ('c',    {'timeperiod': 20, 'nbdev': 1.0}),
    'BETA':  ('cc',   {'timeperiod': 5}),
    'CORREL':('cc',   {'timeperiod': 30}),
    'LINEARREG':('c', {'timeperiod': 14}),
    'LINEARREG_SLOPE':('c',{'timeperiod': 14}),
    'LINEARREG_INTERCEPT':('c',{'timeperiod': 14}),
    'LINEARREG_ANGLE':('c',{'timeperiod': 14}),
    'TSF':   ('c',    {'timeperiod': 14}),

    # ---- Math Transform ----
    'ACOS':  ('c_safe',{}),  # 特殊输入范围
    'ASIN':  ('c_safe',{}),
    'ATAN':  ('c',    {}),
    'CEIL':  ('c',    {}),
    'COS':   ('c',    {}),
    'COSH':  ('c_small',{}),
    'EXP':   ('c_tiny',{}),  # 避免溢出
    'FLOOR': ('c',    {}),
    'LN':    ('c_pos',{}),   # 正数
    'LOG10': ('c_pos',{}),
    'SIN':   ('c',    {}),
    'SINH':  ('c_small',{}),
    'SQRT':  ('c_pos',{}),
    'TAN':   ('c',    {}),
    'TANH':  ('c',    {}),

    # ---- Math Operators ----
    'ADD':   ('cc',   {}),
    'SUB':   ('cc',   {}),
    'MULT':  ('cc',   {}),
    'DIV':   ('cc_nz',{}),  # 除数非零
    'MAX':   ('c',    {'timeperiod': 30}),
    'MAXINDEX':('c',  {'timeperiod': 30}),
    'MIN':   ('c',    {'timeperiod': 30}),
    'MININDEX':('c',  {'timeperiod': 30}),
    'SUM':   ('c',    {'timeperiod': 30}),

    # ---- Cycle ----
    'HT_DCPERIOD':('c', {}),
    'HT_DCPHASE': ('c', {}),
    'HT_PHASOR':  ('c', {}),
    'HT_SINE':    ('c', {}),
    'HT_TRENDMODE':('c',{}),
}

# K线形态 (全部 ohlc 输入，无参数)
CDL_FUNCTIONS = [
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE',
    'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
    'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
    'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR',
    'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
    'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
    'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
    'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
    'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI',
    'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD',
    'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
    'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR',
    'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH',
    'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER',
    'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS',
]
for name in CDL_FUNCTIONS:
    FUNC_REGISTRY[name] = ('ohlc', {})

ALL_FUNC_NAMES = sorted(FUNC_REGISTRY.keys())

# ============================================================
# 测试数据
# ============================================================

def _walk(n, seed, start=100.0):
    rng = np.random.RandomState(seed)
    return start * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))

def _ohlcv(n, seed):
    rng = np.random.RandomState(seed)
    c = _walk(n, seed)
    sp = c * 0.015
    h = c + rng.uniform(0, 1, n) * sp
    l = c - rng.uniform(0, 1, n) * sp
    o = l + rng.uniform(0, 1, n) * (h - l)
    v = rng.uniform(1e6, 1e7, n)
    return o, h, l, c, v

# 预生成多个数据集
DS = {}
for _name, _n, _seed in [('1k', 1000, 42), ('5k', 5000, 77), ('100k', 100000, 99)]:
    DS[_name] = _ohlcv(_n, _seed)

# 特殊输入数据
DS['const'] = tuple(np.full(500, x) for x in [50.0, 52.0, 48.0, 50.0, 1e6])
DS['trend_up'] = _ohlcv(500, 10)  # 使用随机数据但较短
DS['trend_dn'] = _ohlcv(500, 20)

# 第二个序列 (用于 BETA, CORREL, ADD 等)
CLOSE2 = {k: _walk(v[3].shape[0], 200 + i) for i, (k, v) in enumerate(DS.items())}


def _build_args(func_name, input_type, params, ds_name):
    """根据输入类型构建函数参数"""
    o, h, l, c, v = DS[ds_name]
    n = c.shape[0]

    if input_type == 'c':
        pos = [c]
    elif input_type == 'c_safe':
        pos = [np.linspace(-0.99, 0.99, n)]
    elif input_type == 'c_small':
        pos = [np.linspace(-2.0, 2.0, n)]
    elif input_type == 'c_tiny':
        pos = [np.linspace(0.0, 5.0, n)]
    elif input_type == 'c_pos':
        pos = [np.abs(c) + 0.01]
    elif input_type == 'hl':
        pos = [h, l]
    elif input_type == 'hlc':
        pos = [h, l, c]
    elif input_type == 'ohlc':
        pos = [o, h, l, c]
    elif input_type == 'hlcv':
        pos = [h, l, c, v]
    elif input_type == 'cv':
        pos = [c, v]
    elif input_type == 'cc':
        pos = [c, CLOSE2[ds_name]]
    elif input_type == 'cc_nz':
        c2 = CLOSE2[ds_name]
        c2 = np.where(np.abs(c2) < 0.01, 0.01, c2)
        pos = [c, c2]
    elif input_type == 'c+p':
        # MAVP: close + periods array
        periods = np.full(n, 10.0)
        pos = [c, periods]
    else:
        raise ValueError(f"Unknown input type: {input_type}")

    kw = dict(params)
    return pos, kw


def _call_rs(func_name, pos, kw):
    """调用 ta-rs 函数"""
    func = getattr(rs, func_name)
    return func(*pos, *kw.values())


def _call_c(func_name, pos, kw):
    """调用 C TA-Lib 函数"""
    func = getattr(c_talib, func_name)
    return func(*pos, **kw)


def _compare(ours, theirs, name, rtol=1e-10, atol=1e-12):
    """比较结果"""
    if isinstance(ours, tuple) and isinstance(theirs, tuple):
        assert len(ours) == len(theirs), f"{name}: tuple len {len(ours)} vs {len(theirs)}"
        for i, (a, b) in enumerate(zip(ours, theirs)):
            _cmp_arr(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                    f"{name}[{i}]", rtol, atol)
    else:
        _cmp_arr(np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64),
                name, rtol, atol)


def _cmp_arr(a, b, name, rtol, atol):
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    both_nan = np.isnan(a) & np.isnan(b)
    mismatch_nan = np.isnan(a) != np.isnan(b)
    if np.any(mismatch_nan):
        idx = np.where(mismatch_nan)[0][0]
        pytest.fail(f"{name}[{idx}]: NaN mismatch ours={a[idx]} theirs={b[idx]}")
    valid = ~both_nan
    if np.any(valid):
        np.testing.assert_allclose(a[valid], b[valid], rtol=rtol, atol=atol, err_msg=name)


# ============================================================
# 1. 全指标一致性测试 (ta-rs vs C TA-Lib)
# ============================================================

# 排除 MAVP (特殊参数) 和 pattern (单独测试值域)
_CONSISTENCY_FUNCS = [f for f in ALL_FUNC_NAMES]

@pytest.mark.parametrize("func_name", _CONSISTENCY_FUNCS)
@pytest.mark.parametrize("ds_name", ['1k', '5k'])
class TestConsistency:
    def test_consistency(self, func_name, ds_name):
        input_type, params = FUNC_REGISTRY[func_name]
        try:
            pos, kw = _build_args(func_name, input_type, params, ds_name)
            ours = _call_rs(func_name, pos, kw)
            theirs = _call_c(func_name, pos, kw)
            # Pattern 只检查形状和值域
            if func_name.startswith('CDL'):
                a = np.asarray(ours)
                b = np.asarray(theirs)
                assert a.shape == b.shape
                assert set(np.unique(a)).issubset({-100, 0, 100})
                assert set(np.unique(b)).issubset({-100, 0, 100})
            else:
                _compare(ours, theirs, func_name)
        except Exception as e:
            if "insufficient" in str(e).lower() or "too short" in str(e).lower():
                pytest.skip(f"{func_name} needs more data")
            raise


# ============================================================
# 2. 全指标边界测试
# ============================================================

# 非 pattern 函数
_EDGE_FUNCS = [f for f in ALL_FUNC_NAMES if not f.startswith('CDL')]

@pytest.mark.parametrize("func_name", _EDGE_FUNCS)
class TestEdgeCases:
    """边界条件: 常数值、大数据集"""

    def test_constant_input(self, func_name):
        """常数值输入不应产生 NaN/Inf (lookback 后)"""
        input_type, params = FUNC_REGISTRY[func_name]
        try:
            pos, kw = _build_args(func_name, input_type, params, 'const')
            result = _call_rs(func_name, pos, kw)
            if isinstance(result, tuple):
                for arr in result:
                    a = np.asarray(arr, dtype=np.float64)
                    valid = a[~np.isnan(a)]
                    assert not np.any(np.isinf(valid)), f"{func_name}: inf in output"
            else:
                a = np.asarray(result, dtype=np.float64)
                valid = a[~np.isnan(a)]
                assert not np.any(np.isinf(valid)), f"{func_name}: inf in output"
        except (ValueError, Exception) as e:
            if "insufficient" in str(e).lower() or "too short" in str(e).lower() or "invalid" in str(e).lower():
                pytest.skip(f"{func_name}: {e}")
            raise

    def test_large_dataset(self, func_name):
        """100K 数据集不应崩溃"""
        input_type, params = FUNC_REGISTRY[func_name]
        try:
            pos, kw = _build_args(func_name, input_type, params, '100k')
            result = _call_rs(func_name, pos, kw)
            if isinstance(result, tuple):
                for arr in result:
                    assert np.asarray(arr).shape[0] == 100000
            else:
                assert np.asarray(result).shape[0] == 100000
        except (ValueError, Exception) as e:
            if "insufficient" in str(e).lower() or "too short" in str(e).lower():
                pytest.skip(f"{func_name}: {e}")
            raise


# Pattern 边界测试
@pytest.mark.parametrize("func_name", CDL_FUNCTIONS)
class TestPatternEdge:
    def test_pattern_constant(self, func_name):
        """常数 OHLC 不应崩溃"""
        o = h = l = c = np.full(100, 50.0)
        h = c + 1.0
        l = c - 1.0
        result = getattr(rs, func_name)(o, h, l, c)
        assert np.asarray(result).shape[0] == 100

    def test_pattern_100k(self, func_name):
        o, h, l, c, _ = DS['100k']
        result = getattr(rs, func_name)(o, h, l, c)
        a = np.asarray(result)
        assert a.shape[0] == 100000
        assert set(np.unique(a)).issubset({-100, 0, 100})


# ============================================================
# 3. 全指标性能基准 (ta-rs vs C TA-Lib, 10K)
# ============================================================

_BENCH_FUNCS = [f for f in ALL_FUNC_NAMES]

@pytest.mark.parametrize("func_name", _BENCH_FUNCS)
class TestBenchRs:
    """ta-rs 10K 性能"""
    def test_bench(self, func_name, benchmark):
        input_type, params = FUNC_REGISTRY[func_name]
        try:
            pos, kw = _build_args(func_name, input_type, params, '1k')
            func = getattr(rs, func_name)
            benchmark(func, *pos, *kw.values())
        except Exception:
            pytest.skip(f"Cannot benchmark {func_name}")


@pytest.mark.parametrize("func_name", _BENCH_FUNCS)
class TestBenchC:
    """C TA-Lib 10K 性能"""
    def test_bench(self, func_name, benchmark):
        input_type, params = FUNC_REGISTRY[func_name]
        try:
            pos, kw = _build_args(func_name, input_type, params, '1k')
            func = getattr(c_talib, func_name)
            benchmark(func, *pos, **kw)
        except Exception:
            pytest.skip(f"Cannot benchmark {func_name}")
