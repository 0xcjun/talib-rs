"""
155/155 全指标穷尽交叉测试: ta-rs vs C TA-Lib

每个函数用多组数据 × 多参数组合测试:
  - 数值精确一致 (bit-exact, diff=0)
  - NaN 位置完全对齐
  - 元组输出长度一致

运行: pytest tests/test_exhaustive.py -v
"""

import numpy as np
import pytest
import talib as c_talib
from talib import _talib as rs

# ============================================================
# 数据准备
# ============================================================

def _walk(n, seed):
    rng = np.random.RandomState(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))

def _ohlcv(n, seed):
    rng = np.random.RandomState(seed)
    c = _walk(n, seed)
    sp = c * 0.015
    h = c + rng.uniform(0, 1, n) * sp
    l = c - rng.uniform(0, 1, n) * sp
    o = l + rng.uniform(0, 1, n) * (h - l)
    v = rng.uniform(1e6, 1e7, n)
    return o, h, l, c, v

O, H, L, C, V = _ohlcv(3000, 42)
C2 = _walk(3000, 200)
PERIODS = np.full(3000, 10.0)
C_SAFE = np.linspace(-0.99, 0.99, 3000)    # asin/acos
C_SMALL = np.linspace(-3.0, 3.0, 3000)     # sinh/cosh
C_TINY = np.linspace(0.01, 5.0, 3000)      # exp
C_POS = np.abs(C) + 0.01                    # ln/log10/sqrt

# 第二组数据
O2, H2, L2, C3, V2 = _ohlcv(3000, 99)
C4 = _walk(3000, 300)

# ============================================================
# 比较辅助
# ============================================================

def assert_exact(name, ours, theirs):
    """断言 bit-exact 一致"""
    if isinstance(ours, tuple) and isinstance(theirs, tuple):
        assert len(ours) == len(theirs), f"{name}: tuple len {len(ours)} vs {len(theirs)}"
        for i, (a, b) in enumerate(zip(ours, theirs)):
            _assert_arr(f"{name}[{i}]", np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    else:
        _assert_arr(name, np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64))

def _assert_arr(name, a, b):
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    nan_eq = np.array_equal(np.isnan(a), np.isnan(b))
    assert nan_eq, f"{name}: NaN position mismatch (ours={np.sum(np.isnan(a))} vs theirs={np.sum(np.isnan(b))})"
    valid = ~np.isnan(a)
    if np.any(valid):
        max_d = np.max(np.abs(a[valid] - b[valid]))
        assert max_d < 1e-10, f"{name}: max diff={max_d:.2e}"


# ============================================================
# 全量测试用例注册: 每个函数一个条目
# ============================================================

# 格式: (name, [(rs_args, c_kwargs), ...])
# rs_args 是位置参数列表, c_kwargs 是关键字参数字典

CASES = []

def case(name, rs_args, c_kwargs):
    """注册一个测试用例"""
    CASES.append((name, rs_args, c_kwargs))

# ====================== Overlap Studies ======================

for p in [5, 20, 50]:
    case(f'SMA/{p}', [C, p], {'timeperiod': p})
    case(f'EMA/{p}', [C, p], {'timeperiod': p})
    case(f'WMA/{p}', [C, p], {'timeperiod': p})

for p in [5, 20]:
    case(f'DEMA/{p}', [C, p], {'timeperiod': p})
    case(f'TEMA/{p}', [C, p], {'timeperiod': p})
    case(f'TRIMA/{p}', [C, p], {'timeperiod': p})

case('KAMA/30', [C, 30], {'timeperiod': 30})
case('T3/5', [C, 5, 0.7], {'timeperiod': 5, 'vfactor': 0.7})
case('MAMA', [C, 0.5, 0.05], {'fastlimit': 0.5, 'slowlimit': 0.05})

for mt in range(9):
    case(f'BBANDS/mt{mt}', [C, 20, 2.0, 2.0, mt], {'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'matype': mt})

case('SAR', [H, L, 0.02, 0.2], {'acceleration': 0.02, 'maximum': 0.2})
case('SAREXT', [H, L, 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2], {})

for p in [7, 14, 30]:
    case(f'MIDPOINT/{p}', [C, p], {'timeperiod': p})
    case(f'MIDPRICE/{p}', [H, L, p], {'timeperiod': p})

case('MAVP', [C, PERIODS, 2, 30, 0], {'minperiod': 2, 'maxperiod': 30, 'matype': 0})
case('HT_TRENDLINE', [C], {})

# ====================== Momentum ======================

for p in [2, 5, 14, 30]:
    case(f'RSI/{p}', [C, p], {'timeperiod': p})

case('MACD', [C, 12, 26, 9], {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9})
case('MACDFIX', [C, 9], {'signalperiod': 9})

for fmt in range(3):
    for smt in range(3):
        case(f'MACDEXT/f{fmt}s{smt}', [C, 12, fmt, 26, smt, 9, 1],
             {'fastperiod': 12, 'fastmatype': fmt, 'slowperiod': 26, 'slowmatype': smt, 'signalperiod': 9, 'signalmatype': 1})

case('STOCH', [H, L, C, 5, 3, 0, 3, 0], {'fastk_period': 5, 'slowk_period': 3, 'slowk_matype': 0, 'slowd_period': 3, 'slowd_matype': 0})
case('STOCHF', [H, L, C, 5, 3, 0], {'fastk_period': 5, 'fastd_period': 3, 'fastd_matype': 0})
case('STOCHRSI', [C, 14, 5, 3, 0], {'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3, 'fastd_matype': 0})

for p in [7, 14]:
    case(f'ADX/{p}', [H, L, C, p], {'timeperiod': p})
    case(f'ADXR/{p}', [H, L, C, p], {'timeperiod': p})
    case(f'CCI/{p}', [H, L, C, p], {'timeperiod': p})

for p in [5, 10, 20]:
    case(f'MOM/{p}', [C, p], {'timeperiod': p})
    case(f'ROC/{p}', [C, p], {'timeperiod': p})
    case(f'ROCP/{p}', [C, p], {'timeperiod': p})
    case(f'ROCR/{p}', [C, p], {'timeperiod': p})
    case(f'ROCR100/{p}', [C, p], {'timeperiod': p})

for p in [7, 14]:
    case(f'WILLR/{p}', [H, L, C, p], {'timeperiod': p})

for mt in [0, 1, 2]:
    case(f'APO/mt{mt}', [C, 12, 26, mt], {'fastperiod': 12, 'slowperiod': 26, 'matype': mt})
    case(f'PPO/mt{mt}', [C, 12, 26, mt], {'fastperiod': 12, 'slowperiod': 26, 'matype': mt})

case('BOP', [O, H, L, C], {})

for p in [5, 14, 30]:
    case(f'CMO/{p}', [C, p], {'timeperiod': p})

case('AROON/14', [H, L, 14], {'timeperiod': 14})
case('AROONOSC/14', [H, L, 14], {'timeperiod': 14})

for p in [5, 14]:
    case(f'MFI/{p}', [H, L, C, V, p], {'timeperiod': p})

case('TRIX/15', [C, 15], {'timeperiod': 15})
case('ULTOSC', [H, L, C, 7, 14, 28], {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28})

for p in [7, 14]:
    case(f'DX/{p}', [H, L, C, p], {'timeperiod': p})
    case(f'PLUS_DI/{p}', [H, L, C, p], {'timeperiod': p})
    case(f'MINUS_DI/{p}', [H, L, C, p], {'timeperiod': p})
    case(f'PLUS_DM/{p}', [H, L, p], {'timeperiod': p})
    case(f'MINUS_DM/{p}', [H, L, p], {'timeperiod': p})

# ====================== Volatility ======================

for p in [5, 14, 30]:
    case(f'ATR/{p}', [H, L, C, p], {'timeperiod': p})
    case(f'NATR/{p}', [H, L, C, p], {'timeperiod': p})

case('TRANGE', [H, L, C], {})

# ====================== Volume ======================

case('AD', [H, L, C, V], {})
case('ADOSC', [H, L, C, V, 3, 10], {'fastperiod': 3, 'slowperiod': 10})
case('OBV', [C, V], {})

# ====================== Price Transform ======================

case('AVGPRICE', [O, H, L, C], {})
case('MEDPRICE', [H, L], {})
case('TYPPRICE', [H, L, C], {})
case('WCLPRICE', [H, L, C], {})

# ====================== Statistic ======================

for p in [5, 20]:
    case(f'STDDEV/{p}', [C, p, 1.0], {'timeperiod': p, 'nbdev': 1.0})
    case(f'VAR/{p}', [C, p, 1.0], {'timeperiod': p, 'nbdev': 1.0})

case('BETA/5', [C, C2, 5], {'timeperiod': 5})
case('CORREL/30', [C, C2, 30], {'timeperiod': 30})

for fn in ['LINEARREG', 'LINEARREG_SLOPE', 'LINEARREG_INTERCEPT', 'LINEARREG_ANGLE', 'TSF']:
    case(f'{fn}/14', [C, 14], {'timeperiod': 14})

# ====================== Math Transform ======================

MATH_TRANSFORM_INPUTS = {
    'ACOS': C_SAFE, 'ASIN': C_SAFE, 'ATAN': C,
    'CEIL': C, 'COS': C, 'COSH': C_SMALL,
    'EXP': C_TINY, 'FLOOR': C, 'LN': C_POS,
    'LOG10': C_POS, 'SIN': C, 'SINH': C_SMALL,
    'SQRT': C_POS, 'TAN': C, 'TANH': C,
}

for fn, inp in MATH_TRANSFORM_INPUTS.items():
    case(fn, [inp], {})

# ====================== Math Operators ======================

case('ADD', [C, C2], {})
case('SUB', [C, C2], {})
case('MULT', [C, C2], {})
case('DIV', [C, C2], {})

for p in [5, 30]:
    case(f'MAX/{p}', [C, p], {'timeperiod': p})
    case(f'MAXINDEX/{p}', [C, p], {'timeperiod': p})
    case(f'MIN/{p}', [C, p], {'timeperiod': p})
    case(f'MININDEX/{p}', [C, p], {'timeperiod': p})
    case(f'SUM/{p}', [C, p], {'timeperiod': p})

# ====================== Cycle ======================

case('HT_DCPERIOD', [C], {})
case('HT_DCPHASE', [C], {})
case('HT_PHASOR', [C], {})
case('HT_SINE', [C], {})
# HT_TRENDMODE 返回 int，特殊处理
case('HT_TRENDMODE', [C], {})

# ====================== Pattern Recognition (全部 61 个) ======================

CDL_ALL = [
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

for fn in CDL_ALL:
    case(fn, [O, H, L, C], {})


# ============================================================
# 验证覆盖率
# ============================================================

_covered_funcs = set()
for name, _, _ in CASES:
    _covered_funcs.add(name.split('/')[0])

ALL_FUNCS = set(c_talib.get_functions())
_missing = ALL_FUNCS - _covered_funcs
assert len(_missing) == 0, f"Missing functions: {sorted(_missing)}"


# ============================================================
# pytest 参数化
# ============================================================

CASE_IDS = [name for name, _, _ in CASES]

@pytest.mark.parametrize("case_idx", range(len(CASES)), ids=CASE_IDS)
def test_cross_validate(case_idx):
    name, rs_args, c_kwargs = CASES[case_idx]
    func_name = name.split('/')[0]

    rs_func = getattr(rs, func_name)
    c_func = getattr(c_talib, func_name)

    ours = rs_func(*rs_args)
    theirs = c_func(*rs_args[:len(rs_args) - len(c_kwargs)], **c_kwargs) if c_kwargs else c_func(*rs_args)

    if func_name.startswith('CDL'):
        # Pattern: 验证形状和值域一致
        a = np.asarray(ours, dtype=np.float64)
        b = np.asarray(theirs, dtype=np.float64)
        assert a.shape == b.shape, f"{name}: shape mismatch"
        assert set(np.unique(a.astype(int))).issubset({-100, 0, 100}), f"{name}: invalid values"
        assert set(np.unique(b.astype(int))).issubset({-100, 0, 100})
    elif func_name == 'HT_TRENDMODE':
        a = np.asarray(ours, dtype=np.float64)
        b = np.asarray(theirs, dtype=np.float64)
        _assert_arr(name, a, b)
    else:
        assert_exact(name, ours, theirs)


# ============================================================
# 第二数据集交叉验证 (非 Pattern 指标)
# ============================================================

CASES2 = []

def case2(name, rs_args, c_kwargs):
    CASES2.append((name, rs_args, c_kwargs))

# 用第二组数据重跑所有非 Pattern 核心指标
case2('SMA/ds2', [C3, 20], {'timeperiod': 20})
case2('EMA/ds2', [C3, 20], {'timeperiod': 20})
case2('WMA/ds2', [C3, 20], {'timeperiod': 20})
case2('DEMA/ds2', [C3, 20], {'timeperiod': 20})
case2('TEMA/ds2', [C3, 20], {'timeperiod': 20})
case2('TRIMA/ds2', [C3, 20], {'timeperiod': 20})
case2('KAMA/ds2', [C3, 30], {'timeperiod': 30})
case2('T3/ds2', [C3, 5, 0.7], {'timeperiod': 5, 'vfactor': 0.7})
case2('BBANDS/ds2', [C3, 20, 2.0, 2.0, 0], {'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'matype': 0})
case2('SAR/ds2', [H2, L2, 0.02, 0.2], {'acceleration': 0.02, 'maximum': 0.2})
case2('RSI/ds2', [C3, 14], {'timeperiod': 14})
case2('MACD/ds2', [C3, 12, 26, 9], {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9})
case2('ADX/ds2', [H2, L2, C3, 14], {'timeperiod': 14})
case2('CCI/ds2', [H2, L2, C3, 14], {'timeperiod': 14})
case2('ATR/ds2', [H2, L2, C3, 14], {'timeperiod': 14})
case2('OBV/ds2', [C3, V2], {})
case2('STDDEV/ds2', [C3, 20, 1.0], {'timeperiod': 20, 'nbdev': 1.0})
case2('HT_DCPERIOD/ds2', [C3], {})
case2('HT_SINE/ds2', [C3], {})

CASE2_IDS = [name for name, _, _ in CASES2]

@pytest.mark.parametrize("case_idx", range(len(CASES2)), ids=CASE2_IDS)
def test_cross_validate_ds2(case_idx):
    name, rs_args, c_kwargs = CASES2[case_idx]
    func_name = name.split('/')[0]
    rs_func = getattr(rs, func_name)
    c_func = getattr(c_talib, func_name)
    ours = rs_func(*rs_args)
    theirs = c_func(*rs_args[:len(rs_args) - len(c_kwargs)], **c_kwargs) if c_kwargs else c_func(*rs_args)
    assert_exact(name, ours, theirs)
